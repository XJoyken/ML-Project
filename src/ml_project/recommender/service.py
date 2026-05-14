from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_project.constants import PROCESSED_DATASET_PATH, POI_SOURCE_FILES, AIR_QUALITY_RAW_PATH, DEFAULT_EXPLANATION_CATEGORIES
from ml_project.data import load_listings
from ml_project.poi import load_poi_catalog
from ml_project.air import load_air_catalog
from ml_project.thresholds import classify_pm25, score_air_tier

from .mmr import build_similarity_matrix, mmr_select
from .scoring import (
    HARD_WEIGHT_THRESHOLD,
    CategoricalMatch,
    NumericMatch,
    NumericPreference,
    categorical_score,
    hard_filter_passes,
    score_numeric,
)

CATEGORICAL_COLUMNS = ("district", "house_type", "condition", "bathroom_type")
BOOLEAN_FEATURES = (
    "has_complex_id",
    "has_photo",
    "has_microdistrict",
    "is_first_floor",
    "is_last_floor",
)
CANDIDATE_POOL_MULTIPLIER = 8
MIN_CANDIDATE_POOL = 30
DIVERSITY_FEATURES = ("lat", "lon", "target_price_kzt", "rooms")

# Air priority is an additive tiebreaker, never a feature with its own weight.
# Max swing is ±AIR_BONUS_HALF_RANGE on the [0, 1] score, so it can only re-rank
# near-tied candidates and cannot override legitimate preference mismatches.
AIR_BONUS_HALF_RANGE = 0.05  # final_score += (air_score - 0.5) * (2 * 0.05) at most


@dataclass(slots=True)
class QueryTarget:
    feature: str
    preference: NumericPreference | None
    value: Any
    weight: float
    evidence: str | None = None
    kind: str = "numeric"  # numeric | categorical | boolean


@dataclass(slots=True)
class QueryPlan:
    numeric_targets: list[QueryTarget] = field(default_factory=list)
    categorical_targets: list[QueryTarget] = field(default_factory=list)
    boolean_targets: list[QueryTarget] = field(default_factory=list)
    excluded_districts: list[str] = field(default_factory=list)
    intent: str = "unknown"

    def is_empty(self) -> bool:
        # An exclusion alone is not enough to recommend — the user must give at least
        # one positive preference. Otherwise the recommender has nothing to score by.
        return not (self.numeric_targets or self.categorical_targets or self.boolean_targets)


@dataclass(slots=True)
class ScoredCandidate:
    index: int
    total_score: float
    base_score: float
    weight_sum: float
    numeric_matches: list[NumericMatch]
    categorical_matches: list[CategoricalMatch]
    hard_failures: list[str]
    air_pm25_cold_day: float | None = None
    air_pm25_warm_day: float | None = None
    air_score: float | None = None
    air_bonus: float = 0.0


class RecommendationService:
    def __init__(
        self,
        *,
        processed_path: Path = PROCESSED_DATASET_PATH,
    ):
        self.processed_path = processed_path
        self.features = pd.read_csv(processed_path)
        self.listings = load_listings()
        if len(self.features) != len(self.listings):
            raise ValueError(
                f"Feature matrix and listings disagree on row count: "
                f"{len(self.features)} != {len(self.listings)}"
            )

        self.numeric_columns = [
            column for column in self.features.columns if column not in CATEGORICAL_COLUMNS
        ]
        self.categorical_columns = [
            column for column in CATEGORICAL_COLUMNS if column in self.features.columns
        ]
        self.diversity_columns = [c for c in DIVERSITY_FEATURES if c in self.features.columns]
        self._diversity_matrix = self._build_diversity_matrix()
        
        self.poi_catalog = load_poi_catalog(POI_SOURCE_FILES)
        self.air_catalog = load_air_catalog(AIR_QUALITY_RAW_PATH)
        self.air_features = self.air_catalog.build_feature_frame(self.features)

    def recommend(
        self,
        features: Any,
        *,
        limit: int = 5,
        mmr_lambda: float = 0.7,
        prioritize_air_quality: bool = False,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), len(self.features)))
        plan = self._build_query_plan(features)
        if plan.is_empty():
            raise ValueError("Prompt did not map to any supported recommender features.")

        # Tiered hard-filter relaxation. We never drop the user's core requirements
        # (rooms / price / ЖК / district) just because a POI proximity filter was too
        # tight — that would surface 1-room apartments when the user asked for 2.
        #
        # Tier 1 — all hards (max strictness).
        # Tier 2 — drop POI distance hards, keep CORE_HARD_FEATURES.
        # Tier 3 — drop everything except district exclusion.
        candidate_mask = self._hard_filter_mask(plan, relax_level=0)
        candidate_indices = np.flatnonzero(candidate_mask.to_numpy())
        if len(candidate_indices) < limit:
            relaxed_mask = self._hard_filter_mask(plan, relax_level=1)
            candidate_indices = np.flatnonzero(relaxed_mask.to_numpy())
        if len(candidate_indices) < limit:
            relaxed_mask = self._hard_filter_mask(plan, relax_level=2)
            candidate_indices = np.flatnonzero(relaxed_mask.to_numpy())

        scored = self._score_candidates(plan, candidate_indices, prioritize_air_quality)
        if not scored:
            raise ValueError("No candidates left after scoring.")

        scored.sort(key=lambda c: c.total_score, reverse=True)
        pool_size = max(MIN_CANDIDATE_POOL, limit * CANDIDATE_POOL_MULTIPLIER)
        top_pool = scored[:pool_size]

        diversified = self._diversify(top_pool, k=limit, lambda_=mmr_lambda)
        return {
            "intent": plan.intent,
            "limit": limit,
            "matched_candidates": len(candidate_indices),
            "items": [self._build_item(candidate) for candidate in diversified],
            "plan": self._plan_payload(plan),
            "air_priority_active": bool(prioritize_air_quality),
        }

    def _build_query_plan(self, features: Any) -> QueryPlan:
        plan = QueryPlan(intent=str(getattr(features, "intent", "unknown") or "unknown"))

        for target in getattr(features, "numeric_targets", []) or []:
            feature = str(target.feature)
            if feature not in self.numeric_columns:
                continue
            plan.numeric_targets.append(
                QueryTarget(
                    feature=feature,
                    preference=str(target.preference),
                    value=float(target.value),
                    weight=float(target.weight),
                    evidence=getattr(target, "evidence", None),
                    kind="numeric",
                )
            )

        for target in getattr(features, "boolean_targets", []) or []:
            feature = str(target.feature)
            if feature not in self.numeric_columns:
                continue
            plan.boolean_targets.append(
                QueryTarget(
                    feature=feature,
                    preference="equal",
                    value=1.0 if bool(target.value) else 0.0,
                    weight=float(target.weight),
                    evidence=getattr(target, "evidence", None),
                    kind="boolean",
                )
            )

        for target in getattr(features, "categorical_targets", []) or []:
            feature = str(target.feature)
            if feature not in self.categorical_columns:
                continue
            plan.categorical_targets.append(
                QueryTarget(
                    feature=feature,
                    preference=None,
                    value=str(target.value),
                    weight=float(target.weight),
                    evidence=getattr(target, "evidence", None),
                    kind="categorical",
                )
            )

        for location in getattr(features, "location_preferences", []) or []:
            if str(location.kind) != "district" or "district" not in self.categorical_columns:
                continue
            value = str(location.value).strip()
            if not value:
                continue
            if str(location.preference) == "include":
                plan.categorical_targets.append(
                    QueryTarget(
                        feature="district",
                        preference=None,
                        value=value,
                        weight=float(location.weight),
                        evidence=getattr(location, "evidence", None),
                        kind="categorical",
                    )
                )
            elif str(location.preference) == "exclude":
                plan.excluded_districts.append(value)
        return plan

    # Features the user typically considers non-negotiable. Even when we relax POI
    # distances to widen the pool, we keep these strict so that a "двушка" query never
    # returns a "однушка".
    CORE_HARD_FEATURES = frozenset({
        "rooms",
        "target_price_kzt",
        "has_complex_id",
        "is_first_floor",
        "is_last_floor",
    })

    def _hard_filter_mask(self, plan: QueryPlan, *, relax_level: int = 0) -> pd.Series:
        """Return a boolean mask of candidate listings.

        relax_level:
          0 — apply ALL hard filters (weight ≥ HARD_WEIGHT_THRESHOLD).
          1 — drop POI distance hards (anything not in CORE_HARD_FEATURES). Keeps rooms,
              price, ЖК, floor, district categorical filters.
          2 — drop every hard filter; only district exclusions remain.
        """
        mask = pd.Series(True, index=self.features.index)

        # District exclusions are ALWAYS hard — even when the rest of the filters
        # are relaxed for graceful degradation, we never surface listings the user
        # explicitly rejected.
        if plan.excluded_districts and "district" in self.categorical_columns:
            excluded_lower = {d.strip().lower() for d in plan.excluded_districts if d.strip()}
            if excluded_lower:
                column = self.features["district"].astype("string").str.lower()
                mask &= ~column.isin(excluded_lower)

        if relax_level >= 2:
            return mask

        for target in plan.numeric_targets + plan.boolean_targets:
            if target.weight < HARD_WEIGHT_THRESHOLD:
                continue
            if relax_level >= 1 and target.feature not in self.CORE_HARD_FEATURES:
                continue
            actual = self.features[target.feature].to_numpy(dtype=float)
            keep = np.array(
                [
                    hard_filter_passes(
                        actual=value,
                        target=float(target.value),
                        preference=target.preference,  # type: ignore[arg-type]
                        feature=target.feature,
                    )
                    for value in actual
                ]
            )
            mask &= pd.Series(keep, index=self.features.index)

        for target in plan.categorical_targets:
            if target.weight < HARD_WEIGHT_THRESHOLD:
                continue
            if relax_level >= 1 and target.feature != "district":
                continue
            column = self.features[target.feature].astype("string").str.lower()
            mask &= column.eq(str(target.value).lower())
        return mask

    def _score_candidates(
        self,
        plan: QueryPlan,
        indices: np.ndarray,
        prioritize_air_quality: bool = False,
    ) -> list[ScoredCandidate]:
        results: list[ScoredCandidate] = []
        all_targets = plan.numeric_targets + plan.boolean_targets
        weight_sum = sum(target.weight for target in all_targets) + sum(
            target.weight for target in plan.categorical_targets
        )
        if weight_sum <= 0:
            weight_sum = 1.0

        feature_arrays = {
            target.feature: self.features[target.feature].to_numpy(dtype=float)
            for target in all_targets
        }
        categorical_arrays = {
            target.feature: self.features[target.feature].astype("string").to_numpy()
            for target in plan.categorical_targets
        }

        for index in indices:
            numeric_matches: list[NumericMatch] = []
            categorical_matches: list[CategoricalMatch] = []
            score_sum = 0.0
            hard_failures: list[str] = []

            for target in all_targets:
                actual_value = float(feature_arrays[target.feature][index])
                feature_score = score_numeric(
                    actual=actual_value,
                    target=float(target.value),
                    preference=target.preference,  # type: ignore[arg-type]
                    feature=target.feature,
                )
                passes_hard = hard_filter_passes(
                    actual=actual_value,
                    target=float(target.value),
                    preference=target.preference,  # type: ignore[arg-type]
                    feature=target.feature,
                )
                if target.weight >= HARD_WEIGHT_THRESHOLD and not passes_hard:
                    hard_failures.append(target.feature)
                score_sum += feature_score * target.weight
                numeric_matches.append(
                    NumericMatch(
                        feature=target.feature,
                        preference=target.preference,  # type: ignore[arg-type]
                        target=float(target.value),
                        actual=actual_value,
                        weight=target.weight,
                        score=feature_score,
                        hard_failed=not passes_hard and target.weight >= HARD_WEIGHT_THRESHOLD,
                    )
                )

            for target in plan.categorical_targets:
                actual_value = categorical_arrays[target.feature][index]
                actual_text = None if pd.isna(actual_value) else str(actual_value)
                cat_score = categorical_score(actual_text, str(target.value))
                if target.weight >= HARD_WEIGHT_THRESHOLD and cat_score < 1.0:
                    hard_failures.append(target.feature)
                score_sum += cat_score * target.weight
                categorical_matches.append(
                    CategoricalMatch(
                        feature=target.feature,
                        target=str(target.value),
                        actual=actual_text,
                        weight=target.weight,
                        score=cat_score,
                        hard_failed=cat_score < 1.0 and target.weight >= HARD_WEIGHT_THRESHOLD,
                    )
                )

            base_score = score_sum / weight_sum

            pm25_cold = float(self.air_features["air_pm25_cold_day"].iloc[index])
            pm25_warm = float(self.air_features["air_pm25_warm_day"].iloc[index])
            air_score: float | None = None
            air_bonus = 0.0
            if not np.isnan(pm25_cold):
                tier = classify_pm25(pm25_cold)
                air_score = score_air_tier(tier)
            if prioritize_air_quality and air_score is not None:
                # Tiebreaker only: a perfect-air listing gets +AIR_BONUS_HALF_RANGE,
                # a very polluted one gets -AIR_BONUS_HALF_RANGE. The 0.1 total swing
                # cannot promote a 0.7-score candidate above a 0.9-score one.
                air_bonus = (air_score - 0.5) * 2.0 * AIR_BONUS_HALF_RANGE

            total_score = base_score + air_bonus
            results.append(
                ScoredCandidate(
                    index=int(index),
                    total_score=float(total_score),
                    base_score=float(base_score),
                    weight_sum=float(weight_sum),
                    numeric_matches=numeric_matches,
                    categorical_matches=categorical_matches,
                    hard_failures=hard_failures,
                    air_pm25_cold_day=None if np.isnan(pm25_cold) else float(pm25_cold),
                    air_pm25_warm_day=None if np.isnan(pm25_warm) else float(pm25_warm),
                    air_score=air_score,
                    air_bonus=float(air_bonus),
                )
            )
        return results

    def _diversify(
        self,
        candidates: list[ScoredCandidate],
        *,
        k: int,
        lambda_: float,
    ) -> list[ScoredCandidate]:
        if len(candidates) <= k:
            return candidates
        scores = np.array([c.total_score for c in candidates], dtype=float)
        pool_indices = [c.index for c in candidates]
        diversity_subset = self._diversity_matrix[pool_indices]
        similarity = build_similarity_matrix(diversity_subset)
        chosen = mmr_select(
            scores=scores,
            similarity_matrix=similarity,
            k=k,
            lambda_=lambda_,
        )
        return [candidates[i] for i in chosen]

    def _build_diversity_matrix(self) -> np.ndarray:
        if not self.diversity_columns:
            return np.zeros((len(self.features), 1), dtype=np.float32)
        raw = self.features[self.diversity_columns].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=0.0)
        means = raw.mean(axis=0, keepdims=True)
        stds = raw.std(axis=0, keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        standardised = (raw - means) / stds
        return standardised.astype(np.float32)

    def _build_item(self, candidate: ScoredCandidate) -> dict[str, Any]:
        listing = self.listings.iloc[candidate.index]
        lat = _optional_float(listing.get("lat"))
        lon = _optional_float(listing.get("lon"))

        nearby_pois = []
        nearest_air = None
        if lat is not None and lon is not None:
            nearby_pois = self.poi_catalog.nearest_many(lat=lat, lon=lon, categories=DEFAULT_EXPLANATION_CATEGORIES)
            nearest_air = self.air_catalog.nearest_for(lat=lat, lon=lon)

        return {
            "listing_id": _optional_str(listing.get("listing_id")),
            "match_score": round(candidate.total_score, 4),
            "base_score": round(candidate.base_score, 4),
            "air_bonus": round(candidate.air_bonus, 4),
            "air_score": (
                None if candidate.air_score is None else round(candidate.air_score, 4)
            ),
            "air_pm25_cold_day": candidate.air_pm25_cold_day,
            "air_pm25_warm_day": candidate.air_pm25_warm_day,
            "price_kzt": _optional_float(listing.get("target_price_kzt")),
            "rooms": _optional_float(listing.get("rooms")),
            "area_m2": _optional_float(listing.get("area_m2")),
            "floor_current": _optional_float(listing.get("floor_current")),
            "floors_total": _optional_float(listing.get("floors_total")),
            "year_built": _optional_float(listing.get("year_built")),
            "condition": _optional_str(listing.get("condition")),
            "house_type": _optional_str(listing.get("house_type")),
            "ceiling_height_m": _optional_float(listing.get("ceiling_height_m")),
            "dist_to_center_km": _optional_float(listing.get("dist_to_center_km")),
            "district": _optional_str(listing.get("district")),
            "microdistrict": _optional_str(listing.get("microdistrict")),
            "lat": lat,
            "lon": lon,
            "hard_failures": list(candidate.hard_failures),
            "matches": [
                _numeric_match_payload(match)
                for match in candidate.numeric_matches
            ]
            + [
                _categorical_match_payload(match)
                for match in candidate.categorical_matches
            ],
            "description": _optional_str(listing.get("description")),
            "nearby_pois": nearby_pois,
            "nearest_air_sensor": nearest_air,
        }

    def _plan_payload(self, plan: QueryPlan) -> dict[str, Any]:
        return {
            "intent": plan.intent,
            "numeric_targets": [
                {
                    "feature": t.feature,
                    "preference": t.preference,
                    "value": t.value,
                    "weight": t.weight,
                    "evidence": t.evidence,
                }
                for t in plan.numeric_targets + plan.boolean_targets
            ],
            "categorical_targets": [
                {
                    "feature": t.feature,
                    "value": t.value,
                    "weight": t.weight,
                    "evidence": t.evidence,
                }
                for t in plan.categorical_targets
            ],
            "excluded_districts": list(plan.excluded_districts),
        }


def _numeric_match_payload(match: NumericMatch) -> dict[str, Any]:
    return {
        "kind": "numeric",
        "feature": match.feature,
        "preference": match.preference,
        "target": match.target,
        "actual": match.actual,
        "weight": match.weight,
        "score": round(match.score, 4),
        "hard_failed": match.hard_failed,
    }


def _categorical_match_payload(match: CategoricalMatch) -> dict[str, Any]:
    return {
        "kind": "categorical",
        "feature": match.feature,
        "target": match.target,
        "actual": match.actual,
        "weight": match.weight,
        "score": round(match.score, 4),
        "hard_failed": match.hard_failed,
    }


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


# Backwards-compatible alias (older imports still reference this name).
KnnRecommendationService = RecommendationService
