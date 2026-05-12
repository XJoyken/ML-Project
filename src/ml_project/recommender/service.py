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
    intent: str = "unknown"

    def is_empty(self) -> bool:
        return not (self.numeric_targets or self.categorical_targets or self.boolean_targets)


@dataclass(slots=True)
class ScoredCandidate:
    index: int
    total_score: float
    weight_sum: float
    numeric_matches: list[NumericMatch]
    categorical_matches: list[CategoricalMatch]
    hard_failures: list[str]


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

    def recommend(
        self,
        features: Any,
        *,
        limit: int = 5,
        mmr_lambda: float = 0.7,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), len(self.features)))
        plan = self._build_query_plan(features)
        if plan.is_empty():
            raise ValueError("Prompt did not map to any supported recommender features.")

        candidate_mask = self._hard_filter_mask(plan)
        candidate_indices = np.flatnonzero(candidate_mask.to_numpy())

        if len(candidate_indices) < limit:
            # Fall back to full pool if hard filters were too restrictive.
            relaxed_mask = self._hard_filter_mask(plan, drop_hard=True)
            candidate_indices = np.flatnonzero(relaxed_mask.to_numpy())

        scored = self._score_candidates(plan, candidate_indices)
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
            if (
                str(location.kind) == "district"
                and str(location.preference) == "include"
                and "district" in self.categorical_columns
            ):
                plan.categorical_targets.append(
                    QueryTarget(
                        feature="district",
                        preference=None,
                        value=str(location.value),
                        weight=float(location.weight),
                        evidence=getattr(location, "evidence", None),
                        kind="categorical",
                    )
                )
        return plan

    def _hard_filter_mask(self, plan: QueryPlan, *, drop_hard: bool = False) -> pd.Series:
        mask = pd.Series(True, index=self.features.index)
        if drop_hard:
            return mask

        for target in plan.numeric_targets + plan.boolean_targets:
            if target.weight < HARD_WEIGHT_THRESHOLD:
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
            column = self.features[target.feature].astype("string").str.lower()
            mask &= column.eq(str(target.value).lower())
        return mask

    def _score_candidates(
        self,
        plan: QueryPlan,
        indices: np.ndarray,
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

            total_score = score_sum / weight_sum
            results.append(
                ScoredCandidate(
                    index=int(index),
                    total_score=float(total_score),
                    weight_sum=float(weight_sum),
                    numeric_matches=numeric_matches,
                    categorical_matches=categorical_matches,
                    hard_failures=hard_failures,
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
            "price_kzt": _optional_float(listing.get("target_price_kzt")),
            "rooms": _optional_float(listing.get("rooms")),
            "area_m2": _optional_float(listing.get("area_m2")),
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
