from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project.constants import PROCESSED_DATASET_PATH
from ml_project.data import load_listings

CATEGORICAL_COLUMNS = ("district", "house_type", "condition", "bathroom_type")
DEFAULT_LIMIT = 5


@dataclass(slots=True)
class QueryVector:
    numeric_values: dict[str, float]
    numeric_preferences: dict[str, str]
    numeric_weights: dict[str, float]
    categorical_values: dict[str, str]
    categorical_weights: dict[str, float]


class KnnRecommendationService:
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
                "Processed feature matrix and listing metadata must have the same row count: "
                f"{len(self.features)} != {len(self.listings)}"
            )

        self.numeric_columns = [
            column for column in self.features.columns if column not in CATEGORICAL_COLUMNS
        ]
        self.categorical_columns = [
            column for column in CATEGORICAL_COLUMNS if column in self.features.columns
        ]
        self.numeric_means = self.features[self.numeric_columns].mean(numeric_only=True)
        self.scaler = StandardScaler()
        numeric_matrix = self.scaler.fit_transform(
            self.features[self.numeric_columns].fillna(self.numeric_means)
        )

        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_matrix = self.encoder.fit_transform(
            self.features[self.categorical_columns].fillna("unknown").astype(str)
        )

        self.matrix = np.hstack([numeric_matrix, categorical_matrix]).astype(np.float32)

    def recommend(self, features: Any, *, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), len(self.features)))
        query = self._build_query(features)
        if not query.numeric_values and not query.categorical_values:
            raise ValueError("Prompt did not map to any supported recommender features.")
        query_matrix, selected_indices, weights = self._vectorize_query(query)
        if not selected_indices:
            raise ValueError("Prompt did not map to known recommender feature columns.")

        candidate_indices = self._candidate_indices(query, limit)
        candidate_matrix = self.matrix[candidate_indices][:, selected_indices] * weights
        selected_query = query_matrix[:, selected_indices] * weights

        knn = NearestNeighbors(metric="cosine", algorithm="brute")
        knn.fit(candidate_matrix)
        distances, local_indices = knn.kneighbors(
            selected_query,
            n_neighbors=min(limit, len(candidate_indices)),
        )
        indices = candidate_indices[local_indices[0]]

        return [
            self._build_item(int(index), float(1.0 - distance), query)
            for distance, index in zip(distances[0], indices, strict=True)
        ]

    def _build_query(self, features: Any) -> QueryVector:
        numeric_values: dict[str, float] = {}
        numeric_preferences: dict[str, str] = {}
        numeric_weights: dict[str, float] = {}
        categorical_values: dict[str, str] = {}
        categorical_weights: dict[str, float] = {}

        for target in getattr(features, "numeric_targets", []):
            feature = str(target.feature)
            if feature in self.numeric_columns:
                numeric_values[feature] = float(target.value)
                numeric_preferences[feature] = str(target.preference)
                numeric_weights[feature] = float(target.weight)

        for target in getattr(features, "boolean_targets", []):
            feature = str(target.feature)
            if feature in self.numeric_columns:
                numeric_values[feature] = 1.0 if bool(target.value) else 0.0
                numeric_preferences[feature] = "equal"
                numeric_weights[feature] = float(target.weight)

        for target in getattr(features, "categorical_targets", []):
            feature = str(target.feature)
            if feature in self.categorical_columns:
                categorical_values[feature] = str(target.value)
                categorical_weights[feature] = float(target.weight)

        for location in getattr(features, "location_preferences", []):
            if (
                str(location.kind) == "district"
                and str(location.preference) == "include"
                and "district" in self.categorical_columns
            ):
                categorical_values["district"] = str(location.value)
                categorical_weights["district"] = float(location.weight)

        return QueryVector(
            numeric_values=numeric_values,
            numeric_preferences=numeric_preferences,
            numeric_weights=numeric_weights,
            categorical_values=categorical_values,
            categorical_weights=categorical_weights,
        )

    def _vectorize_query(self, query: QueryVector) -> tuple[np.ndarray, list[int], np.ndarray]:
        numeric_row = self.numeric_means.to_frame().T
        for column, value in query.numeric_values.items():
            numeric_row[column] = value
        numeric_matrix = self.scaler.transform(numeric_row[self.numeric_columns])

        categorical_matrix = np.zeros(
            (1, sum(len(categories) for categories in self.encoder.categories_)),
            dtype=np.float32,
        )
        selected_indices: list[int] = []
        weights: list[float] = []

        for column in query.numeric_values:
            selected_indices.append(self.numeric_columns.index(column))
            weights.append(query.numeric_weights.get(column, 1.0))

        categorical_offset = len(self.numeric_columns)
        offset = 0
        for column, categories in zip(
            self.categorical_columns,
            self.encoder.categories_,
            strict=True,
        ):
            value = query.categorical_values.get(column)
            if value is not None:
                matches = np.where(categories == value)[0]
                if len(matches):
                    categorical_matrix[0, offset + int(matches[0])] = 1.0
                    group_indices = list(
                        range(
                            categorical_offset + offset,
                            categorical_offset + offset + len(categories),
                        )
                    )
                    selected_indices.extend(group_indices)
                    weights.extend([query.categorical_weights.get(column, 1.0)] * len(group_indices))
            offset += len(categories)

        matrix = np.hstack([numeric_matrix, categorical_matrix]).astype(np.float32)
        return matrix, selected_indices, np.array(weights, dtype=np.float32)

    def _candidate_indices(self, query: QueryVector, limit: int) -> np.ndarray:
        mask = pd.Series(True, index=self.features.index)

        for feature, value in query.numeric_values.items():
            preference = query.numeric_preferences.get(feature)
            if feature == "target_price_kzt" and preference == "at_most":
                mask &= self.features[feature].le(value)
            elif feature == "target_price_kzt" and preference == "at_least":
                mask &= self.features[feature].ge(value)
            elif feature == "rooms" and preference == "equal":
                mask &= self.features[feature].round().eq(round(value))
            elif feature.startswith("has_") or feature.startswith("is_"):
                mask &= self.features[feature].round().eq(round(value))

        for feature, value in query.categorical_values.items():
            mask &= self.features[feature].astype(str).eq(value)

        if int(mask.sum()) >= limit:
            return np.flatnonzero(mask.to_numpy())
        return np.arange(len(self.features))

    def _build_item(
        self,
        index: int,
        similarity: float,
        query: QueryVector,
    ) -> dict[str, Any]:
        listing = self.listings.iloc[index]
        feature_row = self.features.iloc[index]
        return {
            "listing_id": optional_str(listing.get("listing_id")),
            "similarity": round(similarity, 6),
            "price_kzt": optional_float(listing.get("target_price_kzt")),
            "rooms": optional_float(listing.get("rooms")),
            "area_m2": optional_float(listing.get("area_m2")),
            "district": optional_str(listing.get("district")),
            "microdistrict": optional_str(listing.get("microdistrict")),
            "lat": optional_float(listing.get("lat")),
            "lon": optional_float(listing.get("lon")),
            "reasons": self._build_reasons(feature_row, query),
        }

    def _build_reasons(self, row: pd.Series, query: QueryVector) -> list[str]:
        reasons = []
        for feature, target_value in query.numeric_values.items():
            actual_value = optional_float(row.get(feature))
            if actual_value is not None:
                reasons.append(format_numeric_reason(feature, actual_value, target_value))

        for feature, target_value in query.categorical_values.items():
            actual_value = optional_str(row.get(feature))
            if actual_value is not None:
                reasons.append(f"{feature}: {actual_value} похож на запрос {target_value}")

        return reasons[:5]


def optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def format_numeric_reason(feature: str, actual_value: float, target_value: float) -> str:
    if feature.endswith("_nearest_dist_m"):
        label = feature.removesuffix("_nearest_dist_m")
        return f"{label}: {actual_value:.0f} м при запросе около {target_value:.0f} м"
    if feature.endswith("_count_500m") or feature.endswith("_count_1000m"):
        return f"{feature}: {actual_value:.0f} при запросе {target_value:.0f}"
    if feature == "target_price_kzt":
        return f"цена: {actual_value:,.0f} KZT при запросе {target_value:,.0f} KZT"
    if feature == "has_complex_id":
        return "есть ЖК" if actual_value >= 0.5 else "без ЖК"
    return f"{feature}: {actual_value:.2f} при запросе {target_value:.2f}"
