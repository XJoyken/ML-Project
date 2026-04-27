from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .constants import (
    DEFAULT_EXPLANATION_CATEGORIES,
    EARTH_RADIUS_KM,
    POI_SOURCE_FILES,
)

POI_ID_COLUMNS = ("poi_id", "firm_id", "id")
POI_NAME_COLUMNS = ("name", "org_name")
POI_ADDRESS_COLUMNS = ("address", "listing_address_text")

NEAREST_POI_COUNT = 5
MEAN_DISTANCE_COUNTS = (3, 5)
RADIUS_COUNT_METERS = (500, 1000)


class PoiCatalog:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.trees: dict[str, BallTree] = {}

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(self.frames.keys())

    def build_feature_frame(self, listings: pd.DataFrame) -> pd.DataFrame:
        coordinates_rad = coordinates_to_radians(listings)
        feature_frames = [
            self._build_category_features(category, frame, coordinates_rad, listings.index)
            for category, frame in self.frames.items()
            if not frame.empty
        ]

        if not feature_frames:
            return pd.DataFrame(index=listings.index)
        return pd.concat(feature_frames, axis=1)

    def nearest_many(
        self,
        *,
        lat: float,
        lon: float,
        categories: tuple[str, ...] = DEFAULT_EXPLANATION_CATEGORIES,
    ) -> list[dict[str, Any]]:
        coordinates_rad = np.radians(np.array([[lat, lon]], dtype=float))
        nearby = []

        for category in categories:
            frame = self.frames.get(category)
            if frame is None or frame.empty:
                continue

            tree = self._get_tree(category)
            distances, indices = tree.query(coordinates_rad, k=1)
            matched = frame.iloc[int(indices[0, 0])]
            nearby.append(
                {
                    "category": category,
                    "poi_id": str(matched["poi_id"]),
                    "name": str(matched["name"]),
                    "address": optional_str(matched.get("address")),
                    "distance_m": float(distances[0, 0] * EARTH_RADIUS_KM * 1000),
                    "lat": float(matched["lat"]),
                    "lon": float(matched["lon"]),
                }
            )

        return nearby

    def _build_category_features(
        self,
        category: str,
        frame: pd.DataFrame,
        coordinates_rad: np.ndarray,
        index: pd.Index,
    ) -> pd.DataFrame:
        tree = self._get_tree(category)
        neighbor_count = min(NEAREST_POI_COUNT, len(frame))
        distances_km = query_distances_km(tree, coordinates_rad, neighbor_count)

        features = pd.DataFrame(index=index)
        features[f"{category}_dist_km_1"] = distances_km[:, 0].astype(np.float32)

        for count in MEAN_DISTANCE_COUNTS:
            mean_distances = distances_km[:, : min(count, neighbor_count)].mean(axis=1)
            features[f"{category}_mean_dist_km_{count}"] = mean_distances.astype(np.float32)

        for radius_m in RADIUS_COUNT_METERS:
            features[f"{category}_count_{radius_m}m"] = count_within_radius(
                tree,
                coordinates_rad,
                radius_m,
            )

        return features

    def _get_tree(self, category: str) -> BallTree:
        if category not in self.trees:
            self.trees[category] = BallTree(
                coordinates_to_radians(self.frames[category]),
                metric="haversine",
            )
        return self.trees[category]


def load_poi_catalog(
    poi_sources: dict[str, Path] | None = None,
) -> PoiCatalog:
    frames: dict[str, pd.DataFrame] = {}

    for category, path in (poi_sources or POI_SOURCE_FILES).items():
        if not path.exists():
            continue

        frame = pd.read_csv(path)
        if frame.empty:
            continue

        normalized = normalize_poi_frame(frame)
        if not normalized.empty:
            frames[category] = normalized

    return PoiCatalog(frames)


def normalize_poi_frame(frame: pd.DataFrame) -> pd.DataFrame:
    poi_id_column = first_existing(frame, POI_ID_COLUMNS)
    name_column = first_existing(frame, POI_NAME_COLUMNS)
    address_column = first_existing(frame, POI_ADDRESS_COLUMNS)

    normalized = pd.DataFrame(
        {
            "poi_id": text_or_empty(frame, poi_id_column),
            "name": text_or_empty(frame, name_column),
            "address": text_or_empty(frame, address_column),
            "lat": pd.to_numeric(frame.get("lat"), errors="coerce"),
            "lon": pd.to_numeric(frame.get("lon"), errors="coerce"),
        }
    ).dropna(subset=["poi_id", "name", "lat", "lon"])

    return normalized.drop_duplicates(subset=["lat", "lon"]).reset_index(drop=True)


def coordinates_to_radians(frame: pd.DataFrame) -> np.ndarray:
    return np.radians(frame[["lat", "lon"]].to_numpy())


def query_distances_km(
    tree: BallTree,
    coordinates_rad: np.ndarray,
    neighbor_count: int,
) -> np.ndarray:
    distances, _ = tree.query(coordinates_rad, k=neighbor_count)
    return distances * EARTH_RADIUS_KM


def count_within_radius(
    tree: BallTree,
    coordinates_rad: np.ndarray,
    radius_m: int,
) -> np.ndarray:
    radius_rad = (radius_m / 1000) / EARTH_RADIUS_KM
    return tree.query_radius(
        coordinates_rad,
        r=radius_rad,
        count_only=True,
    ).astype(np.int16)


def text_or_empty(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[column].astype("string")


def first_existing(frame: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
