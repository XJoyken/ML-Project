from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .constants import (
    AIR_PM25_MAX_PLAUSIBLE,
    AIR_QUALITY_RAW_PATH,
    EARTH_RADIUS_KM,
)

COLD_MONTHS = {9, 10, 11, 12, 1, 2}
WARM_MONTHS = {3, 4, 5, 6, 7, 8}
NIGHT_HOURS = range(0, 6)
DAY_HOURS = range(12, 18)

AIR_PM25_FEATURES = [
    "air_pm25_cold_night",
    "air_pm25_cold_day",
    "air_pm25_warm_night",
    "air_pm25_warm_day",
]
AIR_FEATURE_COLUMNS = [*AIR_PM25_FEATURES, "air_nearest_dist_m"]


class AirQualityCatalog:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)
        self._tree: BallTree | None = None

    def build_feature_frame(self, listings: pd.DataFrame) -> pd.DataFrame:
        if self.frame.empty:
            return pd.DataFrame(
                {column: np.nan for column in AIR_FEATURE_COLUMNS},
                index=listings.index,
                dtype=np.float32,
            )

        coordinates_rad = np.radians(listings[["lat", "lon"]].to_numpy())
        distances, indices = self._get_tree().query(coordinates_rad, k=1)
        nearest = self.frame.iloc[indices[:, 0]].reset_index(drop=True)

        features = pd.DataFrame(index=listings.index)
        for column in AIR_PM25_FEATURES:
            features[column] = nearest[column].astype(np.float32).to_numpy()
        features["air_nearest_dist_m"] = (
            distances[:, 0] * EARTH_RADIUS_KM * 1000
        ).astype(np.float32)
        return features

    def nearest_for(self, lat: float, lon: float) -> dict[str, Any] | None:
        if self.frame.empty:
            return None
        coords = np.radians(np.array([[lat, lon]], dtype=float))
        distances, indices = self._get_tree().query(coords, k=1)
        row = self.frame.iloc[int(indices[0, 0])]
        return {
            "location_id": str(row["location_id"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "distance_m": float(distances[0, 0] * EARTH_RADIUS_KM * 1000),
            "pm25_cold_night": optional_float(row["air_pm25_cold_night"]),
            "pm25_cold_day": optional_float(row["air_pm25_cold_day"]),
            "pm25_warm_night": optional_float(row["air_pm25_warm_night"]),
            "pm25_warm_day": optional_float(row["air_pm25_warm_day"]),
        }

    def _get_tree(self) -> BallTree:
        if self._tree is None:
            self._tree = BallTree(
                np.radians(self.frame[["lat", "lon"]].to_numpy()),
                metric="haversine",
            )
        return self._tree


def load_air_catalog(raw_path: Path | None = None) -> AirQualityCatalog:
    return AirQualityCatalog(build_air_aggregate(raw_path or AIR_QUALITY_RAW_PATH))


def build_air_aggregate(raw_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        raw_path,
        usecols=["location_id", "latitude", "longitude", "hour", "month", "pm25"],
    )
    raw = raw.dropna(subset=["pm25"])
    raw = raw[(raw["pm25"] > 0) & (raw["pm25"] <= AIR_PM25_MAX_PLAUSIBLE)]

    raw["season"] = raw["month"].map(season_label)
    raw["time_of_day"] = raw["hour"].map(time_of_day_label)
    raw = raw.dropna(subset=["season", "time_of_day"])

    aggregated = (
        raw.groupby(["location_id", "season", "time_of_day"])["pm25"]
        .mean()
        .unstack(["season", "time_of_day"])
    )
    aggregated.columns = [f"air_pm25_{season}_{tod}" for season, tod in aggregated.columns]
    for column in AIR_PM25_FEATURES:
        if column not in aggregated.columns:
            aggregated[column] = np.nan

    coords = (
        raw.groupby("location_id")[["latitude", "longitude"]]
        .first()
        .rename(columns={"latitude": "lat", "longitude": "lon"})
    )

    frame = aggregated[AIR_PM25_FEATURES].join(coords).reset_index()
    return frame.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def season_label(month: int) -> str | None:
    if month in COLD_MONTHS:
        return "cold"
    if month in WARM_MONTHS:
        return "warm"
    return None


def time_of_day_label(hour: int) -> str | None:
    if hour in NIGHT_HOURS:
        return "night"
    if hour in DAY_HOURS:
        return "day"
    return None


def optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
