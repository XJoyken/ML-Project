from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .constants import (
    ADS_COLUMNS,
    CATEGORICAL_FEATURES,
    DATA_DIR,
    DEFAULT_EXPLANATION_CATEGORIES,
    EARTH_RADIUS_KM,
    POI_SOURCE_FILES,
    TARGET_COLUMN,
    TARGET_LOG_COLUMN,
)


class PoiCatalog:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.trees: dict[str, BallTree] = {}

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(self.frames.keys())

    def build_feature_frame(self, listings: pd.DataFrame) -> pd.DataFrame:
        coordinates_rad = np.radians(listings[["lat", "lon"]].to_numpy())
        feature_frames = []

        for category, frame in self.frames.items():
            if frame.empty:
                continue
            tree = self._get_tree(category)
            neighbors_count = min(5, len(frame))
            distances, _ = tree.query(coordinates_rad, k=neighbors_count)
            distances_km = distances * EARTH_RADIUS_KM

            features = pd.DataFrame(index=listings.index)
            features[f"{category}_dist_km_1"] = distances_km[:, 0].astype(np.float32)
            features[f"{category}_mean_dist_km_3"] = (
                distances_km[:, : min(3, neighbors_count)].mean(axis=1).astype(np.float32)
            )
            features[f"{category}_mean_dist_km_5"] = distances_km.mean(axis=1).astype(
                np.float32
            )
            features[f"{category}_count_500m"] = tree.query_radius(
                coordinates_rad,
                r=0.5 / EARTH_RADIUS_KM,
                count_only=True,
            ).astype(np.int16)
            features[f"{category}_count_1000m"] = tree.query_radius(
                coordinates_rad,
                r=1.0 / EARTH_RADIUS_KM,
                count_only=True,
            ).astype(np.int16)
            feature_frames.append(features)

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

    def _get_tree(self, category: str) -> BallTree:
        if category not in self.trees:
            coordinates_rad = np.radians(self.frames[category][["lat", "lon"]].to_numpy())
            self.trees[category] = BallTree(coordinates_rad, metric="haversine")
        return self.trees[category]


def load_listings(ads_path: Path | None = None) -> pd.DataFrame:
    ads_path = ads_path or (DATA_DIR / "ads.csv")
    frame = pd.read_csv(ads_path, usecols=ADS_COLUMNS, low_memory=False)
    frame = frame.loc[frame["listing_category_alias"].eq("kvartiry")].copy()
    frame = frame.rename(
        columns={
            "listing_price_kzt": TARGET_COLUMN,
            "listing_square_m2": "area_m2",
            "listing_rooms": "rooms",
            "location_lat": "lat",
            "location_lon": "lon",
            "address_district": "district",
            "address_microdistrict": "microdistrict",
            "listing_complex_id": "complex_id",
            "summary_Год_постройки": "year_built",
            "summary_Этаж": "floor_text",
            "summary_Тип_дома": "house_type",
            "summary_Состояние_квартиры": "condition",
            "summary_Высота_потолков": "ceiling_height_text",
            "summary_Санузел": "bathroom_type",
            "listing_has_photo": "has_photo",
            "listing_photo_count": "photo_count",
        }
    )

    frame = normalize_listing_frame(frame, fit_mode=True)
    frame = frame.loc[
        frame[TARGET_COLUMN].gt(0)
        & frame["area_m2"].gt(0)
        & frame["rooms"].notna()
        & frame["lat"].between(43.0, 44.0)
        & frame["lon"].between(76.0, 77.2)
    ].copy()
    return frame.reset_index(drop=True)


def get_listing_input(
    listing_id: str | int,
    *,
    listings: pd.DataFrame | None = None,
    ads_path: Path | None = None,
) -> dict[str, Any]:
    listings = load_listings(ads_path=ads_path) if listings is None else listings
    matched = listings.loc[listings["listing_id"].astype("string").eq(str(listing_id))]
    if matched.empty:
        raise KeyError(f"Listing with id '{listing_id}' was not found.")
    return matched.iloc[0].to_dict()


def load_poi_catalog(
    poi_sources: dict[str, Path] | None = None,
) -> PoiCatalog:
    poi_sources = poi_sources or POI_SOURCE_FILES
    frames: dict[str, pd.DataFrame] = {}

    for category, path in poi_sources.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue

        poi_id_column = first_existing(frame, "poi_id", "firm_id", "id")
        name_column = first_existing(frame, "name", "org_name")
        address_column = first_existing(frame, "address", "listing_address_text")

        normalized = pd.DataFrame(
            {
                "poi_id": (
                    frame[poi_id_column].astype("string")
                    if poi_id_column
                    else pd.Series(pd.NA, index=frame.index, dtype="string")
                ),
                "name": (
                    frame[name_column].astype("string")
                    if name_column
                    else pd.Series(pd.NA, index=frame.index, dtype="string")
                ),
                "address": (
                    frame[address_column].astype("string")
                    if address_column
                    else pd.Series(pd.NA, index=frame.index, dtype="string")
                ),
                "lat": pd.to_numeric(frame.get("lat"), errors="coerce"),
                "lon": pd.to_numeric(frame.get("lon"), errors="coerce"),
            }
        ).dropna(subset=["poi_id", "name", "lat", "lon"])

        frames[category] = normalized.drop_duplicates(subset=["lat", "lon"]).reset_index(
            drop=True
        )

    return PoiCatalog(frames)


def normalize_listing_frame(
    frame: pd.DataFrame,
    *,
    reference_ads: pd.DataFrame | None = None,
    fit_mode: bool = False,
) -> pd.DataFrame:
    canonical = frame.copy()
    ensure_columns(
        canonical,
        [
            "listing_id",
            TARGET_COLUMN,
            "area_m2",
            "rooms",
            "lat",
            "lon",
            "district",
            "microdistrict",
            "complex_id",
            "year_built",
            "floor_text",
            "floor_current",
            "floors_total",
            "house_type",
            "condition",
            "ceiling_height_text",
            "ceiling_height_m",
            "bathroom_type",
            "has_photo",
            "photo_count",
            "scraped_at",
            "listing_added_at",
            "days_since_added_to_scrape",
        ],
    )

    for column in [TARGET_COLUMN, "area_m2", "rooms", "lat", "lon", "photo_count", "year_built"]:
        canonical[column] = pd.to_numeric(canonical[column], errors="coerce")

    if canonical["ceiling_height_m"].isna().all():
        canonical["ceiling_height_m"] = extract_numeric(canonical["ceiling_height_text"])
    else:
        canonical["ceiling_height_m"] = pd.to_numeric(
            canonical["ceiling_height_m"],
            errors="coerce",
        )

    if canonical[["floor_current", "floors_total"]].isna().all(axis=None):
        floor_parts = parse_floor_columns(canonical["floor_text"])
        canonical[["floor_current", "floors_total"]] = floor_parts
    else:
        canonical["floor_current"] = pd.to_numeric(
            canonical["floor_current"],
            errors="coerce",
        )
        canonical["floors_total"] = pd.to_numeric(
            canonical["floors_total"],
            errors="coerce",
        )

    canonical["scraped_at"] = pd.to_datetime(
        canonical["scraped_at"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)
    canonical["listing_added_at"] = pd.to_datetime(
        canonical["listing_added_at"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    if canonical["days_since_added_to_scrape"].isna().all():
        canonical["days_since_added_to_scrape"] = (
            canonical["scraped_at"] - canonical["listing_added_at"]
        ).dt.days
    canonical["days_since_added_to_scrape"] = pd.to_numeric(
        canonical["days_since_added_to_scrape"],
        errors="coerce",
    ).fillna(0.0)

    has_photo = canonical["has_photo"]
    if has_photo.dtype == "object" or str(has_photo.dtype).startswith("string"):
        has_photo = binary_from_text(has_photo)
    canonical["has_photo"] = pd.to_numeric(has_photo, errors="coerce").fillna(
        (canonical["photo_count"].fillna(0) > 0).astype(float)
    ).astype(float)

    canonical["has_complex_id"] = canonical["complex_id"].notna().astype(np.int8)
    canonical["has_microdistrict"] = canonical["microdistrict"].notna().astype(np.int8)

    if fit_mode or reference_ads is None:
        complex_counts = canonical["complex_id"].value_counts(dropna=True).to_dict()
    else:
        complex_counts = reference_ads["complex_id"].value_counts(dropna=True).to_dict()
    canonical["complex_listing_count"] = (
        canonical["complex_id"].map(complex_counts).fillna(0).astype(np.int32)
    )

    for column in CATEGORICAL_FEATURES:
        canonical[column] = canonical[column].fillna("unknown").astype("string")

    canonical[TARGET_LOG_COLUMN] = np.log1p(canonical[TARGET_COLUMN])
    return canonical.reset_index(drop=True)


def validate_inference_frame(frame: pd.DataFrame) -> None:
    required = {
        "area_m2": "area_m2",
        "rooms": "rooms",
        "lat": "lat",
        "lon": "lon",
        "district": "district",
        "house_type": "house_type",
        "condition": "condition",
        "bathroom_type": "bathroom_type",
        TARGET_COLUMN: "listing_price_kzt",
    }
    missing = [
        public_name
        for column, public_name in required.items()
        if frame[column].isna().any()
    ]
    if missing:
        raise ValueError(f"Missing required listing fields: {', '.join(missing)}")


def extract_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", ".", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_floor_columns(series: pd.Series) -> pd.DataFrame:
    parts = series.astype("string").str.extract(
        r"(?P<floor_current>\d+)\s*из\s*(?P<floors_total>\d+)"
    )
    return parts.apply(pd.to_numeric, errors="coerce")


def binary_from_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "true": 1.0,
            "false": 0.0,
            "yes": 1.0,
            "no": 0.0,
            "да": 1.0,
            "нет": 0.0,
        }
    )


def ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA


def first_existing(frame: pd.DataFrame, *columns: str) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def optional_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)
