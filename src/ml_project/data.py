from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    ADS_COLUMNS,
    CATEGORICAL_FEATURES,
    DATA_DIR,
    TARGET_COLUMN,
    TARGET_LOG_COLUMN,
)
from .poi import PoiCatalog, load_poi_catalog

LISTING_COLUMN_MAP = {
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

NORMALIZED_LISTING_COLUMNS = [
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
]

NUMERIC_LISTING_COLUMNS = [
    TARGET_COLUMN,
    "area_m2",
    "rooms",
    "lat",
    "lon",
    "photo_count",
    "year_built",
]

def load_listings(ads_path: Path | None = None) -> pd.DataFrame:
    ads_path = ads_path or (DATA_DIR / "ads.csv")
    frame = pd.read_csv(ads_path, usecols=ADS_COLUMNS, low_memory=False)
    frame = frame.loc[frame["listing_category_alias"].eq("kvartiry")].copy()
    frame = frame.rename(columns=LISTING_COLUMN_MAP)

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
    return matched.iloc[0].to_dict()


def normalize_listing_frame(
    frame: pd.DataFrame,
    *,
    reference_ads: pd.DataFrame | None = None,
    fit_mode: bool = False,
) -> pd.DataFrame:
    canonical = frame.copy()
    ensure_columns(canonical, NORMALIZED_LISTING_COLUMNS)
    coerce_numeric_columns(canonical, NUMERIC_LISTING_COLUMNS)
    normalize_ceiling_height(canonical)
    normalize_floor_columns(canonical)
    normalize_photo_features(canonical)
    add_presence_flags(canonical)
    add_complex_listing_count(canonical, reference_ads=reference_ads, fit_mode=fit_mode)
    fill_categorical_features(canonical)

    canonical[TARGET_LOG_COLUMN] = np.log1p(canonical[TARGET_COLUMN])
    return canonical.reset_index(drop=True)


def coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def normalize_ceiling_height(frame: pd.DataFrame):
    if frame["ceiling_height_m"].isna().all():
        frame["ceiling_height_m"] = extract_numeric(frame["ceiling_height_text"])
        return

    frame["ceiling_height_m"] = pd.to_numeric(
        frame["ceiling_height_m"],
        errors="coerce",
    )


def normalize_floor_columns(frame: pd.DataFrame):
    if frame[["floor_current", "floors_total"]].isna().all(axis=None):
        frame[["floor_current", "floors_total"]] = parse_floor_columns(frame["floor_text"])
        return

    coerce_numeric_columns(frame, ["floor_current", "floors_total"])


def normalize_photo_features(frame: pd.DataFrame):
    has_photo = frame["has_photo"]
    if has_photo.dtype == "object" or str(has_photo.dtype).startswith("string"):
        has_photo = binary_from_text(has_photo)

    frame["has_photo"] = pd.to_numeric(has_photo, errors="coerce").fillna(
        (frame["photo_count"].fillna(0) > 0).astype(float)
    ).astype(float)


def add_presence_flags(frame: pd.DataFrame):
    frame["has_complex_id"] = frame["complex_id"].notna().astype(np.int8)
    frame["has_microdistrict"] = frame["microdistrict"].notna().astype(np.int8)


def add_complex_listing_count(
    frame: pd.DataFrame,
    *,
    reference_ads: pd.DataFrame | None,
    fit_mode: bool,
):
    source = frame if fit_mode or reference_ads is None else reference_ads
    complex_counts = source["complex_id"].value_counts(dropna=True).to_dict()
    frame["complex_listing_count"] = (
        frame["complex_id"].map(complex_counts).fillna(0).astype(np.int32)
    )


def fill_categorical_features(frame: pd.DataFrame):
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("unknown").astype("string")


def validate_inference_frame(frame: pd.DataFrame):
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


def ensure_columns(frame: pd.DataFrame, columns: list[str]):
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
