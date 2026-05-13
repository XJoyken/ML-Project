from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import (
    AREA_M2_MAX,
    AREA_M2_MIN,
    RENT_ADS_COLUMNS,
    RENT_CATEGORICAL_FEATURES,
    RENT_DATASET_PATH,
    RENT_PPM_LOWER_QUANTILE,
    RENT_PPM_UPPER_QUANTILE,
    RENT_TARGET_COLUMN,
    ROOMS_MAX,
)
from ..data import (
    DISTRICT_NAME_MAP,
    add_complex_listing_count,
    add_derived_features,
    add_presence_flags,
    canonicalize_district,
    clean_implausible_year_built,
    coerce_numeric_columns,
    ensure_columns,
    fill_text_features,
    normalize_ceiling_height,
    normalize_floor_columns,
    normalize_photo_features,
)

RENT_LISTING_COLUMN_MAP = {
    "listing_price_kzt": RENT_TARGET_COLUMN,
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
    "parameter_Санузел": "bathroom_type",
    "summary_Квартира_меблирована": "furnished",
    "listing_has_photo": "has_photo",
    "listing_photo_count": "photo_count",
    "listing_description": "description",
}

NORMALIZED_RENT_COLUMNS = [
    "listing_id",
    RENT_TARGET_COLUMN,
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
    "furnished",
    "has_photo",
    "photo_count",
    "description",
]

RENT_NUMERIC_COLUMNS = [
    RENT_TARGET_COLUMN,
    "area_m2",
    "rooms",
    "lat",
    "lon",
    "photo_count",
    "year_built",
]


def load_rent_listings(ads_path: Path | None = None) -> pd.DataFrame:
    ads_path = ads_path or RENT_DATASET_PATH
    frame = pd.read_csv(ads_path, usecols=RENT_ADS_COLUMNS, low_memory=False)
    frame = frame.loc[frame["listing_category_alias"].eq("kvartiry")].copy()
    frame = frame.rename(columns=RENT_LISTING_COLUMN_MAP)

    frame = normalize_rent_listing_frame(frame, fit_mode=True)
    frame = filter_implausible_rent_listings(frame)
    return frame.reset_index(drop=True)


def filter_implausible_rent_listings(frame: pd.DataFrame) -> pd.DataFrame:
    price_per_m2 = frame[RENT_TARGET_COLUMN] / frame["area_m2"]
    low = price_per_m2.quantile(RENT_PPM_LOWER_QUANTILE)
    high = price_per_m2.quantile(RENT_PPM_UPPER_QUANTILE)
    keep = (
        frame[RENT_TARGET_COLUMN].gt(0)
        & frame["area_m2"].between(AREA_M2_MIN, AREA_M2_MAX)
        & frame["rooms"].notna()
        & frame["rooms"].le(ROOMS_MAX)
        & frame["lat"].between(43.0, 44.0)
        & frame["lon"].between(76.0, 77.2)
        & price_per_m2.between(low, high)
    )
    return frame.loc[keep].copy()


def normalize_rent_listing_frame(
    frame: pd.DataFrame,
    *,
    reference_ads: pd.DataFrame | None = None,
    fit_mode: bool = False,
) -> pd.DataFrame:
    canonical = frame.copy()
    ensure_columns(canonical, NORMALIZED_RENT_COLUMNS)
    coerce_numeric_columns(canonical, RENT_NUMERIC_COLUMNS)
    clean_implausible_year_built(canonical)
    normalize_ceiling_height(canonical)
    normalize_floor_columns(canonical)
    normalize_photo_features(canonical)
    add_presence_flags(canonical)
    add_complex_listing_count(canonical, reference_ads=reference_ads, fit_mode=fit_mode)
    add_derived_features(canonical)
    canonicalize_district(canonical)
    fill_rent_categorical_features(canonical)
    fill_text_features(canonical)
    return canonical.reset_index(drop=True)


def fill_rent_categorical_features(frame: pd.DataFrame):
    for column in RENT_CATEGORICAL_FEATURES:
        frame[column] = frame[column].fillna("unknown").astype("string")


def validate_rent_inference_frame(frame: pd.DataFrame):
    required = {
        "area_m2": "area_m2",
        "rooms": "rooms",
        "lat": "lat",
        "lon": "lon",
    }
    missing = [
        public_name
        for column, public_name in required.items()
        if frame[column].isna().any()
    ]
    if missing:
        raise ValueError(f"Missing required rent listing fields: {', '.join(missing)}")


__all__ = [
    "DISTRICT_NAME_MAP",  # re-exported for convenience
    "RENT_LISTING_COLUMN_MAP",
    "NORMALIZED_RENT_COLUMNS",
    "RENT_NUMERIC_COLUMNS",
    "load_rent_listings",
    "normalize_rent_listing_frame",
    "validate_rent_inference_frame",
]
