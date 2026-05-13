from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..air import AIR_FEATURE_COLUMNS, AirQualityCatalog
from ..constants import (
    BASE_NUMERIC_FEATURES,
    RENT_CATEGORICAL_FEATURES,
    RENT_TARGET_COLUMN,
)
from ..crime import CRIME_FEATURE_COLUMNS, CrimeCatalog
from ..features import (
    FeatureSchema,
    build_air_features,
    build_crime_features,
    build_poi_features,
)
from ..poi import PoiCatalog
from .data import normalize_rent_listing_frame, validate_rent_inference_frame


def build_rent_base_features(listings: pd.DataFrame) -> pd.DataFrame:
    return listings[[*BASE_NUMERIC_FEATURES, *RENT_CATEGORICAL_FEATURES]].copy()


def build_rent_feature_frame(
    listings: pd.DataFrame,
    poi_catalog: PoiCatalog,
    air_catalog: AirQualityCatalog,
    crime_catalog: CrimeCatalog,
) -> tuple[pd.DataFrame, FeatureSchema]:
    base_features = build_rent_base_features(listings)
    poi_features = build_poi_features(listings, poi_catalog)
    air_features = build_air_features(listings, air_catalog)
    crime_features = build_crime_features(listings, crime_catalog)
    poi_feature_columns = sorted(poi_features.columns.tolist())
    schema = FeatureSchema(
        numeric_features=[
            *BASE_NUMERIC_FEATURES,
            *poi_feature_columns,
            *AIR_FEATURE_COLUMNS,
            *CRIME_FEATURE_COLUMNS,
        ],
        categorical_features=list(RENT_CATEGORICAL_FEATURES),
        target_column=RENT_TARGET_COLUMN,
    )
    features = pd.concat([base_features, poi_features, air_features, crime_features], axis=1)
    return features[schema.feature_columns].copy(), schema


def build_rent_processed_dataset(
    listings: pd.DataFrame,
    poi_catalog: PoiCatalog,
    air_catalog: AirQualityCatalog,
    crime_catalog: CrimeCatalog,
    *,
    save_path: Path | None = None,
) -> tuple[pd.DataFrame, FeatureSchema]:
    features, schema = build_rent_feature_frame(listings, poi_catalog, air_catalog, crime_catalog)
    processed = pd.concat([listings[[schema.target_column]], features], axis=1)
    processed = processed[[schema.target_column, *schema.feature_columns]].copy()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(save_path, index=False)

    return processed, schema


def build_rent_inference_frame(
    listing: dict[str, Any],
    *,
    reference_ads: pd.DataFrame,
    poi_catalog: PoiCatalog,
    air_catalog: AirQualityCatalog,
    crime_catalog: CrimeCatalog,
) -> tuple[pd.DataFrame, FeatureSchema]:
    record = dict(listing)
    # Accept both "listing_price_kzt" (from a rent-ad parser) and the canonical target name.
    if "listing_price_kzt" in record and RENT_TARGET_COLUMN not in record:
        record[RENT_TARGET_COLUMN] = record.pop("listing_price_kzt")
    # The investment endpoint runs rent inference on a SALE listing — its price is the sale
    # price, not the rent. We don't want to leak it as the rent label, so when the caller
    # is asking for a pure rent prediction (no real rent label) we set NaN; downstream
    # validators don't require the target.
    if RENT_TARGET_COLUMN not in record:
        record[RENT_TARGET_COLUMN] = np.nan

    normalized = normalize_rent_listing_frame(
        pd.DataFrame([record]),
        reference_ads=reference_ads,
        fit_mode=False,
    )
    validate_rent_inference_frame(normalized)
    return build_rent_processed_dataset(
        normalized,
        poi_catalog,
        air_catalog,
        crime_catalog,
        save_path=None,
    )


__all__ = [
    "build_rent_base_features",
    "build_rent_feature_frame",
    "build_rent_processed_dataset",
    "build_rent_inference_frame",
]
