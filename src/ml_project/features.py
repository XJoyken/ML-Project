from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from .data import normalize_listing_frame, validate_inference_frame
from .poi import PoiCatalog


@dataclass(slots=True)
class FeatureSchema:
    numeric_features: list[str]
    categorical_features: list[str]
    target_column: str = TARGET_COLUMN

    @property
    def feature_columns(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_column": self.target_column,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
        }


def build_base_features(listings: pd.DataFrame) -> pd.DataFrame:
    return listings[[*BASE_NUMERIC_FEATURES, *CATEGORICAL_FEATURES]].copy()


def build_poi_features(listings: pd.DataFrame, poi_catalog: PoiCatalog) -> pd.DataFrame:
    return poi_catalog.build_feature_frame(listings)


def build_feature_frame(
    listings: pd.DataFrame,
    poi_catalog: PoiCatalog,
) -> tuple[pd.DataFrame, FeatureSchema]:
    base_features = build_base_features(listings)
    poi_features = build_poi_features(listings, poi_catalog)
    poi_feature_columns = sorted(poi_features.columns.tolist())
    schema = FeatureSchema(
        numeric_features=[*BASE_NUMERIC_FEATURES, *poi_feature_columns],
        categorical_features=list(CATEGORICAL_FEATURES),
    )
    features = pd.concat([base_features, poi_features], axis=1)
    return features[schema.feature_columns].copy(), schema


def build_processed_dataset(
    listings: pd.DataFrame,
    poi_catalog: PoiCatalog,
    *,
    save_path: Path | None = None,
) -> tuple[pd.DataFrame, FeatureSchema]:
    features, schema = build_feature_frame(listings, poi_catalog)
    processed = pd.concat([listings[[schema.target_column]], features], axis=1)
    processed = processed[[schema.target_column, *schema.feature_columns]].copy()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(save_path, index=False)

    return processed, schema


def build_inference_frame(
    listing: dict[str, Any],
    *,
    reference_ads: pd.DataFrame,
    poi_catalog: PoiCatalog,
) -> tuple[pd.DataFrame, FeatureSchema]:
    record = dict(listing)
    if "listing_price_kzt" in record and TARGET_COLUMN not in record:
        record[TARGET_COLUMN] = record.pop("listing_price_kzt")

    normalized = normalize_listing_frame(
        pd.DataFrame([record]),
        reference_ads=reference_ads,
        fit_mode=False,
    )
    validate_inference_frame(normalized)
    return build_processed_dataset(normalized, poi_catalog, save_path=None)
