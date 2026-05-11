from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .air import load_air_catalog
from .constants import (
    AIR_QUALITY_RAW_PATH,
    CATEGORICAL_FEATURES,
    DEFAULT_EXPLANATION_CATEGORIES,
    POI_SOURCE_FILES,
    PROCESSED_DATASET_PATH,
    TARGET_COLUMN,
)
from .data import load_listings
from .features import FeatureSchema, build_inference_frame
from .models import create_model
from .poi import load_poi_catalog
from .predict import build_explanation, get_verdict


class ApartmentEvaluationService:
    def __init__(
        self,
        *,
        model_name: str = "catboost",
        processed_path: Path = PROCESSED_DATASET_PATH,
        model_params: dict[str, Any] | None = None,
    ):
        self.reference_ads = load_listings()
        self.poi_catalog = load_poi_catalog(POI_SOURCE_FILES)
        self.air_catalog = load_air_catalog(AIR_QUALITY_RAW_PATH)
        self.processed = pd.read_csv(processed_path)
        self.schema = schema_from_processed_frame(self.processed)
        self.model = create_model(model_name, **(model_params or {}))
        self.model.fit(self.processed, self.schema)
        self.model_name = model_name

    def evaluate(self, listing: dict[str, Any]) -> dict[str, Any]:
        processed, _ = build_inference_frame(
            listing,
            reference_ads=self.reference_ads,
            poi_catalog=self.poi_catalog,
            air_catalog=self.air_catalog,
        )
        predicted_price = float(self.model.predict(processed, self.schema)[0])
        listing_price = float(processed.iloc[0][self.schema.target_column])
        delta_percent = ((predicted_price - listing_price) / predicted_price) * 100.0
        verdict = get_verdict(listing_price, predicted_price)
        lat = float(processed.iloc[0]["lat"])
        lon = float(processed.iloc[0]["lon"])
        nearby_pois = self.poi_catalog.nearest_many(
            lat=lat,
            lon=lon,
            categories=DEFAULT_EXPLANATION_CATEGORIES,
        )
        nearest_air = self.air_catalog.nearest_for(lat=lat, lon=lon)

        return {
            "model": self.model_name,
            "predicted_price_kzt": predicted_price,
            "listing_price_kzt": listing_price,
            "delta_percent": delta_percent,
            "verdict": verdict,
            "summary_text": build_explanation(verdict, delta_percent, nearby_pois, nearest_air),
            "nearby_pois": nearby_pois,
            "nearest_air_sensor": nearest_air,
        }


def schema_from_processed_frame(processed: pd.DataFrame) -> FeatureSchema:
    categorical_features = [
        column for column in CATEGORICAL_FEATURES if column in processed.columns
    ]
    numeric_features = [
        column
        for column in processed.columns
        if column not in {TARGET_COLUMN, *categorical_features}
    ]
    return FeatureSchema(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
