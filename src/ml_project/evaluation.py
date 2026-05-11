from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow.catboost

from .air import load_air_catalog
from .constants import (
    AIR_QUALITY_RAW_PATH,
    DEFAULT_EXPLANATION_CATEGORIES,
    POI_SOURCE_FILES,
    ROOT_DIR,
)
from .data import load_listings
from .features import FeatureSchema, build_inference_frame
from .models import wrap_loaded_model
from .poi import load_poi_catalog
from .predict import build_explanation, get_verdict

DEFAULT_MODEL_ARTIFACT_PATH = (
    ROOT_DIR / "mlruns/1/models/m-7288ada3b93d42b79bcd9a35ca469d02/artifacts"
)
DEFAULT_SCHEMA_PATH = (
    ROOT_DIR / "mlruns/1/5829d06ee0da4d3185a4d40e58791ee2/artifacts/schema.json"
)


class ApartmentEvaluationService:
    def __init__(
        self,
        *,
        model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT_PATH,
        schema_path: Path = DEFAULT_SCHEMA_PATH,
        model_name: str = "catboost",
    ):
        self.reference_ads = load_listings()
        self.poi_catalog = load_poi_catalog(POI_SOURCE_FILES)
        self.air_catalog = load_air_catalog(AIR_QUALITY_RAW_PATH)
        self.schema = load_schema(schema_path)
        loaded_model = mlflow.catboost.load_model(str(model_artifact_path))
        self.model = wrap_loaded_model(model_name, loaded_model)
        self.model_name = model_name
        self.model_artifact_path = str(model_artifact_path)

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
            "model_artifact_path": self.model_artifact_path,
            "predicted_price_kzt": predicted_price,
            "listing_price_kzt": listing_price,
            "delta_percent": delta_percent,
            "verdict": verdict,
            "summary_text": build_explanation(verdict, delta_percent, nearby_pois, nearest_air),
            "nearby_pois": nearby_pois,
            "nearest_air_sensor": nearest_air,
        }


def load_schema(path: Path) -> FeatureSchema:
    schema_dict = json.loads(path.read_text(encoding="utf-8"))
    return FeatureSchema(
        numeric_features=list(schema_dict["numeric_features"]),
        categorical_features=list(schema_dict["categorical_features"]),
        target_column=schema_dict["target_column"],
    )
