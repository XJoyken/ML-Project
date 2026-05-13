from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow.lightgbm

from ..air import load_air_catalog
from ..calibration import PriceCalibration, load_or_compute_calibration
from ..constants import (
    AIR_QUALITY_RAW_PATH,
    CRIME_RAW_PATH,
    DEFAULT_APARTMENT_RENT_MODEL_DIR,
    DEFAULT_EXPLANATION_CATEGORIES,
    POI_SOURCE_FILES,
    RENT_PROCESSED_DATASET_PATH,
)
from ..crime import load_crime_catalog
from ..features import FeatureSchema
from ..models import wrap_loaded_model
from ..poi import load_poi_catalog
from .data import load_rent_listings
from .features import build_rent_inference_frame

MODEL_NAME = "lightgbm"


class RentEvaluationService:
    def __init__(
        self,
        *,
        model_dir: Path = DEFAULT_APARTMENT_RENT_MODEL_DIR,
        processed_path: Path = RENT_PROCESSED_DATASET_PATH,
    ):
        self.reference_ads = load_rent_listings()
        self.poi_catalog = load_poi_catalog(POI_SOURCE_FILES)
        self.air_catalog = load_air_catalog(AIR_QUALITY_RAW_PATH)
        self.crime_catalog = load_crime_catalog(CRIME_RAW_PATH)
        self.schema = load_schema(model_dir / "schema.json")
        loaded_model = mlflow.lightgbm.load_model(str(model_dir))
        self.model = wrap_loaded_model(MODEL_NAME, loaded_model)
        self.model_dir = str(model_dir)
        self.calibration: PriceCalibration = load_or_compute_calibration(
            model_dir=model_dir,
            model=self.model,
            schema=self.schema,
            processed_path=processed_path,
        )

    def evaluate(self, listing: dict[str, Any]) -> dict[str, Any]:
        processed, _ = build_rent_inference_frame(
            listing,
            reference_ads=self.reference_ads,
            poi_catalog=self.poi_catalog,
            air_catalog=self.air_catalog,
            crime_catalog=self.crime_catalog,
        )
        predicted_rent = float(self.model.predict(processed, self.schema)[0])
        lat = float(processed.iloc[0]["lat"])
        lon = float(processed.iloc[0]["lon"])
        district = str(processed.iloc[0]["district"]) if "district" in processed.columns else None
        nearby_pois = self.poi_catalog.nearest_many(
            lat=lat,
            lon=lon,
            categories=DEFAULT_EXPLANATION_CATEGORIES,
        )
        nearest_air = self.air_catalog.nearest_for(lat=lat, lon=lon)
        crime_info = self.crime_catalog.lookup(district)
        interval_low, interval_high = self.calibration.interval_for(predicted_rent)

        return {
            "model": MODEL_NAME,
            "model_dir": self.model_dir,
            "predicted_rent_kzt": predicted_rent,
            "rent_interval": {
                "coverage": self.calibration.coverage,
                "low_kzt": interval_low,
                "high_kzt": interval_high,
            },
            "nearby_pois": nearby_pois,
            "nearest_air_sensor": nearest_air,
            "crime": crime_info,
        }


def load_schema(path: Path) -> FeatureSchema:
    schema_dict = json.loads(path.read_text(encoding="utf-8"))
    return FeatureSchema(
        numeric_features=list(schema_dict["numeric_features"]),
        categorical_features=list(schema_dict["categorical_features"]),
        target_column=schema_dict["target_column"],
    )


__all__ = ["RentEvaluationService"]
