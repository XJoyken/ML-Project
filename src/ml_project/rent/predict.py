from __future__ import annotations

from pathlib import Path
from typing import Any

from ..air import load_air_catalog
from ..constants import (
    AIR_QUALITY_RAW_PATH,
    CRIME_RAW_PATH,
    DEFAULT_EXPLANATION_CATEGORIES,
    POI_SOURCE_FILES,
    RENT_MLFLOW_EXPERIMENT_NAME,
)
from ..crime import load_crime_catalog
from ..poi import load_poi_catalog
from ..tracking import load_latest_model
from .data import load_rent_listings
from .features import build_rent_inference_frame
from .train import train_rent_model


def predict_rent(
    model_name: str = "lightgbm",
    *,
    listing: dict[str, Any] | None = None,
    ads_path: Path | None = None,
    poi_sources: dict[str, Path] | None = None,
    air_quality_path: Path | None = None,
    crime_path: Path | None = None,
    model_params: dict[str, Any] | None = None,
    load_from_mlflow: bool = False,
    experiment_name: str = RENT_MLFLOW_EXPERIMENT_NAME,
) -> dict[str, Any]:
    if listing is None:
        raise ValueError("predict_rent requires a `listing` dict.")

    if load_from_mlflow:
        reference_ads = load_rent_listings(ads_path=ads_path)
        poi_catalog = load_poi_catalog(poi_sources=poi_sources or POI_SOURCE_FILES)
        air_catalog = load_air_catalog(raw_path=air_quality_path or AIR_QUALITY_RAW_PATH)
        crime_catalog = load_crime_catalog(raw_path=crime_path or CRIME_RAW_PATH)
        run, model, schema = load_latest_model(experiment_name, model_name)
        mlflow_run_id: str | None = run.info.run_id
    else:
        training = train_rent_model(
            model_name,
            ads_path=ads_path,
            poi_sources=poi_sources,
            air_quality_path=air_quality_path,
            crime_path=crime_path,
            processed_path=None,
            save_processed=False,
            log_to_mlflow=False,
            model_params=model_params,
        )
        reference_ads = training["listings"]
        poi_catalog = training["poi_catalog"]
        air_catalog = training["air_catalog"]
        crime_catalog = training["crime_catalog"]
        model = training["model"]
        schema = training["schema"]
        mlflow_run_id = None

    processed, _ = build_rent_inference_frame(
        listing,
        reference_ads=reference_ads,
        poi_catalog=poi_catalog,
        air_catalog=air_catalog,
        crime_catalog=crime_catalog,
    )
    predicted_rent = float(model.predict(processed, schema)[0])
    lat = float(processed.iloc[0]["lat"])
    lon = float(processed.iloc[0]["lon"])
    nearby_pois = poi_catalog.nearest_many(
        lat=lat,
        lon=lon,
        categories=DEFAULT_EXPLANATION_CATEGORIES,
    )
    nearest_air = air_catalog.nearest_for(lat=lat, lon=lon)

    return {
        "predicted_rent_kzt": predicted_rent,
        "nearby_pois": nearby_pois,
        "nearest_air_sensor": nearest_air,
        "mlflow_run_id": mlflow_run_id,
    }


__all__ = ["predict_rent"]
