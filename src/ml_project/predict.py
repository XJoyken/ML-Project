from __future__ import annotations

from pathlib import Path
from typing import Any

from .air import load_air_catalog
from .constants import (
    AIR_QUALITY_RAW_PATH,
    CRIME_RAW_PATH,
    DEFAULT_EXPLANATION_CATEGORIES,
    EXPLANATION_RADIUS_M,
    MLFLOW_EXPERIMENT_NAME,
    POI_CATEGORY_LABELS,
    POI_SOURCE_FILES,
    VERDICT_OVERPRICED_THRESHOLD,
    VERDICT_UNDERVALUED_THRESHOLD,
)
from .crime import load_crime_catalog
from .data import get_listing_input, load_listings
from .features import build_inference_frame
from .poi import load_poi_catalog
from .tracking import load_latest_model
from .train import train_model


def predict_price(
    model_name: str = "lightgbm",
    *,
    listing_id: str | int | None = None,
    listing: dict[str, Any] | None = None,
    ads_path: Path | None = None,
    poi_sources: dict[str, Path] | None = None,
    air_quality_path: Path | None = None,
    model_params: dict[str, Any] | None = None,
    load_from_mlflow: bool = False,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> dict[str, Any]:
    if load_from_mlflow:
        reference_ads = load_listings(ads_path=ads_path)
        poi_catalog = load_poi_catalog(poi_sources=poi_sources or POI_SOURCE_FILES)
        air_catalog = load_air_catalog(raw_path=air_quality_path or AIR_QUALITY_RAW_PATH)
        crime_catalog = load_crime_catalog(raw_path=CRIME_RAW_PATH)
        run, model, schema = load_latest_model(experiment_name, model_name)
        mlflow_run_id: str | None = run.info.run_id
    else:
        training = train_model(
            model_name,
            ads_path=ads_path,
            poi_sources=poi_sources,
            air_quality_path=air_quality_path,
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

    if listing_id is not None:
        listing_payload = get_listing_input(listing_id, listings=reference_ads)
    else:
        listing_payload = dict(listing)

    processed, _ = build_inference_frame(
        listing_payload,
        reference_ads=reference_ads,
        poi_catalog=poi_catalog,
        air_catalog=air_catalog,
        crime_catalog=crime_catalog,
    )
    predicted_price = float(model.predict(processed, schema)[0])
    listing_price = float(processed.iloc[0][schema.target_column])
    delta_percent = ((predicted_price - listing_price) / predicted_price) * 100.0
    verdict = get_verdict(listing_price, predicted_price)
    listing_lat = float(processed.iloc[0]["lat"])
    listing_lon = float(processed.iloc[0]["lon"])
    nearby_pois = poi_catalog.nearest_many(
        lat=listing_lat,
        lon=listing_lon,
        categories=DEFAULT_EXPLANATION_CATEGORIES,
    )
    nearest_air = air_catalog.nearest_for(lat=listing_lat, lon=listing_lon)

    return {
        "predicted_price_kzt": predicted_price,
        "listing_price_kzt": listing_price,
        "delta_percent": delta_percent,
        "verdict": verdict,
        "nearby_pois": nearby_pois,
        "nearest_air_sensor": nearest_air,
        "summary_text": build_explanation(verdict, delta_percent, nearby_pois, nearest_air),
        "mlflow_run_id": mlflow_run_id,
    }


def build_explanation(
    verdict: str,
    delta_percent: float,
    nearby_pois: list[dict[str, Any]],
    nearest_air: dict[str, Any] | None = None,
) -> str:
    delta = abs(delta_percent)
    if verdict == "undervalued":
        lead = f"Хорошая цена квартиры, выгодно на {delta:.1f}%."
    elif verdict == "overpriced":
        lead = f"Цена квартиры завышена на {delta:.1f}%."
    else:
        lead = f"Цена квартиры близка к рыночной, отклонение {delta:.1f}%."

    details = []
    for poi in nearby_pois:
        if float(poi["distance_m"]) <= EXPLANATION_RADIUS_M:
            label = POI_CATEGORY_LABELS.get(str(poi["category"]), str(poi["category"]))
            details.append(f"В 300 м находится {label}: {poi['name']}.")

    air_sentence = describe_air(nearest_air)
    if air_sentence:
        details.append(air_sentence)

    return " ".join([lead, *details]) if details else lead


def describe_air(nearest_air: dict[str, Any] | None) -> str | None:
    if nearest_air is None:
        return None
    parts = []
    if nearest_air.get("pm25_cold_day") is not None:
        parts.append(f"зимой ~{nearest_air['pm25_cold_day']:.0f}")
    if nearest_air.get("pm25_warm_day") is not None:
        parts.append(f"летом ~{nearest_air['pm25_warm_day']:.0f}")
    if not parts:
        return None
    distance_m = nearest_air["distance_m"]
    return f"Ближайший сенсор PM2.5 в {distance_m:.0f} м показывает в среднем " + ", ".join(parts) + " мкг/м³."


def get_verdict(listing_price: float, predicted_price: float) -> str:
    if listing_price <= predicted_price * VERDICT_UNDERVALUED_THRESHOLD:
        return "undervalued"
    if listing_price >= predicted_price * VERDICT_OVERPRICED_THRESHOLD:
        return "overpriced"
    return "fair"
