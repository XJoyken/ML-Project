from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import (
    ARTIFACTS_DIR,
    DEFAULT_EXPLANATION_CATEGORIES,
    EXPLANATION_RADIUS_M,
    POI_CATEGORY_LABELS,
    POI_SOURCE_FILES,
    VERDICT_OVERPRICED_THRESHOLD,
    VERDICT_UNDERVALUED_THRESHOLD,
)
from .data import get_listing_input, load_listings, load_poi_catalog
from .features import build_inference_frame
from .models import load_model
from .train import artifact_dir, load_schema, schema_path


def predict_price(
    model_name: str = "catboost",
    *,
    listing_id: str | int | None = None,
    listing: dict[str, Any] | None = None,
    ads_path: Path | None = None,
    poi_sources: dict[str, Path] | None = None,
    artifact_root: Path | None = ARTIFACTS_DIR,
) -> dict[str, Any]:
    reference_ads = load_listings(ads_path=ads_path)
    poi_catalog = load_poi_catalog(poi_sources=poi_sources or POI_SOURCE_FILES)

    if listing_id is not None:
        listing_payload = get_listing_input(listing_id, listings=reference_ads)
    elif listing is not None:
        listing_payload = dict(listing)
    else:
        raise ValueError("Pass either listing_id or listing.")

    processed, fallback_schema = build_inference_frame(
        listing_payload,
        reference_ads=reference_ads,
        poi_catalog=poi_catalog,
    )
    schema = (
        load_schema(model_name, artifact_root=artifact_root)
        if schema_path(model_name, artifact_root=artifact_root).exists()
        else fallback_schema
    )
    model = load_model(model_name, artifact_dir(model_name, artifact_root=artifact_root))
    predicted_price = float(model.predict(processed, schema)[0])
    listing_price = float(processed.iloc[0][schema.target_column])
    delta_percent = (
        ((predicted_price - listing_price) / predicted_price) * 100.0
        if predicted_price
        else 0.0
    )
    verdict = get_verdict(listing_price, predicted_price)
    nearby_pois = poi_catalog.nearest_many(
        lat=float(processed.iloc[0]["lat"]),
        lon=float(processed.iloc[0]["lon"]),
        categories=DEFAULT_EXPLANATION_CATEGORIES,
    )

    return {
        "predicted_price_kzt": predicted_price,
        "listing_price_kzt": listing_price,
        "delta_percent": delta_percent,
        "verdict": verdict,
        "nearby_pois": nearby_pois,
        "summary_text": build_explanation(verdict, delta_percent, nearby_pois),
    }


def build_explanation(
    verdict: str,
    delta_percent: float,
    nearby_pois: list[dict[str, Any]],
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

    return " ".join([lead, *details]) if details else lead


def get_verdict(listing_price: float, predicted_price: float) -> str:
    if listing_price <= predicted_price * VERDICT_UNDERVALUED_THRESHOLD:
        return "undervalued"
    if listing_price >= predicted_price * VERDICT_OVERPRICED_THRESHOLD:
        return "overpriced"
    return "fair"
