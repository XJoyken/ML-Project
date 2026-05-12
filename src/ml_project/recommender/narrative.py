from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Literal

from pydantic import BaseModel, Field

from ml_project.narrative import (
    DEFAULT_GEMINI_MODEL,
    FallbackReason,
    Language,
    NarrativeSource,
    _classify_llm_error,
    _fallback_reason_message,
    _strip_proper_noun_quotes,
)


class ItemExplanation(BaseModel):
    listing_id: str = Field(description="Listing id of the item being explained.")
    explanation: str = Field(description="A detailed explanation of why this listing is recommended, including how it matches user parameters, pros, cons, and nearby POIs.")
    url: str | None = Field(default=None, description="Link to the krisha.kz listing, formatted as https://krisha.kz/a/show/{listing_id}.")


class RecommendationNarrative(BaseModel):
    language: Language = Field(description="Output language.")
    summary: str = Field(description="One paragraph overview of why these 5 listings cover different needs.")
    items: list[ItemExplanation] = Field(description="Per-listing explanations.")
    source: NarrativeSource = Field(default="fallback")
    fallback_reason_code: FallbackReason | None = Field(default=None)
    fallback_reason: str | None = Field(default=None)


def generate(
    *,
    plan_payload: dict[str, Any],
    items: list[dict[str, Any]],
    language: Language = "ru",
    use_llm: bool | None = None,
    llm_client: Any | None = None,
    llm_model: str | None = None,
) -> RecommendationNarrative:
    if use_llm is None:
        use_llm = bool(os.getenv("GEMINI_API_KEY"))

    if use_llm:
        try:
            report = _generate_with_gemini(
                plan_payload=plan_payload,
                items=items,
                language=language,
                client=llm_client,
                model=llm_model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            )
            report.source = "gemini"
            _strip_quotes_in_place(report)
            return report
        except Exception as exc:
            print(
                f"[recommender] Gemini call failed, falling back to deterministic: {exc!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            reason_code = _classify_llm_error(exc)
            report = _deterministic_narrative(items=items, language=language)
            report.fallback_reason_code = reason_code
            report.fallback_reason = _fallback_reason_message(reason_code, language)
            return report

    return _deterministic_narrative(items=items, language=language)


def _generate_with_gemini(
    *,
    plan_payload: dict[str, Any],
    items: list[dict[str, Any]],
    language: Language,
    client: Any | None,
    model: str,
) -> RecommendationNarrative:
    from google import genai
    from google.genai import types

    if client is None:
        client = genai.Client()

    system_prompt = _system_prompt(language)
    payload = {
        "plan": plan_payload,
        "items": [_compact_item(item) for item in items],
    }
    user_payload = json.dumps(payload, ensure_ascii=False)
    response = client.models.generate_content(
        model=model,
        contents=user_payload,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3,
            response_mime_type="application/json",
            response_json_schema=RecommendationNarrative.model_json_schema(),
        ),
    )
    return RecommendationNarrative.model_validate_json(response.text)


def _system_prompt(language: Language) -> str:
    language_instruction = (
        "Write all text fields in Russian." if language == "ru" else "Write all text fields in English."
    )
    return f"""
You explain why a set of apartments was recommended to a buyer in detail.

Input has two parts:
- plan: the user's preferences (numeric, categorical, intent). Each target has a weight 0..1.
- items: candidate listings, each with a structured `matches` list, original seller's `description`, `nearby_pois`, and `nearest_air_sensor`.

Rules:
- {language_instruction}
- summary: one short paragraph (2-4 sentences) explaining how these listings cover DIFFERENT scenarios.
  Mention dispersion across price, district, rooms, key amenities. Avoid generic praise.
- items: Provide a DETAILED explanation for EACH listing. You MUST explain how the listing matches the user's explicit parameters from the plan. Also, elaborate on the pros and cons of the listing, mention nearby points of interest (objects nearby), conveniences, and inconveniences based on the provided data, so the user can make an informed choice.
- description usage: If the original seller `description` is available, you MUST read it and extract any useful perks (e.g. "good neighbors", "garage", "gated community") and explicitly mention: "Продавец упоминает в своем объявлении: ..." (or English equivalent) formatting the useful perks extracted from it.
- url: You must provide the URL to the listing formatted exactly as "https://krisha.kz/a/show/{{listing_id}}".
- Never wrap proper nouns (POI names, district names) in quotation marks of any kind.
  Bad: "Бостандыкский район". Good: Бостандыкский район.
- If a listing has hard_failures, clearly mention the compromise without hiding it.
- Use only data from the payload. Never invent POIs or features not present in the input.
""".strip()


def _compact_item(item: dict[str, Any]) -> dict[str, Any]:
    matches = [
        {
            "kind": m.get("kind"),
            "feature": m.get("feature"),
            "preference": m.get("preference"),
            "target": m.get("target"),
            "actual": m.get("actual"),
            "score": m.get("score"),
            "hard_failed": m.get("hard_failed", False),
        }
        for m in item.get("matches", [])
    ]
    return {
        "listing_id": item.get("listing_id"),
        "match_score": item.get("match_score"),
        "price_kzt": item.get("price_kzt"),
        "rooms": item.get("rooms"),
        "area_m2": item.get("area_m2"),
        "district": item.get("district"),
        "microdistrict": item.get("microdistrict"),
        "matches": matches,
        "hard_failures": item.get("hard_failures", []),
        "description": item.get("description"),
        "nearby_pois": item.get("nearby_pois"),
        "nearest_air_sensor": item.get("nearest_air_sensor"),
    }


def _deterministic_narrative(
    *,
    items: list[dict[str, Any]],
    language: Language,
) -> RecommendationNarrative:
    explanations = [
        ItemExplanation(
            listing_id=str(item.get("listing_id") or ""),
            explanation=_deterministic_item_text(item, language),
            url=f"https://krisha.kz/a/show/{item.get('listing_id')}" if item.get("listing_id") else None,
        )
        for item in items
    ]
    summary = _deterministic_summary(items, language)
    return RecommendationNarrative(
        language=language,
        summary=summary,
        items=explanations,
        source="fallback",
    )


def _deterministic_item_text(item: dict[str, Any], language: Language) -> str:
    matches = sorted(
        (m for m in item.get("matches", []) if not m.get("hard_failed")),
        key=lambda m: m.get("score") or 0.0,
        reverse=True,
    )[:3]
    parts: list[str] = []
    for match in matches:
        parts.append(_describe_match(match, language))
    listing_summary_parts: list[str] = []
    if item.get("price_kzt") is not None:
        price_text = _format_price(item["price_kzt"], language)
        listing_summary_parts.append(price_text)
    if item.get("rooms") is not None:
        rooms_value = int(round(item["rooms"]))
        listing_summary_parts.append(f"{rooms_value}-к." if language == "ru" else f"{rooms_value} rooms")
    if item.get("area_m2") is not None:
        area_value = float(item["area_m2"])
        unit = "м²" if language == "ru" else "m²"
        listing_summary_parts.append(f"{area_value:.0f} {unit}")
    if item.get("district"):
        listing_summary_parts.append(str(item["district"]))
    header = ", ".join(listing_summary_parts)

    if not parts:
        return header

    matches_text = "; ".join(p for p in parts if p)
    if language == "en":
        return f"{header}. Matches: {matches_text}."
    return f"{header}. Совпадения: {matches_text}."


def _describe_match(match: dict[str, Any], language: Language) -> str:
    feature = str(match.get("feature", ""))
    actual = match.get("actual")
    target = match.get("target")
    preference = match.get("preference")
    if actual is None or target is None:
        return ""
    if feature.endswith("_nearest_dist_m") and isinstance(actual, (int, float)):
        label = feature.removesuffix("_nearest_dist_m").replace("_", " ")
        distance = float(actual)
        if language == "en":
            return f"{label} {distance:.0f} m"
        return f"{label} {distance:.0f} м"
    if feature == "target_price_kzt":
        return _format_price(float(actual), language)
    if feature == "rooms":
        rooms = int(round(float(actual)))
        return f"{rooms}-к." if language == "ru" else f"{rooms} rooms"
    if match.get("kind") == "categorical":
        return str(actual)
    if isinstance(actual, (int, float)):
        if language == "en":
            return f"{feature}={actual:.1f}"
        return f"{feature}={actual:.1f}"
    return f"{feature}: {actual}"


def _deterministic_summary(items: list[dict[str, Any]], language: Language) -> str:
    if not items:
        return ""
    districts = sorted({str(item.get("district")) for item in items if item.get("district")})
    prices = [float(item["price_kzt"]) for item in items if item.get("price_kzt") is not None]
    if language == "en":
        district_part = (
            f"across {len(districts)} district(s) ({', '.join(districts)})" if districts else "across the city"
        )
        price_part = ""
        if prices:
            price_part = f", prices {min(prices) / 1e6:.0f}-{max(prices) / 1e6:.0f}M KZT"
        return f"{len(items)} listings {district_part}{price_part}."
    district_part = (
        f"по {len(districts)} районам ({', '.join(districts)})" if districts else "по городу"
    )
    price_part = ""
    if prices:
        price_part = f", цены {min(prices) / 1e6:.0f}–{max(prices) / 1e6:.0f} млн ₸"
    return f"{len(items)} вариантов {district_part}{price_part}."


def _format_price(price: float, language: Language) -> str:
    if language == "en":
        return f"{price / 1e6:.1f}M KZT"
    return f"{price / 1e6:.1f} млн ₸"


def _strip_quotes_in_place(report: RecommendationNarrative) -> None:
    report.summary = _strip_proper_noun_quotes(report.summary)
    for explanation in report.items:
        explanation.explanation = _strip_proper_noun_quotes(explanation.explanation)


__all__ = [
    "ItemExplanation",
    "RecommendationNarrative",
    "generate",
]
