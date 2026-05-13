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
    prioritize_air_quality: bool = False,
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
                prioritize_air_quality=prioritize_air_quality,
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
    prioritize_air_quality: bool = False,
) -> RecommendationNarrative:
    from google import genai
    from google.genai import types

    if client is None:
        client = genai.Client()

    system_prompt = _system_prompt(language, prioritize_air_quality)
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


def _system_prompt(language: Language, prioritize_air_quality: bool = False) -> str:
    language_instruction = (
        "Write all text fields in Russian." if language == "ru" else "Write all text fields in English."
    )
    if prioritize_air_quality:
        air_priority_instruction = (
            "- AIR PRIORITY MODE is on. Each item carries `air_pm25_cold_day` (µg/m³, winter day average),\n"
            "  `air_pm25_warm_day` (summer), `air_score` (0..1, higher=cleaner), and `air_bonus` — the small\n"
            "  ranking nudge that air priority added (it is intentionally capped at ±0.05 and CANNOT override\n"
            "  larger preference matches).\n"
            "- For EACH listing, state the PM2.5 numbers and translate them to a verdict: ≤15 — отличный,\n"
            "  ≤25 — хороший, ≤40 — умеренный, ≤60 — плохой, >60 — очень плохой воздух.\n"
            "- Compare PM2.5 across the 5 listings. If a listing has noticeably cleaner air than the others,\n"
            "  say so directly and quantify (e.g., 'PM2.5 22 vs средние 38 в остальной выдаче').\n"
            "- If a listing was bumped up partly because of air (air_bonus > 0) AND it has a compromise\n"
            "  (higher price than another option, larger distance to user-requested POI, or any hard_failures),\n"
            "  state the trade-off EXPLICITLY: e.g., 'квартира дороже на 3 млн, но воздух чище на 40% по PM2.5'.\n"
            "- Do NOT pretend air priority forced a compromise when it did not — only call it out when the\n"
            "  numbers actually show a worse main metric paired with a meaningfully cleaner air reading."
        )
    else:
        air_priority_instruction = ""
    return f"""
You explain why a set of apartments was recommended to a buyer.

Input has two parts:
- plan: the user's preferences (numeric, categorical, intent, excluded_districts).
  Each target has a weight 0..1. excluded_districts lists areas the user explicitly rejected.
- items: candidate listings. Each carries:
    - matches: structured list of how the listing scored against every user target (score 0..1, hard_failed flag).
    - base_score: ranking score from user preferences alone (before any air bonus).
    - match_score: final score actually used for ranking.
    - air_pm25_cold_day / air_pm25_warm_day / air_score / air_bonus.
    - nearby_pois: nearest POI per category, each with `category`, `name`, `distance_m`.
    - nearest_air_sensor: nearest PM2.5 monitoring station (distance, pm25_cold_day, pm25_warm_day).
    - description: original seller text in Russian (may be missing).

Rules:
- {language_instruction}

- summary: 2-4 sentences. Explain how the listings cover DIFFERENT scenarios
  (price band, district, rooms, key amenities). Avoid generic praise. Mention dispersion concretely
  using real numbers. If excluded_districts is non-empty, briefly confirm those districts are excluded.

- items: For EACH listing produce one cohesive paragraph (NOT a bullet list, NOT a one-liner)
  that integrates the following FOUR sections in this order, written naturally, woven into sentences:

  Section 1 — Match against plan (REQUIRED). State every plan target this listing satisfies with concrete
  numbers: cite the asked-for price and the actual price, rooms requested vs rooms found, district,
  has_complex_id (ЖК), dist_to_center_km, floor constraints, etc. If a target has hard_failed=true,
  name it as the explicit compromise: "комнат меньше на 1", "цена выше бюджета на 4 млн", etc.

  Section 2 — Nearby objects (REQUIRED, never skip even if obvious). Read nearby_pois and
  describe AT LEAST 4 distinct POI categories with their distance in meters (rounded to 10 m) and the
  POI name. Group them by relevance: schools/kindergartens/universities for families and students,
  metro/bus_stops for commuting, parks/restaurants_coffee/fitness for daily life, clinics/hospitals
  for medical access. If a category is unusually far (acceptable or far tier per the eval module),
  explicitly say so — that is a real downside.

  Section 3 — Air quality (REQUIRED whenever air data is present). State the PM2.5 numbers for both
  cold day and warm day with the verdict tier (≤15 отличный, ≤25 хороший, ≤25-40 умеренный,
  ≤40-60 плохой, >60 очень плохой). One concrete sentence is enough — do not skip this section.

  Section 4 — Seller description perks (REQUIRED whenever `description` is non-empty). Read the
  Russian seller text and pull 1–3 concrete perks ("евроремонт", "гардеробная", "панорамные окна",
  "охраняемый двор", "вид на горы", "новая сантехника"). Quote them with the lead-in
  "Продавец упоминает: …" (RU) or "Seller notes: …" (EN). If `description` is missing or empty,
  skip this section entirely — do NOT mention its absence.

  Length target: 4–7 sentences per listing. Dense and concrete, no filler.

- url: emit exactly "https://krisha.kz/a/show/{{listing_id}}".

{air_priority_instruction}

- Never wrap proper nouns (POI names, district names, ЖК names) in quotation marks of any kind.
  Bad: "Бостандыкский район". Good: Бостандыкский район.
- Use only data from the payload. Never invent POIs, district names, or numbers.
- Write distances in meters when < 1000 m, in kilometers (one decimal) when ≥ 1000 m.
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
        "base_score": item.get("base_score"),
        "air_bonus": item.get("air_bonus"),
        "air_score": item.get("air_score"),
        "air_pm25_cold_day": item.get("air_pm25_cold_day"),
        "air_pm25_warm_day": item.get("air_pm25_warm_day"),
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

    blocks = [header]

    if parts:
        matches_text = "; ".join(p for p in parts if p)
        if language == "en":
            blocks.append(f"Matches: {matches_text}.")
        else:
            blocks.append(f"Совпадения: {matches_text}.")

    poi_text = ""
    if item.get("nearby_pois"):
        pois = item["nearby_pois"]
        poi_items = []
        for poi in pois[:4]:
            from ..thresholds import category_label
            cat = category_label(poi["category"], language)
            dist = poi["distance_m"]
            poi_items.append(f"{cat} ({dist:.0f} м)")
        if poi_items:
            poi_text = ("Nearby: " if language == "en" else "Рядом: ") + ", ".join(poi_items) + "."
            blocks.append(poi_text)
            
    air_text = ""
    if item.get("nearest_air_sensor"):
        pm25 = item["nearest_air_sensor"].get("pm25_cold_day")
        if pm25 is not None:
            air_text = (f"Air PM2.5: {pm25:.0f}" if language == "en" else f"Воздух PM2.5: {pm25:.0f}") + "."
            blocks.append(air_text)
            
    desc_text = ""
    if item.get("description"):
        desc = str(item["description"]).strip()
        if desc:
            if len(desc) > 150:
                desc = desc[:147] + "..."
            desc_text = (f"Description: {desc}" if language == "en" else f"Описание продавца: {desc}")
            blocks.append(desc_text)

    return "\n".join(blocks)


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
    if feature == "has_complex_id":
        if float(actual) > 0.0:
            return "в ЖК" if language == "ru" else "in residential complex"
        return "не в ЖК" if language == "ru" else "not in residential complex"
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
