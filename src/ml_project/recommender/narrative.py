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
You write a sales-quality apartment recommendation for a buyer. The reader is non-technical — write
like a thoughtful agent: warm, concrete, honest about trade-offs. Help them CHOOSE between the listings.

═══════════════════════════════════════════════════════════════════════════════════════
INPUT STRUCTURE
═══════════════════════════════════════════════════════════════════════════════════════
- plan: user's preferences. `numeric_targets` and `categorical_targets` each carry `feature`,
  `value`, `weight` 0..1, and the original `evidence` phrase (the user's actual words).
  `excluded_districts` lists rejected areas.
- items: candidate listings. Each carries:
    - matches: how the listing scored against every plan target (score 0..1, hard_failed flag).
    - base_score, match_score: ranking scores.
    - price_kzt, rooms, area_m2, floor_current, floors_total, year_built, condition,
      house_type, ceiling_height_m, dist_to_center_km, district, microdistrict.
    - air_pm25_cold_day / air_pm25_warm_day / air_score / air_bonus.
    - nearby_pois: nearest POI per category. Each item: `category`, `name`, `distance_m`.
      bus_stops items additionally carry `routes_list`.
    - nearest_air_sensor: distance, pm25_cold_day, pm25_warm_day.
    - description: seller's original Russian text (may be missing/empty).

═══════════════════════════════════════════════════════════════════════════════════════
ABSOLUTE RULES — VIOLATIONS ARE A FAILURE
═══════════════════════════════════════════════════════════════════════════════════════
1. {language_instruction} ALL fields, including tier names.
2. NEVER wrap proper nouns (POI names, districts, ЖК names) in quotes — plain inline text only.
3. Use EXACT numbers from the payload. Do NOT round distances, prices, or PM2.5.
4. Distances: meters when < 1000 ("547 м"), kilometers with 1 decimal when ≥ 1000 ("1.3 км").
5. Prices: "38.5 млн ₸" (1 decimal); areas: "42.3 м²" (1 decimal).

6. ★ DO NOT EXPOSE INTERNAL NUMERIC TARGETS the user did not say in `evidence`.
   The system may have inserted defaults the user never typed (e.g. "рядом школа" became
   `schools_nearest_dist_m at_most 500`). The user did NOT write "500 m".
   ❌ Bad: "школа в 322 м, что укладывается в ваш лимит 500 м".
   ✅ Good: "школа №51 в 322 м — действительно рядом".
   RULE: When citing a POI match, mention only the ACTUAL distance (`distance_m` from nearby_pois)
   and the POI name. The internal target value is invisible to the user — keep it that way.
   This rule also applies to price targets when the user did not write a price evidence.
   When citing an attribute the user explicitly mentioned, you MAY reference their literal phrase
   from `evidence` (e.g. "укладывается в ваше «до 40 млн»") — but only if the evidence string
   really contains that figure.

═══════════════════════════════════════════════════════════════════════════════════════
FIELD: summary  (2–4 sentences)
═══════════════════════════════════════════════════════════════════════════════════════
Compare the SET. State price range, room/area spread, district variety, and whether all listings
cover the user's hard requirements or some required compromises. Cite concrete numbers.
If excluded_districts is non-empty, briefly confirm those are filtered out.
NO filler like "отличная подборка". Be useful: tell the reader what makes the set diverse.

═══════════════════════════════════════════════════════════════════════════════════════
FIELD: items[].explanation  (6–9 sentences PER listing, ONE cohesive paragraph)
═══════════════════════════════════════════════════════════════════════════════════════
Five sections, woven in prose with natural connectors (Также, Кроме того, При этом, Из минусов).

────────────────────────────────────────────────────────────────────────────────────
SECTION 1 — Headline match (1–2 sentences)
────────────────────────────────────────────────────────────────────────────────────
Open with rooms + (ЖК / тип дома) + price + district + floor. Then in 1 sentence address
each plan target the user explicitly evidenced. If a `hard_failed=true` match exists,
NAME the compromise plainly: "комнат на 1 меньше", "цена выше бюджета на 3 млн".

────────────────────────────────────────────────────────────────────────────────────
SECTION 2 — Locality (2–3 sentences, ≥3 distinct POI categories)
────────────────────────────────────────────────────────────────────────────────────
Pick from nearby_pois. If the user asked for a POI category in their evidence — mention it FIRST.
Phrase POI matches as: "<категория> <название> в <дистанция>" — never reference internal target
values. Mention ≥3 categories spread across: семьи (schools/kindergartens), быт (parks/supermarkets/
fitness/cafe), транспорт (metro/bus_stops/transport), медицина (clinics/hospitals/dentistry).
For bus_stops: list routes in parentheses ("автобусы 18, 95; троллейбус Тр6"; "Тр" = trolleybus).

────────────────────────────────────────────────────────────────────────────────────
SECTION 3 — Plus features for THIS listing (1–2 sentences, REQUIRED)
────────────────────────────────────────────────────────────────────────────────────
Concrete distinguishing pros the buyer cares about. Pick 1–3 of:
  - Seller description perks ("евроремонт", "панорамные окна", "вид на горы", "охраняемый двор",
    "гардеробная", "тёплый пол", "новая сантехника") — lead with "Продавец упоминает: …" (RU)
    or "Seller notes: …" (EN).
  - Recent year_built (≥ 2018 = свежая новостройка; 2010–2017 = относительно новый дом).
  - Reasonable floor (mid-floors 3–7 in a high building are usually optimal).
  - High ceilings (ceiling_height_m ≥ 3.0).
  - Excellent air (PM2.5 cold ≤ 15).
  - Notably close to center (dist_to_center_km ≤ 3 = центр; ≤ 5 = ближе к центру).
  - Condition such as "евроремонт", "хорошее", "отличное".
If nothing distinguishing exists, write ONE sentence with the strongest available perk
(e.g. price-per-area). Never skip this section.

────────────────────────────────────────────────────────────────────────────────────
SECTION 4 — Compromises / minuses for THIS listing (1–2 sentences, REQUIRED)
────────────────────────────────────────────────────────────────────────────────────
Be honest. List 1–2 real drawbacks (do NOT invent). Pick from:
  - Hard-failed targets ("комнат меньше на 1, чем просили").
  - Far POIs vs typical needs: kindergartens/schools > 1500 m, parks > 1000 m,
    transport/bus_stops > 800 m, supermarkets > 1000 m, clinics > 2000 m.
  - Old building (year_built < 1980 = старый фонд; 1980–2005 = типовая советская/постсоветская).
  - Edge floors when relevant (1-й или последний без указания обратного желания пользователя).
  - High PM2.5 (cold > 40 = умеренный/плохой).
  - Highest price in the set, smallest area, longest commute to center vs the rest.
  - "Поднимет" or "под ремонт" condition.
If you genuinely cannot find a drawback, write ONE sentence like:
  "Очевидных минусов по данным нет — стоит лично проверить состояние при просмотре."
Never SKIP section 4.

────────────────────────────────────────────────────────────────────────────────────
SECTION 5 — Air quality (REQUIRED when air data present, 1 sentence)
────────────────────────────────────────────────────────────────────────────────────
Both PM2.5 numbers with verdict tier:
  ≤15 — отличный · ≤25 — хороший · ≤40 — умеренный · ≤60 — плохой · >60 — очень плохой.
Example (RU): "Воздух — PM2.5 22 (хороший) зимой и 17 (хороший) летом."

═══════════════════════════════════════════════════════════════════════════════════════
FIELD: items[].url
═══════════════════════════════════════════════════════════════════════════════════════
Exactly "https://krisha.kz/a/show/{{listing_id}}".

{air_priority_instruction}

═══════════════════════════════════════════════════════════════════════════════════════
WORKED OUTPUT EXAMPLE (RU)
═══════════════════════════════════════════════════════════════════════════════════════
2-комнатная за 38.5 млн ₸ в ЖК Алмалы — точно по вашему запросу: 2 комнаты, новостройка 2021 года в Алмалинском районе, 4-й этаж из 9. Из ближайшего: школа №51 в 322 м, детский сад Балапан в 180 м, парк им. Ганди в 540 м, остановка Музей искусств в 210 м (автобусы 18, 95; троллейбус Тр6). Плюсы — продавец упоминает: евроремонт, панорамные окна на горы, охраняемый двор; дом 2021 года, потолки 3.0 м. Из минусов — поликлиника далеко (1.6 км) и квартира в верхней половине бюджета. Воздух — PM2.5 22 (хороший) зимой и 17 (хороший) летом.

═══════════════════════════════════════════════════════════════════════════════════════
QUALITY CHECKLIST — MENTAL PASS BEFORE EMITTING JSON
═══════════════════════════════════════════════════════════════════════════════════════
For EVERY item, verify:
  ☐ Section 1: rooms / ЖК / price / district / floor explicitly stated.
  ☐ Section 2: ≥3 POI categories, exact distances + names, no internal-target leakage.
  ☐ Section 3: ≥1 concrete pro from description / year / floor / condition / air / center.
  ☐ Section 4: ≥1 concrete con (or honest "no obvious cons in data").
  ☐ Section 5: BOTH PM2.5 numbers AND the tier word.
  ☐ Hard-failed matches surfaced as compromises in section 1.
  ☐ No proper nouns in quotes.
  ☐ No invented numbers; every figure traceable to payload.
  ☐ No internal target leakage ("укладывается в лимит 500 м") — only user-evidenced numbers cited.
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
        "floor_current": item.get("floor_current"),
        "floors_total": item.get("floors_total"),
        "year_built": item.get("year_built"),
        "condition": item.get("condition"),
        "house_type": item.get("house_type"),
        "ceiling_height_m": item.get("ceiling_height_m"),
        "dist_to_center_km": item.get("dist_to_center_km"),
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
