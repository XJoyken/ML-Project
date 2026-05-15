from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Literal

from pydantic import BaseModel, Field

from ml_project.narrative import (
    FallbackReason,
    Language,
    NarrativeSource,
    _call_with_model_fallback,
    _classify_llm_error,
    _fallback_reason_message,
    _resolve_model_chain,
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
                model=llm_model,
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
    model: str | None,
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

    def _call(model_name: str) -> RecommendationNarrative:
        response = client.models.generate_content(
            model=model_name,
            contents=user_payload,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                response_mime_type="application/json",
                response_json_schema=RecommendationNarrative.model_json_schema(),
            ),
        )
        return RecommendationNarrative.model_validate_json(response.text)

    return _call_with_model_fallback(
        _call, _resolve_model_chain(model), log_prefix="recommender"
    )


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
like a thoughtful, attentive agent: warm, concrete, honest about trade-offs. Your single most important
job is to make the buyer feel HEARD — every wish they expressed must be acknowledged explicitly, and
every object you name (POI, ЖК, sensor, district landmark) must be paired with its exact distance.

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
3. Use EXACT numbers from the payload. Do NOT round distances, prices, PM2.5, areas, ceiling heights.
4. Distances: meters when < 1000 ("547 м"), kilometers with 1 decimal when ≥ 1000 ("1.3 км").
5. Prices: "38.5 млн ₸" (1 decimal); areas: "42.3 м²" (1 decimal); ceilings: "3.0 м".

6. ★ EVERY OBJECT YOU NAME MUST CARRY ITS EXACT DISTANCE.
   This is non-negotiable. If you mention a school, kindergarten, park, supermarket, fitness club,
   clinic, hospital, metro/bus stop, university, café, ЖК, air sensor or anything else from
   `nearby_pois` / `nearest_air_sensor`, immediately follow it with the distance in the formatted form.
   ❌ Bad: "рядом школа и парк" · "недалеко метро".
   ✅ Good: "школа №51 в 322 м, парк им. Ганди в 540 м, станция метро Райымбек в 1.1 км".
   If the payload does not contain a named POI for a category, DO NOT mention that category.
   Never use vague proximity words ("рядом", "недалеко", "поблизости") without a number next to them.

7. ★ ECHO EVERY USER WISH FROM `plan`.
   Walk through `plan.numeric_targets` and `plan.categorical_targets`. For EACH target with non-zero
   weight, the explanation must show what happened with it on THIS listing:
     - met → acknowledge ("2 комнаты — как и просили", "район Алмалинский — из вашего списка").
     - missed (hard_failed=true) → call the compromise out plainly with the gap
       ("комнат на 1 меньше", "цена на 3.2 млн выше бюджета", "до школы 740 м вместо желаемой шаговой доступности").
     - partially met → say so honestly ("чуть-чуть выше бюджета — на 600 тыс").
   When the user's `evidence` literally contains a number or phrase, you MAY quote it
   ("укладывается в ваше «до 40 млн»"). When evidence does not contain that number,
   DO NOT invent one — see rule 8.

8. ★ DO NOT EXPOSE INTERNAL NUMERIC TARGETS the user did not type in `evidence`.
   The system may have inserted defaults (e.g. "рядом школа" became `schools_nearest_dist_m at_most 500`).
   The user did NOT write "500 m" — never put it in the answer.
   ❌ Bad: "школа в 322 м, что укладывается в ваш лимит 500 м".
   ✅ Good: "школа №51 в 322 м — действительно рядом, как вы и хотели".
   This rule also applies to price targets when the user did not write a price evidence.

═══════════════════════════════════════════════════════════════════════════════════════
FIELD: summary  (2–4 sentences)
═══════════════════════════════════════════════════════════════════════════════════════
Compare the SET. State price range, room/area spread, district variety, and whether all listings
cover the user's hard requirements or some required compromises. Cite concrete numbers.
If excluded_districts is non-empty, briefly confirm those are filtered out.
NO filler like "отличная подборка". Be useful: tell the reader what makes the set diverse.

═══════════════════════════════════════════════════════════════════════════════════════
FIELD: items[].explanation  (TARGET: 7 sentences, range 7–9, ONE cohesive paragraph)
═══════════════════════════════════════════════════════════════════════════════════════
Write a single dense paragraph of around SEVEN sentences. Weave the six sections below with natural
Russian connectors (Также, Кроме того, При этом, Отдельно стоит отметить, Из минусов, Что касается воздуха).
Each sentence must carry real information — never filler, never generic praise. If a section can be
combined with the next in one fluent sentence, do so — but cover all six.

────────────────────────────────────────────────────────────────────────────────────
SECTION 1 — Headline + apartment essentials (1–2 sentences)
────────────────────────────────────────────────────────────────────────────────────
Open with rooms + (ЖК name if present, else тип дома) + price + district + floor (e.g. "4/9 этаж").
Immediately add the apartment's defining numbers: area (м²), area per room if revealing,
year_built, ceiling_height_m if available. Example opener (RU):
"3-комнатная за 52.4 млн ₸ в ЖК Алмалы — 78.5 м², 6/12 этаж, 2021 год постройки, потолки 3.0 м,
расположен в Алмалинском районе."

────────────────────────────────────────────────────────────────────────────────────
SECTION 2 — How this listing meets the user's wishes (1–2 sentences)
────────────────────────────────────────────────────────────────────────────────────
Walk through the active user targets from `plan` (see ABSOLUTE RULE 7). For each evidenced wish
say plainly whether it is met, missed (with numeric gap), or partial. If the user evidenced a POI
category (например, "хочу рядом школу"), name the actual POI and its distance HERE in addition
to repeating it in Section 3. Lead with the wishes the user weighted highest.

────────────────────────────────────────────────────────────────────────────────────
SECTION 3 — Locality (1–2 sentences, ≥3 distinct POI categories, distances mandatory)
────────────────────────────────────────────────────────────────────────────────────
Pick from `nearby_pois`. Every POI you name MUST appear in the form
"<категория> <название> в <дистанция>". Cover ≥3 categories from different needs groups:
семьи (schools/kindergartens), быт (parks/supermarkets/fitness/restaurants_coffee),
транспорт (metro/bus_stops/transport), медицина (clinics/polyclinics/hospitals/dentistry/medcenters),
образование (universities).
For bus_stops: ALWAYS list routes in parentheses — "автобусы 18, 95; троллейбус Тр6"
(префикс "Тр" обозначает троллейбус). For metro: name the station.

────────────────────────────────────────────────────────────────────────────────────
SECTION 4 — Apartment-specific plus features (1–2 sentences, REQUIRED)
────────────────────────────────────────────────────────────────────────────────────
Dig into what makes THIS apartment special. Pick 2–4 of:
  - Seller `description` perks: "евроремонт", "панорамные окна", "вид на горы", "охраняемый двор",
    "гардеробная", "тёплый пол", "новая сантехника", "сигнализация", "встроенная кухня",
    "большой балкон", "лоджия застеклена", "паркинг", "пластиковые окна".
    Lead with "Продавец отдельно отмечает: …" (RU) or "Seller notes: …" (EN). Quote concrete phrases.
  - Building age verdict: ≥ 2020 = свежая новостройка; 2015–2019 = современный дом;
    2008–2014 = относительно новый; 1990–2007 = типовой постсоветский; < 1990 = старый фонд.
  - house_type narrative: монолит / кирпич — премиум; панель — стандарт; каркасный — лёгкая
    шумоизоляция. Cite it if known.
  - Floor placement narrative: 1-й — спорно (шум/безопасность); 2–7 в высотке — оптимально;
    верхний этаж — вид, но течёт крыша риск.
  - condition (евроремонт / хорошее / отличное / свежий ремонт) — quote literally if present.
  - ceiling_height_m ≥ 3.0 → подчеркни как премиум-признак.
  - bathroom_type / parking / balcony, если эти поля есть в `matches` или `description`.
  - dist_to_center_km ≤ 3 → "в центре"; ≤ 5 → "ближе к центру"; ≥ 10 → "на периферии".
  - area_per_room ≥ 25 м² → "просторные комнаты"; < 14 м² → "комнаты компактные".
If nothing distinguishing is in the payload, write ONE sentence with the strongest neutral perk
(e.g. price-per-area или новый дом). Never skip this section.

────────────────────────────────────────────────────────────────────────────────────
SECTION 5 — Compromises / minuses for THIS listing (1 sentence, REQUIRED)
────────────────────────────────────────────────────────────────────────────────────
Be honest. List 1–2 real drawbacks (do NOT invent). Pick from:
  - Hard-failed targets ("комнат меньше на 1, чем просили", "выше бюджета на 3.2 млн").
  - Far POIs vs typical needs: kindergartens/schools > 1500 m, parks > 1000 m,
    transport/bus_stops > 800 m, supermarkets > 1000 m, clinics > 2000 m — cite the exact distance.
  - Старый фонд (year_built < 1980), типовой панельный дом без ремонта.
  - Edge floors (1-й или последний без обратного желания пользователя).
  - High PM2.5 (cold > 40 = умеренный/плохой).
  - Highest price in the set, smallest area, longest commute to center vs the rest.
  - Condition "поднимет" / "под ремонт" / "требует ремонта".
If you genuinely cannot find a drawback, write ONE sentence like:
  "Очевидных минусов по данным нет — стоит лично проверить состояние при просмотре."
Never SKIP this section.

────────────────────────────────────────────────────────────────────────────────────
SECTION 6 — Air quality (REQUIRED when air data present, 1 sentence)
────────────────────────────────────────────────────────────────────────────────────
Both PM2.5 numbers + sensor distance + verdict tier:
  ≤15 — отличный · ≤25 — хороший · ≤40 — умеренный · ≤60 — плохой · >60 — очень плохой.
Example (RU): "Воздух у дома (датчик в 420 м) — PM2.5 22 (хороший) зимой и 17 (хороший) летом."

═══════════════════════════════════════════════════════════════════════════════════════
FIELD: items[].url
═══════════════════════════════════════════════════════════════════════════════════════
Exactly "https://krisha.kz/a/show/{{listing_id}}".

{air_priority_instruction}

═══════════════════════════════════════════════════════════════════════════════════════
WORKED OUTPUT EXAMPLE (RU, ~7 sentences)
═══════════════════════════════════════════════════════════════════════════════════════
3-комнатная за 52.4 млн ₸ в ЖК Алмалы — 78.5 м², 6/12 этаж, 2021 год постройки, монолит, потолки 3.0 м, Алмалинский район. По вашим пожеланиям всё совпало: 3 комнаты как и просили, цена укладывается в ваше «до 55 млн», а ваше «хочу рядом школу» подтверждается — школа-гимназия №51 в 322 м, то есть в 4 минутах пешком. Из бытовой инфраструктуры в шаговой доступности: детский сад Балапан в 180 м, парк им. Ганди в 540 м, супермаркет Magnum в 290 м и остановка Музей искусств в 210 м (автобусы 18, 95, 121; троллейбус Тр6). Продавец отдельно отмечает: евроремонт, панорамные окна с видом на горы, охраняемый двор и гардеробная — для дома 2021 года и монолита это полный пакет. Этаж 6 из 12 удачный — не первый и не последний, плюс высота потолков 3.0 м делает квартиру визуально просторнее. Из минусов — поликлиника далековата (1.6 км) и квартира в верхней половине вашего бюджета. Воздух у дома (датчик в 420 м) — PM2.5 22 (хороший) зимой и 17 (хороший) летом.

═══════════════════════════════════════════════════════════════════════════════════════
QUALITY CHECKLIST — MENTAL PASS BEFORE EMITTING JSON
═══════════════════════════════════════════════════════════════════════════════════════
For EVERY item, verify:
  ☐ Length is ~7 sentences (7–9 acceptable). No filler, every sentence carries data.
  ☐ Section 1: rooms / ЖК / price / district / floor / area / year / ceiling stated.
  ☐ Section 2: every weighted user wish from plan addressed (met / partial / missed with gap).
  ☐ Section 3: ≥3 POI categories, EACH with name + exact distance, no internal-target leakage.
  ☐ Section 4: ≥2 concrete apartment-specific pros (description / year / house_type / floor /
    condition / ceilings / area-per-room / parking / balcony).
  ☐ Section 5: ≥1 concrete con (or honest "no obvious cons in data").
  ☐ Section 6: BOTH PM2.5 numbers, sensor distance, AND the tier word.
  ☐ Hard-failed matches surfaced explicitly with numeric gaps.
  ☐ EVERY named POI / sensor / landmark carries a distance — no vague "рядом" anywhere.
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
