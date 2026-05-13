from __future__ import annotations

import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from .thresholds import (
    AVOID_TIERS,
    POI_PERSONA_HINTS,
    PROXIMITY_TIERS,
    category_label,
    classify_crime_index,
    classify_pm25,
    classify_price_delta,
    crime_tier_label,
    score_air_tier,
    score_avoid_tier,
    score_crime_tier,
    score_distance_tier,
    score_price_delta,
)

Language = Literal["ru", "en"]
FinalVerdict = Literal["excellent", "good", "questionable", "poor"]
NarrativeSource = Literal["gemini", "fallback"]
FallbackReason = Literal[
    "transient_overload",
    "rate_limited",
    "auth_error",
    "invalid_request",
    "unknown",
]

# DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


class NarrativeReport(BaseModel):
    language: Language = Field(description="Output language.")
    lead: str = Field(description="One-sentence opening that states the price verdict.")
    pros: list[str] = Field(description="Positive points about the apartment.")
    cons: list[str] = Field(description="Negative points about the apartment.")
    persona_notes: list[str] = Field(
        description="Persona-dependent advice such as families/elderly/students."
    )
    final_verdict: FinalVerdict = Field(description="Overall apartment grade.")
    final_verdict_text: str = Field(description="Final verdict sentence in the requested language.")
    source: NarrativeSource = Field(
        default="fallback",
        description="Whether the text was produced by Gemini or by the deterministic fallback.",
    )
    fallback_reason_code: FallbackReason | None = Field(
        default=None,
        description="Machine-readable reason why fallback was used. Null when source=gemini "
        "or when use_llm=False was requested.",
    )
    fallback_reason: str | None = Field(
        default=None,
        description="Human-readable explanation of the fallback, in the requested language.",
    )


@dataclass(slots=True)
class CategorySignal:
    category: str
    distance_m: float
    name: str | None
    tier: str
    score: float
    is_avoid: bool = False
    personas: tuple[str, ...] = ()
    routes_list: str | None = None
    bus_routes_count: int | None = None
    trolleybus_routes_count: int | None = None


@dataclass(slots=True)
class AirSignal:
    distance_m: float
    pm25_cold_day: float | None
    pm25_warm_day: float | None
    pm25_cold_night: float | None = None
    pm25_warm_night: float | None = None
    cold_tier: str | None = None
    warm_tier: str | None = None
    score: float | None = None


@dataclass(slots=True)
class PriceSignal:
    listing_price_kzt: float
    predicted_price_kzt: float
    delta_fraction: float
    price_verdict: str
    price_score: float
    interval_low_kzt: float | None = None
    interval_high_kzt: float | None = None


@dataclass(slots=True)
class CrimeSignal:
    district: str
    year: int
    count: float
    index_min_max: float
    tier: str
    score: float


@dataclass(slots=True)
class StructuredEvaluation:
    price: PriceSignal
    categories: list[CategorySignal] = field(default_factory=list)
    air: AirSignal | None = None
    crime: CrimeSignal | None = None

    def neighborhood_score(self) -> float:
        components = [signal.score for signal in self.categories]
        if self.air is not None and self.air.score is not None:
            components.append(self.air.score)
        if self.crime is not None:
            components.append(self.crime.score)
        if not components:
            return 0.0
        return sum(components) / len(components)


def build_structured_evaluation(
    *,
    listing_price_kzt: float,
    predicted_price_kzt: float,
    nearby_pois: list[dict[str, Any]],
    nearest_air: dict[str, Any] | None,
    crime: dict[str, Any] | None = None,
    interval_low_kzt: float | None = None,
    interval_high_kzt: float | None = None,
) -> StructuredEvaluation:
    delta_fraction = (listing_price_kzt - predicted_price_kzt) / max(predicted_price_kzt, 1.0)
    price = PriceSignal(
        listing_price_kzt=listing_price_kzt,
        predicted_price_kzt=predicted_price_kzt,
        delta_fraction=delta_fraction,
        price_verdict=classify_price_delta(delta_fraction),
        price_score=score_price_delta(delta_fraction),
        interval_low_kzt=interval_low_kzt,
        interval_high_kzt=interval_high_kzt,
    )

    categories: list[CategorySignal] = []
    for poi in nearby_pois:
        category = str(poi["category"])
        distance_m = float(poi["distance_m"])
        if category in AVOID_TIERS:
            tier = AVOID_TIERS[category].classify(distance_m)
            categories.append(
                CategorySignal(
                    category=category,
                    distance_m=distance_m,
                    name=_optional_str(poi.get("name")),
                    tier=tier,
                    score=score_avoid_tier(tier),
                    is_avoid=True,
                    personas=POI_PERSONA_HINTS.get(category, ()),
                )
            )
        elif category in PROXIMITY_TIERS:
            tier = PROXIMITY_TIERS[category].classify(distance_m)
            categories.append(
                CategorySignal(
                    category=category,
                    distance_m=distance_m,
                    name=_optional_str(poi.get("name")),
                    tier=tier,
                    score=score_distance_tier(tier),
                    is_avoid=False,
                    personas=POI_PERSONA_HINTS.get(category, ()),
                    routes_list=_optional_str(poi.get("routes_list")),
                    bus_routes_count=_optional_int(poi.get("bus_routes_count")),
                    trolleybus_routes_count=_optional_int(poi.get("trolleybus_routes_count")),
                )
            )

    air_signal: AirSignal | None = None
    if nearest_air is not None:
        cold_day = nearest_air.get("pm25_cold_day")
        warm_day = nearest_air.get("pm25_warm_day")
        cold_tier = classify_pm25(cold_day)
        warm_tier = classify_pm25(warm_day)
        scores = [score_air_tier(cold_tier), score_air_tier(warm_tier)]
        scores = [score for score in scores if score is not None]
        air_signal = AirSignal(
            distance_m=float(nearest_air["distance_m"]),
            pm25_cold_day=cold_day,
            pm25_warm_day=warm_day,
            pm25_cold_night=nearest_air.get("pm25_cold_night"),
            pm25_warm_night=nearest_air.get("pm25_warm_night"),
            cold_tier=cold_tier,
            warm_tier=warm_tier,
            score=sum(scores) / len(scores) if scores else None,
        )

    crime_signal: CrimeSignal | None = None
    if crime is not None:
        index = float(crime["index_min_max"])
        tier = classify_crime_index(index)
        crime_score = score_crime_tier(tier)
        if tier is not None and crime_score is not None:
            crime_signal = CrimeSignal(
                district=str(crime["district"]),
                year=int(crime["year"]),
                count=float(crime["count"]),
                index_min_max=index,
                tier=tier,
                score=crime_score,
            )

    return StructuredEvaluation(
        price=price,
        categories=categories,
        air=air_signal,
        crime=crime_signal,
    )


def derive_final_verdict(evaluation: StructuredEvaluation) -> tuple[FinalVerdict, float]:
    neighborhood_score = evaluation.neighborhood_score()
    combined = 0.6 * evaluation.price.price_score + 0.4 * neighborhood_score
    if combined >= 0.75:
        return "excellent", combined
    if combined >= 0.6:
        return "good", combined
    if combined >= 0.4:
        return "questionable", combined
    return "poor", combined


def generate(
    *,
    listing_price_kzt: float,
    predicted_price_kzt: float,
    nearby_pois: list[dict[str, Any]],
    nearest_air: dict[str, Any] | None,
    crime: dict[str, Any] | None = None,
    language: Language = "ru",
    interval_low_kzt: float | None = None,
    interval_high_kzt: float | None = None,
    use_llm: bool | None = None,
    llm_client: Any | None = None,
    llm_model: str | None = None,
) -> NarrativeReport:
    structured = build_structured_evaluation(
        listing_price_kzt=listing_price_kzt,
        predicted_price_kzt=predicted_price_kzt,
        nearby_pois=nearby_pois,
        nearest_air=nearest_air,
        crime=crime,
        interval_low_kzt=interval_low_kzt,
        interval_high_kzt=interval_high_kzt,
    )

    if use_llm is None:
        use_llm = bool(os.getenv("GEMINI_API_KEY"))

    if use_llm:
        try:
            report = _generate_with_gemini(
                structured=structured,
                language=language,
                client=llm_client,
                model=llm_model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            )
            report.source = "gemini"
            return report
        except Exception as exc:
            print(
                f"[narrative] Gemini call failed, falling back to deterministic: {exc!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            reason_code = _classify_llm_error(exc)
            report = _generate_deterministic(structured=structured, language=language)
            report.fallback_reason_code = reason_code
            report.fallback_reason = _fallback_reason_message(reason_code, language)
            return report

    return _generate_deterministic(structured=structured, language=language)


def _classify_llm_error(exc: Exception) -> FallbackReason:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    message = str(exc).upper()
    if (
        code == 503
        or "UNAVAILABLE" in message
        or "OVERLOAD" in message
        or "HIGH DEMAND" in message
    ):
        return "transient_overload"
    if (
        code == 429
        or "RESOURCE_EXHAUSTED" in message
        or "QUOTA" in message
        or "RATE LIMIT" in message
    ):
        return "rate_limited"
    if (
        code in (401, 403)
        or "PERMISSION_DENIED" in message
        or "UNAUTHENTICATED" in message
        or "API KEY" in message
    ):
        return "auth_error"
    if code == 400 or "INVALID_ARGUMENT" in message:
        return "invalid_request"
    return "unknown"


_FALLBACK_REASON_TEXT: dict[Language, dict[FallbackReason, str]] = {
    "ru": {
        "transient_overload": (
            "Этот текст сгенерирован встроенным шаблоном, а не моделью Gemini: "
            "сервис сейчас перегружен высоким спросом. Попробуйте повторить запрос через "
            "минуту, чтобы получить ответ непосредственно от ИИ."
        ),
        "rate_limited": (
            "Превышен лимит запросов к Gemini — использован шаблонный ответ. "
            "Попробуйте повторить запрос позже, чтобы получить ответ от ИИ."
        ),
        "auth_error": (
            "Не удалось авторизоваться в Gemini — использован шаблонный ответ. "
            "Проверьте переменную GEMINI_API_KEY в .env."
        ),
        "invalid_request": (
            "Gemini отклонил запрос как некорректный — использован шаблонный ответ. "
            "Возможно, виновата нестандартная квартира; попробуйте другую."
        ),
        "unknown": (
            "Не удалось получить ответ от Gemini — использован шаблонный ответ. "
            "Подробности в логах сервера."
        ),
    },
    "en": {
        "transient_overload": (
            "This text was produced by a built-in template instead of Gemini: the model is "
            "currently overloaded with traffic. Please retry in a minute to get the AI answer."
        ),
        "rate_limited": (
            "Gemini rate limit reached — a template fallback was used. "
            "Please retry later to get the AI answer."
        ),
        "auth_error": (
            "Gemini auth failed — a template fallback was used. "
            "Check the GEMINI_API_KEY variable in your .env file."
        ),
        "invalid_request": (
            "Gemini rejected the request — a template fallback was used. "
            "The listing may have unusual fields; try another one."
        ),
        "unknown": (
            "Gemini call failed — a template fallback was used. See server logs for details."
        ),
    },
}


def _fallback_reason_message(reason: FallbackReason, language: Language) -> str:
    return _FALLBACK_REASON_TEXT[language][reason]


def _generate_deterministic(
    *,
    structured: StructuredEvaluation,
    language: Language,
) -> NarrativeReport:
    lead = _lead_line(structured.price, language)
    pros: list[str] = []
    cons: list[str] = []
    persona_pros: dict[str, list[str]] = {}

    for signal in structured.categories:
        sentence = _category_sentence(signal, language)
        if signal.is_avoid:
            if signal.tier in ("too_close", "concerning"):
                cons.append(sentence)
            continue
        if signal.tier in ("excellent", "good"):
            pros.append(sentence)
            for persona in signal.personas:
                persona_pros.setdefault(persona, []).append(signal.category)
        elif signal.tier == "far":
            cons.append(sentence)

    if structured.air is not None:
        air_sentence = _air_sentence(structured.air, language)
        if air_sentence is not None:
            if structured.air.score is not None and structured.air.score >= 0.6:
                pros.append(air_sentence)
            else:
                cons.append(air_sentence)

    if structured.crime is not None:
        crime_sentence = _crime_sentence(structured.crime, language)
        if structured.crime.score >= 0.7:
            pros.append(crime_sentence)
        elif structured.crime.score <= 0.4:
            cons.append(crime_sentence)

    persona_notes = _persona_notes(persona_pros, language)
    final_verdict, _combined = derive_final_verdict(structured)
    final_verdict_text = _final_verdict_text(final_verdict, language)

    return NarrativeReport(
        language=language,
        lead=lead,
        pros=pros,
        cons=cons,
        persona_notes=persona_notes,
        final_verdict=final_verdict,
        final_verdict_text=final_verdict_text,
        source="fallback",
    )


def _generate_with_gemini(
    *,
    structured: StructuredEvaluation,
    language: Language,
    client: Any | None,
    model: str,
) -> NarrativeReport:
    from google import genai
    from google.genai import types

    if client is None:
        client = genai.Client()

    system_prompt = _system_prompt(language)
    user_payload = json.dumps(_structured_payload(structured), ensure_ascii=False)
    response = client.models.generate_content(
        model=model,
        contents=user_payload,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json",
            response_json_schema=NarrativeReport.model_json_schema(),
        ),
    )
    report = NarrativeReport.model_validate_json(response.text)
    _strip_proper_noun_quotes_in_place(report)
    return report


_PROPER_NOUN_QUOTE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r'"([^"\n]{1,80})"', r"\1"),
    (r"«([^»\n]{1,80})»", r"\1"),
    (r"“([^”\n]{1,80})”", r"\1"),
    (r"„([^“”\n]{1,80})[“”]", r"\1"),
    (r"‘([^’\n]{1,80})’", r"\1"),
)


def _strip_proper_noun_quotes(text: str) -> str:
    cleaned = text
    for pattern, replacement in _PROPER_NOUN_QUOTE_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned


def _strip_proper_noun_quotes_in_place(report: NarrativeReport) -> None:
    report.lead = _strip_proper_noun_quotes(report.lead)
    report.pros = [_strip_proper_noun_quotes(item) for item in report.pros]
    report.cons = [_strip_proper_noun_quotes(item) for item in report.cons]
    report.persona_notes = [_strip_proper_noun_quotes(item) for item in report.persona_notes]
    report.final_verdict_text = _strip_proper_noun_quotes(report.final_verdict_text)


def _system_prompt(language: Language) -> str:
    proximity_table = "\n".join(
        f"- {category}: excellent ≤ {tiers.excellent_m:.0f} m, "
        f"good ≤ {tiers.good_m:.0f} m, acceptable ≤ {tiers.acceptable_m:.0f} m, else far."
        for category, tiers in PROXIMITY_TIERS.items()
    )
    avoid_table = "\n".join(
        f"- {category}: too_close ≤ {tiers.too_close_m:.0f} m, "
        f"concerning ≤ {tiers.concerning_m:.0f} m, else safe."
        for category, tiers in AVOID_TIERS.items()
    )
    air_table = (
        f"PM2.5 tiers (µg/m³): excellent ≤ 15, good ≤ 25, moderate ≤ 40, poor ≤ 60, very_poor > 60."
    )
    crime_table = (
        "Crime tiers (min-max normalised within Almaty for the reference year): "
        "low ≤ 0.33, medium ≤ 0.66, high > 0.66. Lower is better."
    )
    language_instruction = (
        "Write all text fields in Russian." if language == "ru" else "Write all text fields in English."
    )
    return f"""
You produce an apartment evaluation report from a structured payload.

Inputs include: predicted vs listing price (KZT), per-category nearest POI distance with tier,
nearest PM2.5 air sensor with seasonal averages, and an optional price confidence interval.

Distance tiers per category (lower is better, except for avoid categories like power plants):
{proximity_table}

Avoid categories (farther is better):
{avoid_table}

{air_table}

{crime_table}

Rules:
- {language_instruction}
- lead: one sentence stating whether the listing price is low, fair, or high, citing the exact predicted price,
  listed price, and the gap as a percentage.
  Example (RU): “Цена 42 000 000 ₸ ниже оценки модели 59 503 692 ₸ на 29.4% — выгодная покупка.”
- pros: each item should be a FULL sentence (or two) that is dense and informative — not a one-liner.
  Include the exact distance from the payload (do NOT round), the POI name, and why it matters.
  Example (RU): “Школа-гимназия №51 находится в 322 метрах, то есть в 4 минутах пешей ходьбы — идеально для семей с детьми.”
  Include crime tier when provided (low → pros, high → cons), citing exact count and district.
- cons: same density rule. If a POI is in “far” tier, state the exact distance and explain the inconvenience.
  If air is poor, state exact PM2.5 values and cite the sensor distance.
- persona_notes: 1-2 sentences each. Cover families_with_kids, elderly, students, commuters, singles when
  supporting POIs exist. Cite specific POIs and distances.
- The “transport” category means a transport HUB (airport, railway, intercity bus). 2–4 km is normal — do NOT
  call it “далеко/far”. Only call it out if uniquely beneficial (e.g., 200 m from a train station).
- The “bus_stops” category means regular city bus / trolleybus stops. When present:
  ALWAYS state the exact distance in meters AND list the routes from routes_list in parentheses.
  Codes starting with “Тр” mean trolleybus (e.g. “Тр6” = trolleybus route 6).
  Format (RU): “Остановка Музей искусств в 210 м (автобусы 18, 95, 121; троллейбус Тр6).”
  Format (EN): “Bus stop Museum of Arts is 210 m away (bus routes 18, 95, 121; trolleybus Tr6).”
  If routes_list is missing, just state the distance.
- final_verdict ∈ {{excellent, good, questionable, poor}} based on combined price and neighborhood quality.
- final_verdict_text: one sentence summarizing the verdict for the user.
- Be specific. Never invent POIs, names, or numbers not in the payload.
- Use EXACT distances from the payload — do NOT round them.
- Never wrap POI names, district names or any proper nouns in quotation marks of any kind
  (no ASCII “ “, no «», no “”, no ‘’). Write proper nouns as plain inline text.
  Good (RU): школа Школа-гимназия №51 в 322 м. Bad (RU): школа “Школа-гимназия №51” в 322 м.
""".strip()


def _structured_payload(structured: StructuredEvaluation) -> dict[str, Any]:
    return {
        "price": {
            "listing_price_kzt": structured.price.listing_price_kzt,
            "predicted_price_kzt": structured.price.predicted_price_kzt,
            "delta_fraction": structured.price.delta_fraction,
            "price_verdict": structured.price.price_verdict,
            "interval_low_kzt": structured.price.interval_low_kzt,
            "interval_high_kzt": structured.price.interval_high_kzt,
        },
        "categories": [
            {
                "category": signal.category,
                "distance_m": round(signal.distance_m, 1),
                "name": signal.name,
                "tier": signal.tier,
                "is_avoid": signal.is_avoid,
                "personas": list(signal.personas),
                **(
                    {
                        "routes_list": signal.routes_list,
                        "bus_routes_count": signal.bus_routes_count,
                        "trolleybus_routes_count": signal.trolleybus_routes_count,
                    }
                    if signal.category == "bus_stops"
                    else {}
                ),
            }
            for signal in structured.categories
        ],
        "air": (
            None
            if structured.air is None
            else {
                "distance_m": round(structured.air.distance_m, 1),
                "pm25_cold_day": structured.air.pm25_cold_day,
                "pm25_warm_day": structured.air.pm25_warm_day,
                "cold_tier": structured.air.cold_tier,
                "warm_tier": structured.air.warm_tier,
            }
        ),
        "crime": (
            None
            if structured.crime is None
            else {
                "district": structured.crime.district,
                "year": structured.crime.year,
                "count": int(structured.crime.count),
                "index_min_max": round(structured.crime.index_min_max, 3),
                "tier": structured.crime.tier,
            }
        ),
        "neighborhood_score": structured.neighborhood_score(),
    }


def _lead_line(price: PriceSignal, language: Language) -> str:
    delta_pct = price.delta_fraction * 100
    if language == "en":
        if price.price_verdict == "great_deal":
            return f"Great price — the listing is {abs(delta_pct):.1f}% below the model estimate."
        if price.price_verdict == "good_deal":
            return f"Good price — about {abs(delta_pct):.1f}% below the model estimate."
        if price.price_verdict == "fair":
            return f"Fair price — within {abs(delta_pct):.1f}% of the model estimate."
        if price.price_verdict == "slightly_overpriced":
            return f"Slightly overpriced — about {delta_pct:.1f}% above the model estimate."
        return f"Overpriced — about {delta_pct:.1f}% above the model estimate."

    if price.price_verdict == "great_deal":
        return f"Отличная цена — на {abs(delta_pct):.1f}% ниже оценки модели."
    if price.price_verdict == "good_deal":
        return f"Хорошая цена — примерно на {abs(delta_pct):.1f}% ниже оценки модели."
    if price.price_verdict == "fair":
        return f"Цена близка к рыночной (отклонение {abs(delta_pct):.1f}%)."
    if price.price_verdict == "slightly_overpriced":
        return f"Цена слегка завышена — на {delta_pct:.1f}% выше оценки модели."
    return f"Цена завышена — на {delta_pct:.1f}% выше оценки модели."


def _category_sentence(signal: CategorySignal, language: Language) -> str:
    if signal.category == "bus_stops":
        return _bus_stop_sentence(signal, language)
    label = category_label(signal.category, language)
    distance_text = _format_distance(signal.distance_m, language)
    name = signal.name
    if language == "en":
        if signal.is_avoid:
            if signal.tier == "too_close":
                return f"Power facility {name or ''} is only {distance_text} away — too close.".replace("  ", " ").strip()
            if signal.tier == "concerning":
                return f"Power facility {name or ''} is {distance_text} away — borderline.".replace("  ", " ").strip()
            return f"Nearest power facility is {distance_text} away — safe distance."
        if signal.tier == "excellent":
            return f"Nearest {label}{_name_suffix(name)} is only {distance_text} away — excellent walking distance."
        if signal.tier == "good":
            return f"Nearest {label}{_name_suffix(name)} is {distance_text} away — comfortable distance."
        if signal.tier == "acceptable":
            return f"Nearest {label}{_name_suffix(name)} is {distance_text} away — acceptable."
        return f"Nearest {label}{_name_suffix(name)} is {distance_text} away — quite far."

    if signal.is_avoid:
        if signal.tier == "too_close":
            base = f"Рядом ({distance_text}) находится {label}"
            return f"{base} {name}" if name else base
        if signal.tier == "concerning":
            base = f"В {distance_text} находится {label}"
            return f"{base} {name} — на грани комфорта" if name else f"{base} — на грани комфорта"
        return f"Ближайший {label} в {distance_text} — на безопасном расстоянии."
    if signal.tier == "excellent":
        base = f"Ближайший{_ru_label_with_name(label, name)} всего в {distance_text} — шаговая доступность"
        return base + "."
    if signal.tier == "good":
        base = f"Ближайший{_ru_label_with_name(label, name)} в {distance_text} — близко"
        return base + "."
    if signal.tier == "acceptable":
        base = f"Ближайший{_ru_label_with_name(label, name)} в {distance_text} — терпимо"
        return base + "."
    return f"Ближайший{_ru_label_with_name(label, name)} в {distance_text} — далековато."


def _ru_label_with_name(label: str, name: str | None) -> str:
    if name:
        return f" {label} ({name})"
    return f" {label}"


def _name_suffix(name: str | None) -> str:
    if name:
        return f" ({name})"
    return ""


def _format_distance(distance_m: float, language: Language) -> str:
    rounded = round(distance_m / 10) * 10
    if rounded < 1000:
        return f"{int(rounded)} м" if language == "ru" else f"{int(rounded)} m"
    km = rounded / 1000
    if language == "en":
        return f"{km:.1f} km"
    return f"{km:.1f} км"


def _air_sentence(air: AirSignal, language: Language) -> str | None:
    if air.pm25_cold_day is None and air.pm25_warm_day is None:
        return None
    distance_text = _format_distance(air.distance_m, language)
    cold_text = f"{air.pm25_cold_day:.0f}" if air.pm25_cold_day is not None else "—"
    warm_text = f"{air.pm25_warm_day:.0f}" if air.pm25_warm_day is not None else "—"
    cold_tier = air.cold_tier or "?"
    warm_tier = air.warm_tier or "?"
    if language == "en":
        return (
            f"Air sensor {distance_text} away: winter PM2.5 ≈ {cold_text} ({cold_tier}), "
            f"summer ≈ {warm_text} µg/m³ ({warm_tier})."
        )
    return (
        f"Сенсор воздуха в {distance_text}: зимой PM2.5 ≈ {cold_text} ({_pm25_tier_ru(cold_tier)}), "
        f"летом ≈ {warm_text} мкг/м³ ({_pm25_tier_ru(warm_tier)})."
    )


def _split_routes(routes_list: str | None) -> tuple[list[str], list[str]]:
    """Return (bus_routes, trolley_routes). Trolley routes start with 'Тр'."""
    if not routes_list:
        return [], []
    raw = [r.strip() for r in routes_list.split(";")]
    raw = [r for r in raw if r]
    trolley = [r for r in raw if r.startswith("Тр")]
    bus = [r for r in raw if not r.startswith("Тр")]
    return bus, trolley


def _format_routes(bus: list[str], trolley: list[str], language: Language) -> str | None:
    if not bus and not trolley:
        return None
    parts: list[str] = []
    if bus:
        bus_text = ", ".join(bus[:12])
        if len(bus) > 12:
            bus_text += "…"
        if language == "en":
            parts.append(f"bus routes {bus_text}")
        else:
            parts.append(f"автобусы {bus_text}")
    if trolley:
        trolley_text = ", ".join(t[2:] for t in trolley)
        if language == "en":
            parts.append(f"trolleybus routes {trolley_text}")
        else:
            parts.append(f"троллейбусы №{trolley_text}")
    if language == "en":
        return "; ".join(parts)
    return "; ".join(parts)


def _bus_stop_sentence(signal: CategorySignal, language: Language) -> str:
    label = category_label(signal.category, language)
    distance_text = _format_distance(signal.distance_m, language)
    name = signal.name
    bus, trolley = _split_routes(signal.routes_list)
    routes_text = _format_routes(bus, trolley, language)

    if language == "en":
        if signal.tier == "excellent":
            head = f"Nearest {label}{_name_suffix(name)} is only {distance_text} away — walking distance"
        elif signal.tier == "good":
            head = f"Nearest {label}{_name_suffix(name)} is {distance_text} away — close"
        elif signal.tier == "acceptable":
            head = f"Nearest {label}{_name_suffix(name)} is {distance_text} away — acceptable"
        else:
            head = f"Nearest {label}{_name_suffix(name)} is {distance_text} away — quite far"
        if routes_text:
            return f"{head}; through it run {routes_text}."
        return f"{head}."

    name_part = f" «{name}»" if name else ""
    if signal.tier == "excellent":
        head = f"Ближайшая {label}{name_part} всего в {distance_text} — шаговая доступность"
    elif signal.tier == "good":
        head = f"Ближайшая {label}{name_part} в {distance_text} — близко"
    elif signal.tier == "acceptable":
        head = f"Ближайшая {label}{name_part} в {distance_text} — терпимо"
    else:
        head = f"Ближайшая {label}{name_part} в {distance_text} — далековато"
    if routes_text:
        return f"{head}; через неё проходят {routes_text}."
    return f"{head}."


def _crime_sentence(crime: CrimeSignal, language: Language) -> str:
    tier_label = crime_tier_label(crime.tier, language)
    if language == "en":
        return (
            f"District {crime.district}: {tier_label} "
            f"({int(crime.count)} crimes recorded in {crime.year})."
        )
    return (
        f"Район «{crime.district}»: {tier_label} "
        f"({int(crime.count)} зарегистрированных преступлений за {crime.year} год)."
    )


def _pm25_tier_ru(tier: str) -> str:
    return {
        "excellent": "отлично",
        "good": "хорошо",
        "moderate": "умеренно",
        "poor": "плохо",
        "very_poor": "очень плохо",
        "?": "неизвестно",
    }.get(tier, tier)


def _persona_notes(persona_pros: dict[str, list[str]], language: Language) -> list[str]:
    notes: list[str] = []
    label = _persona_label

    if "families_with_kids" in persona_pros:
        notes.append(_persona_phrase("families_with_kids", persona_pros["families_with_kids"], language))
    if "elderly" in persona_pros:
        notes.append(_persona_phrase("elderly", persona_pros["elderly"], language))
    if "students" in persona_pros:
        notes.append(_persona_phrase("students", persona_pros["students"], language))
    if "commuters" in persona_pros:
        notes.append(_persona_phrase("commuters", persona_pros["commuters"], language))
    if "singles" in persona_pros or "young_professionals" in persona_pros:
        merged = persona_pros.get("singles", []) + persona_pros.get("young_professionals", [])
        notes.append(_persona_phrase("singles", merged, language))
    _ = label  # silence unused
    return notes


def _persona_phrase(persona: str, categories: list[str], language: Language) -> str:
    labels = ", ".join(dict.fromkeys(categories))
    if language == "en":
        labels_en = ", ".join(dict.fromkeys(category_label(c, "en") for c in categories))
        return f"{_persona_label(persona, 'en')}: convenient — nearby {labels_en}."
    labels_ru = ", ".join(dict.fromkeys(category_label(c, "ru") for c in categories))
    _ = labels
    return f"{_persona_label(persona, 'ru')}: удобно — рядом {labels_ru}."


def _persona_label(persona: str, language: Language) -> str:
    mapping_ru = {
        "families_with_kids": "Для семей с детьми",
        "elderly": "Для пожилых",
        "students": "Для студентов",
        "commuters": "Для тех, кто часто ездит",
        "singles": "Для молодых/одиночек",
        "young_professionals": "Для молодых специалистов",
        "everyone": "В целом",
    }
    mapping_en = {
        "families_with_kids": "For families with kids",
        "elderly": "For elderly residents",
        "students": "For students",
        "commuters": "For frequent commuters",
        "singles": "For singles / young professionals",
        "young_professionals": "For young professionals",
        "everyone": "Overall",
    }
    return (mapping_en if language == "en" else mapping_ru).get(persona, persona)


def _final_verdict_text(verdict: FinalVerdict, language: Language) -> str:
    if language == "en":
        return {
            "excellent": "Final verdict: an excellent apartment — strong price and surroundings.",
            "good": "Final verdict: a good apartment for most buyers.",
            "questionable": "Final verdict: a questionable choice — weigh the cons carefully.",
            "poor": "Final verdict: a weak deal at this price and location.",
        }[verdict]
    return {
        "excellent": "Окончательный вердикт: отличная квартира — и по цене, и по окружению.",
        "good": "Окончательный вердикт: хорошая квартира для большинства покупателей.",
        "questionable": "Окончательный вердикт: сомнительный вариант — взвесьте минусы.",
        "poor": "Окончательный вердикт: слабая сделка по цене и расположению.",
    }[verdict]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
