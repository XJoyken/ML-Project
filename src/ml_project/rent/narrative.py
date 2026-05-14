from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..narrative import (
    DEFAULT_GEMINI_MODEL,
    FallbackReason,
    Language,
    NarrativeSource,
    _classify_llm_error,
    _fallback_reason_message,
    _strip_proper_noun_quotes,
)
from .investment import InvestmentResult, InvestmentVerdict


class InvestmentNarrative(BaseModel):
    language: Language = Field(description="Output language.")
    lead: str = Field(description="One sentence: deal verdict + headline number (e.g. payback years).")
    price_paragraph: str = Field(
        description="How the listed sale price compares to the predicted fair price; "
        "the price interval; what a buyer should make of it."
    )
    investment_paragraph: str = Field(
        description="Concrete investment numbers: predicted monthly rent, gross/net yield, "
        "payback period, NPV at horizon, IRR."
    )
    market_context_paragraph: str = Field(
        description="Comparison to the latest Almaty market yield, inflation outlook, "
        "rent growth trend. Mention concrete percentages."
    )
    location_paragraph: str = Field(
        description="Rental-market value of the location: universities/metro pull students, "
        "schools/parks pull families, etc. Cite nearest POI distances. "
        "Comment on PM2.5 air quality and crime tier."
    )
    risks: list[str] = Field(
        description="Concrete risks (vacancy assumption, repair cost, rent shock, etc.)."
    )
    verdict: InvestmentVerdict
    verdict_text: str = Field(description="One sentence summarising the final verdict.")
    source: NarrativeSource = Field(default="fallback")
    fallback_reason_code: FallbackReason | None = Field(default=None)
    fallback_reason: str | None = Field(default=None)


def generate(
    *,
    investment: dict[str, Any],
    sale_evaluation: dict[str, Any],
    rent_evaluation: dict[str, Any],
    market_summary: dict[str, Any],
    language: Language = "ru",
    use_llm: bool | None = None,
    llm_client: Any | None = None,
    llm_model: str | None = None,
) -> InvestmentNarrative:
    if use_llm is None:
        use_llm = bool(os.getenv("GEMINI_API_KEY"))

    if use_llm:
        try:
            report = _generate_with_gemini(
                investment=investment,
                sale_evaluation=sale_evaluation,
                rent_evaluation=rent_evaluation,
                market_summary=market_summary,
                language=language,
                client=llm_client,
                model=llm_model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            )
            report.source = "gemini"
            _strip_quotes_in_place(report)
            return report
        except Exception as exc:
            print(
                f"[investment] Gemini call failed, falling back to deterministic: {exc!r}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            reason_code = _classify_llm_error(exc)
            report = _deterministic_narrative(
                investment=investment,
                sale_evaluation=sale_evaluation,
                rent_evaluation=rent_evaluation,
                market_summary=market_summary,
                language=language,
            )
            report.fallback_reason_code = reason_code
            report.fallback_reason = _fallback_reason_message(reason_code, language)
            return report

    return _deterministic_narrative(
        investment=investment,
        sale_evaluation=sale_evaluation,
        rent_evaluation=rent_evaluation,
        market_summary=market_summary,
        language=language,
    )


def _generate_with_gemini(
    *,
    investment: dict[str, Any],
    sale_evaluation: dict[str, Any],
    rent_evaluation: dict[str, Any],
    market_summary: dict[str, Any],
    language: Language,
    client: Any | None,
    model: str,
) -> InvestmentNarrative:
    from google import genai
    from google.genai import types

    if client is None:
        client = genai.Client()

    payload = {
        "investment": investment,
        "sale": {
            "predicted_price_kzt": sale_evaluation.get("predicted_price_kzt"),
            "listing_price_kzt": sale_evaluation.get("listing_price_kzt"),
            "delta_percent": sale_evaluation.get("delta_percent"),
            "price_interval": sale_evaluation.get("price_interval"),
            "final_verdict": sale_evaluation.get("final_verdict"),
        },
        "rent": {
            "predicted_rent_kzt": rent_evaluation.get("predicted_rent_kzt"),
            "rent_interval": rent_evaluation.get("rent_interval"),
            "crime": rent_evaluation.get("crime"),
            "nearby_pois": rent_evaluation.get("nearby_pois"),
            "nearest_air_sensor": rent_evaluation.get("nearest_air_sensor"),
        },
        "market": market_summary,
    }
    user_payload = json.dumps(payload, ensure_ascii=False)
    response = client.models.generate_content(
        model=model,
        contents=user_payload,
        config=types.GenerateContentConfig(
            system_instruction=_system_prompt(language),
            temperature=0.2,
            response_mime_type="application/json",
            response_json_schema=InvestmentNarrative.model_json_schema(),
        ),
    )
    return InvestmentNarrative.model_validate_json(response.text)


def _system_prompt(language: Language) -> str:
    language_instruction = (
        "Write all text fields in Russian." if language == "ru" else "Write all text fields in English."
    )
    return f"""
You write an investment analysis for an Almaty apartment that someone is considering
buying and renting out. You receive a structured payload with four parts:
- investment: pre-computed economic metrics (yields, payback, NPV, IRR, verdict).
- sale: Model 1 prediction (predicted vs listed price, conformal interval, final_verdict).
- rent: Model 3 prediction (predicted monthly rent, conformal interval) plus location
  context — nearby_pois, nearest_air_sensor, district crime info.
- market: Almaty macro context — inflation, rent growth, primary/secondary market yield.

Rules:
- {language_instruction}
- Treat ALL numbers as facts from the payload. Never invent figures.
- For KZT: use exact values or millions with TWO decimals (X.XX млн ₸) — never round to nearest 10k.
  For percentages: use exact values from the payload, e.g. "8.29%", not "8%".
  For distances: use EXACT values from the payload, do NOT round.
- Never wrap proper nouns (POI names, district names, ЖК names) in quotation marks of any kind.
- Each paragraph 2-4 dense sentences. No filler, no generic praise.

Required content per paragraph:
- lead: one short sentence with the overall verdict (excellent / good / average / poor) and the
  single most important number (payback in years, or yield-vs-market gap in pp).

- price_paragraph: cite the EXACT predicted price, listed price, gap as %, and the conformal
  interval bounds (low_kzt and high_kzt). State whether the listing is undervalued, fair, or overpriced.

- investment_paragraph: THIS IS THE MOST IMPORTANT PARAGRAPH. You MUST include:
  1. Predicted monthly rent (exact figure, e.g. "330 952 ₸/мес").
  2. Gross yield and net yield (exact figures, e.g. "gross 8.29% / net 7.33%").
  3. Payback — both nominal and inflation-adjusted years.
  4. NPV at the specified horizon and IRR (exact figures).
  5. ALL assumptions used, stated explicitly and naturally in the text. Use this exact format
     for the assumption list (adapt language to RU/EN):
     "Расчёт выполнен при следующих условиях: вакантность X%, затраты на ремонт Y% от цены покупки,
     комиссия агента Z мес/арендатора при смене раз в N лет, обслуживание M% от аренды в год,
     налог на недвижимость T% в год, ставка дисконтирования R% (инфляция + риск-премия), горизонт H лет."
     Replace every letter with the EXACT value from investment.params.
  6. Do not say "default" or "non-default" — always state the number.

- market_context_paragraph: cite market.rental_yield_annual_pct, market.rent_growth_annual_pct,
  market.secondary_growth_annual_pct, inflation.latest_annual_pct (exact values). Say whether
  this listing beats or trails market yield and by exactly how many pp.

- location_paragraph: pick 4-6 relevant nearby_pois with EXACT distances and POI names.
  Frame through the RENTAL lens: who is the likely tenant for this location?
    - universities/metro near → students, young professionals
    - schools/kindergartens/parks → families with children
    - bus_stops/clinics → broad demographics
  For bus_stops: list routes in parentheses. Codes starting with "Тр" mean trolleybus
  (e.g. "Тр6" = троллейбус №6). Format: "остановка Имя в X м (автобусы A, B; троллейбус ТрC)".
  Cite exact PM2.5 values: cold day and warm day. Tier: ≤15 отлично, ≤25 хорошо, ≤40 умеренно,
  ≤60 плохо, >60 очень плохо. Mention crime tier with exact count and district name.

- risks: 2-4 concrete bullet points that reference the ACTUAL numbers already used:
  e.g. "если реальная вакантность окажется 15% вместо заложенных 8%, net yield упадёт с 7.33% до ~6.5%",
  "ставка дисконтирования 15.3% чувствительна к инфляции — рост до 18% снизит NPV на ~X млн ₸".
  Make risks feel like they belong to THIS specific apartment, not generic advice.

- verdict + verdict_text: copy verdict from `investment.verdict`, write ONE sentence with the final call,
  then add ONE supporting sentence that explains WHY the verdict is what it is given the Kazakhstan
  macro context. Use these calibration thresholds explicitly so the reader understands the rule:
    excellent — inflation-adjusted payback < 8 yr · good — < 12 yr · average — < 16 yr · poor — ≥ 16 yr.
  When the verdict is "poor" or "average" because of the discount rate, say so plainly. Example (RU):
    "Инвестиция оценена как «плохо»: при инфляции 12.3% и риск-премии 3 пп ставка дисконтирования
     15.3% — реальная окупаемость растягивается до 20+ лет. Чтобы попасть в «хорошо», нужна валовая
     доходность 10%+ или снижение цены покупки на ~15%."
  If the user lowers risk_premium toward zero, the verdict can improve — say so when honest.

Do not output JSON outside the schema fields. Do not use markdown formatting characters (no **, no ##).
""".strip()


def _deterministic_narrative(
    *,
    investment: dict[str, Any],
    sale_evaluation: dict[str, Any],
    rent_evaluation: dict[str, Any],
    market_summary: dict[str, Any],
    language: Language,
) -> InvestmentNarrative:
    verdict: InvestmentVerdict = investment.get("verdict", "average")
    lead = _det_lead(investment, language)
    return InvestmentNarrative(
        language=language,
        lead=lead,
        price_paragraph=_det_price(sale_evaluation, language),
        investment_paragraph=_det_investment(investment, language),
        market_context_paragraph=_det_market(investment, market_summary, language),
        location_paragraph=_det_location(rent_evaluation, language),
        risks=_det_risks(investment, language),
        verdict=verdict,
        verdict_text=_det_verdict_text(verdict, language),
        source="fallback",
    )


def _det_lead(investment: dict[str, Any], language: Language) -> str:
    payback = investment.get("payback_years_inflation_adjusted")
    verdict = investment.get("verdict", "average")
    if language == "en":
        verdict_word = {"excellent": "excellent", "good": "good", "average": "average", "poor": "weak"}[verdict]
        if payback is None:
            return f"Investment verdict: {verdict_word}. The cashflow does not pay back the investment in a reasonable horizon."
        return f"Investment verdict: {verdict_word}. Inflation-adjusted payback ≈ {payback:.1f} years."
    verdict_word = {
        "excellent": "отличная",
        "good": "хорошая",
        "average": "средняя",
        "poor": "слабая",
    }[verdict]
    if payback is None:
        return f"Инвестиционная оценка: {verdict_word}. Денежный поток не окупает вложения в обозримом горизонте."
    return f"Инвестиционная оценка: {verdict_word}. Окупаемость с учётом инфляции ≈ {payback:.1f} лет."


def _det_price(sale_evaluation: dict[str, Any], language: Language) -> str:
    pred = sale_evaluation.get("predicted_price_kzt")
    listed = sale_evaluation.get("listing_price_kzt")
    delta = sale_evaluation.get("delta_percent")
    interval = sale_evaluation.get("price_interval") or {}
    low = interval.get("low_kzt")
    high = interval.get("high_kzt")
    if pred is None or listed is None:
        return ""
    if language == "en":
        parts = [
            f"Listed at {int(listed):,} KZT ({listed / 1e6:.2f}M) vs model estimate "
            f"{int(pred):,} KZT ({pred / 1e6:.2f}M) — gap {(delta or 0):+.1f}%."
        ]
        if low is not None and high is not None:
            parts.append(
                f"90% confidence interval: {int(low):,}–{int(high):,} KZT "
                f"({low / 1e6:.2f}M–{high / 1e6:.2f}M)."
            )
        return " ".join(parts)
    parts = [
        f"Цена объявления {int(listed):,} ₸ ({listed / 1e6:.2f} млн ₸), "
        f"оценка модели {int(pred):,} ₸ ({pred / 1e6:.2f} млн ₸), "
        f"отклонение {(delta or 0):+.1f}%."
    ]
    if low is not None and high is not None:
        parts.append(
            f"Доверительный интервал 90%: {int(low):,}–{int(high):,} ₸ "
            f"({low / 1e6:.2f}–{high / 1e6:.2f} млн ₸)."
        )
    return " ".join(parts)


def _det_investment(investment: dict[str, Any], language: Language) -> str:
    monthly = investment.get("monthly_rent_kzt", 0)
    gross = investment.get("gross_yield_pct", 0)
    net = investment.get("net_yield_pct", 0)
    payback_nom = investment.get("payback_years_nominal")
    payback_real = investment.get("payback_years_inflation_adjusted")
    npv = investment.get("npv_kzt", 0)
    irr = investment.get("irr_pct")
    p = investment.get("params") or {}
    horizon = p.get("horizon_years", 10)
    vacancy = p.get("vacancy_rate", 0.08)
    repair = p.get("repair_cost_pct", 0.05)
    agent_months = p.get("agent_commission_months", 0.5)
    turnover = p.get("tenant_turnover_years", 1.5)
    maint = p.get("maintenance_pct", 0.05)
    tax = p.get("property_tax_pct", 0.003)
    inflation = p.get("inflation_rate_pct", 12.3)
    risk_prem = p.get("risk_premium_pct", 3.0)
    discount = p.get("discount_rate_pct", inflation + risk_prem)

    if language == "en":
        parts = [
            f"Predicted monthly rent: {int(monthly):,} KZT.",
            f"Gross yield {gross:.2f}% / net yield {net:.2f}%.",
        ]
        if payback_nom is not None:
            parts.append(f"Nominal payback {payback_nom:.1f} yr.")
        if payback_real is not None:
            parts.append(f"Inflation-adjusted payback {payback_real:.1f} yr.")
        parts.append(f"NPV over {horizon} years: {npv / 1e6:+.2f}M KZT.")
        if irr is not None:
            parts.append(f"IRR {irr:.2f}%.")
        parts.append(
            f"Assumptions: vacancy {vacancy * 100:.0f}%, repair {repair * 100:.0f}% of price, "
            f"agent commission {agent_months} months per {turnover:.1f}-year tenancy, "
            f"maintenance {maint * 100:.0f}% of rent/yr, property tax {tax * 100:.1f}% of price/yr, "
            f"discount rate {discount:.1f}% (inflation {inflation:.1f}% + risk premium {risk_prem:.1f}%), "
            f"horizon {horizon} years."
        )
        return " ".join(parts)

    parts = [
        f"Прогноз месячной аренды: {int(monthly):,} ₸.",
        f"Gross yield {gross:.2f}% / net yield {net:.2f}%.",
    ]
    if payback_nom is not None:
        parts.append(f"Номинальная окупаемость {payback_nom:.1f} лет.")
    if payback_real is not None:
        parts.append(f"Окупаемость с учётом инфляции {payback_real:.1f} лет.")
    parts.append(f"NPV за {horizon} лет: {npv / 1e6:+.2f} млн ₸.")
    if irr is not None:
        parts.append(f"IRR {irr:.2f}%.")
    parts.append(
        f"Расчёт выполнен при следующих условиях: вакантность {vacancy * 100:.0f}%, "
        f"ремонт {repair * 100:.0f}% от цены покупки, "
        f"комиссия риелтора {agent_months} мес/арендатора при смене раз в {turnover:.1f} лет, "
        f"обслуживание {maint * 100:.0f}% от аренды в год, "
        f"налог на недвижимость {tax * 100:.1f}% в год, "
        f"ставка дисконтирования {discount:.1f}% "
        f"(инфляция {inflation:.1f}% + риск-премия {risk_prem:.1f}%), "
        f"горизонт {horizon} лет."
    )
    return " ".join(parts)


def _det_market(
    investment: dict[str, Any],
    market_summary: dict[str, Any],
    language: Language,
) -> str:
    market = market_summary.get("market") or {}
    inflation = market_summary.get("inflation") or {}
    market_yield = market.get("rental_yield_annual_pct")
    rent_growth = market.get("rent_growth_annual_pct")
    secondary_growth = market.get("secondary_growth_annual_pct")
    cpi = inflation.get("latest_annual_pct")
    diff = investment.get("yield_vs_market_pct_points")
    if language == "en":
        parts = []
        if market_yield is not None:
            parts.append(f"Almaty market yield ≈ {market_yield:.1f}%.")
        if diff is not None:
            sign = "above" if diff >= 0 else "below"
            parts.append(f"This listing is {abs(diff):.1f} pp {sign} market.")
        if rent_growth is not None:
            parts.append(f"Rent has grown ≈ {rent_growth:.1f}% per year.")
        if secondary_growth is not None:
            parts.append(f"Secondary-market prices grew ≈ {secondary_growth:.1f}% per year.")
        if cpi is not None:
            parts.append(f"Latest CPI ≈ {cpi:.1f}%.")
        return " ".join(parts) if parts else ""
    parts = []
    if market_yield is not None:
        parts.append(f"Средний yield по Алматы ≈ {market_yield:.1f}%.")
    if diff is not None:
        sign = "выше" if diff >= 0 else "ниже"
        parts.append(f"Это объявление — на {abs(diff):.1f} п.п. {sign} рынка.")
    if rent_growth is not None:
        parts.append(f"Аренда растёт ≈ {rent_growth:.1f}% в год.")
    if secondary_growth is not None:
        parts.append(f"Вторичный рынок рос ≈ {secondary_growth:.1f}% в год.")
    if cpi is not None:
        parts.append(f"Последняя инфляция ≈ {cpi:.1f}%.")
    return " ".join(parts) if parts else ""


def _det_location(rent_evaluation: dict[str, Any], language: Language) -> str:
    nearby = rent_evaluation.get("nearby_pois") or []
    air = rent_evaluation.get("nearest_air_sensor")
    crime = rent_evaluation.get("crime")
    fragments: list[str] = []
    for poi in nearby[:4]:
        distance = float(poi.get("distance_m", 0))
        cat = str(poi.get("category", ""))
        name = poi.get("name") or ""
        unit = "м" if language == "ru" else "m"
        dist_text = f"{distance:.0f} {unit}" if distance < 1000 else (
            f"{distance / 1000:.1f} км" if language == "ru" else f"{distance / 1000:.1f} km"
        )
        fragments.append(f"{cat} {name} ({dist_text})".strip())
    if air is not None:
        cold = air.get("pm25_cold_day")
        warm = air.get("pm25_warm_day")
        if cold is not None and warm is not None:
            if language == "en":
                fragments.append(f"PM2.5 winter {cold:.0f} / summer {warm:.0f} µg/m³")
            else:
                fragments.append(f"PM2.5 зимой {cold:.0f} / летом {warm:.0f} мкг/м³")
    if crime is not None:
        if language == "en":
            fragments.append(
                f"{crime.get('district', '')}: {int(crime.get('count', 0))} crimes in {crime.get('year')}"
            )
        else:
            fragments.append(
                f"{crime.get('district', '')}: {int(crime.get('count', 0))} преступлений за {crime.get('year')}"
            )
    if not fragments:
        return ""
    head = "Nearby: " if language == "en" else "Рядом: "
    return head + "; ".join(fragments) + "."


def _det_risks(investment: dict[str, Any], language: Language) -> list[str]:
    params = investment.get("params") or {}
    vacancy = params.get("vacancy_rate", 0.08)
    discount = params.get("discount_rate_pct", 15)
    if language == "en":
        return [
            f"Vacancy assumption: {vacancy * 100:.0f}%. Higher real-world vacancy directly cuts net yield.",
            f"Discount rate used: {discount:.1f}%. A higher rate shrinks NPV; a lower rate boosts it.",
            "Repair cost is assumed at 5% of the sale price — a major renovation may double this.",
            "Rent forecasts rely on stable demand; oversupply or economic shocks can compress them.",
        ]
    return [
        f"Вакантность заложена {vacancy * 100:.0f}%. Реальный простой выше — net yield падает прямо пропорционально.",
        f"Ставка дисконтирования {discount:.1f}%. Выше — NPV сжимается, ниже — растёт.",
        "Ремонт заложен в 5% от цены. Капитальный ремонт может удвоить эту цифру.",
        "Прогноз аренды зависит от спроса — переизбыток предложения или экономический шок сожмут доход.",
    ]


def _det_verdict_text(verdict: InvestmentVerdict, language: Language) -> str:
    if language == "en":
        return {
            "excellent": "Final call: an excellent rental investment.",
            "good": "Final call: a solid rental investment.",
            "average": "Final call: an average rental investment — weigh the risks.",
            "poor": "Final call: a weak rental investment at this price.",
        }[verdict]
    return {
        "excellent": "Окончательный вывод: отличная инвестиция в аренду.",
        "good": "Окончательный вывод: хорошая инвестиция в аренду.",
        "average": "Окончательный вывод: средняя инвестиция — взвесьте риски.",
        "poor": "Окончательный вывод: слабая инвестиция при такой цене.",
    }[verdict]


def _strip_quotes_in_place(report: InvestmentNarrative) -> None:
    report.lead = _strip_proper_noun_quotes(report.lead)
    report.price_paragraph = _strip_proper_noun_quotes(report.price_paragraph)
    report.investment_paragraph = _strip_proper_noun_quotes(report.investment_paragraph)
    report.market_context_paragraph = _strip_proper_noun_quotes(report.market_context_paragraph)
    report.location_paragraph = _strip_proper_noun_quotes(report.location_paragraph)
    report.risks = [_strip_proper_noun_quotes(r) for r in report.risks]
    report.verdict_text = _strip_proper_noun_quotes(report.verdict_text)


__all__ = [
    "InvestmentNarrative",
    "generate",
]
