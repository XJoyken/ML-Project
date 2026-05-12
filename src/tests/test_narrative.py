from __future__ import annotations

import pytest

from ml_project.narrative import (
    _classify_llm_error,
    _fallback_reason_message,
    _strip_proper_noun_quotes,
    build_structured_evaluation,
    derive_final_verdict,
    generate,
)
from ml_project.thresholds import classify_crime_index


@pytest.fixture
def sample_pois():
    return [
        {"category": "schools", "distance_m": 320, "name": "Гимназия №14"},
        {"category": "kindergartens", "distance_m": 450, "name": "Детский сад №5"},
        {"category": "metro", "distance_m": 1800, "name": "Алатау"},
        {"category": "bus_stops", "distance_m": 210, "name": "Музей искусств",
         "routes_list": "18; 95; 121; Тр6"},
        {"category": "transport", "distance_m": 3500, "name": "Аэропорт Алматы"},
        {"category": "hospitals", "distance_m": 5500, "name": "БСНП"},
        {"category": "energy", "distance_m": 600, "name": "ТЭЦ-2"},
        {"category": "restaurants_coffee", "distance_m": 250, "name": "Coffee Boom"},
        {"category": "parks", "distance_m": 380, "name": "Парк им. Ганди"},
        {"category": "supermarkets", "distance_m": 200, "name": "Magnum"},
        {"category": "fitness", "distance_m": 600, "name": "World Class"},
    ]


@pytest.fixture
def sample_crime_low():
    return {"district": "Наурызбайский район", "year": 2025, "count": 1408, "index_min_max": 0.298}


@pytest.fixture
def sample_crime_high():
    return {"district": "Алмалинский район", "year": 2025, "count": 4721, "index_min_max": 1.0}


@pytest.fixture
def sample_air():
    return {
        "distance_m": 1200,
        "pm25_cold_day": 55.0,
        "pm25_warm_day": 18.0,
        "pm25_cold_night": 60.0,
        "pm25_warm_night": 12.0,
    }


def test_build_structured_evaluation_assigns_tiers(sample_pois, sample_air):
    structured = build_structured_evaluation(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
    )
    by_category = {signal.category: signal for signal in structured.categories}
    assert by_category["schools"].tier == "excellent"
    assert by_category["kindergartens"].tier == "good"
    assert by_category["metro"].tier == "acceptable"
    assert by_category["bus_stops"].tier == "good"
    assert by_category["bus_stops"].routes_list == "18; 95; 121; Тр6"
    assert by_category["transport"].tier == "good"
    assert by_category["hospitals"].tier == "far"
    assert by_category["energy"].tier == "concerning"
    assert by_category["energy"].is_avoid is True
    assert by_category["parks"].tier == "excellent"
    assert by_category["supermarkets"].tier == "excellent"
    assert by_category["fitness"].tier == "good"
    assert structured.air is not None
    assert structured.air.cold_tier == "poor"
    assert structured.air.warm_tier == "good"
    assert structured.price.price_verdict in {"good_deal", "great_deal"}


def test_classify_crime_index_buckets():
    assert classify_crime_index(0.1) == "low"
    assert classify_crime_index(0.5) == "medium"
    assert classify_crime_index(0.9) == "high"
    assert classify_crime_index(None) is None


def test_crime_low_adds_pro(sample_pois, sample_air, sample_crime_low):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        crime=sample_crime_low,
        language="ru",
        use_llm=False,
    )
    assert any("Наурызбайский" in pro for pro in report.pros)
    assert any("низкий" in pro for pro in report.pros)


def test_crime_high_adds_con(sample_pois, sample_air, sample_crime_high):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        crime=sample_crime_high,
        language="ru",
        use_llm=False,
    )
    assert any("Алмалинский" in con for con in report.cons)
    assert any("высокий" in con for con in report.cons)


def test_generate_handles_missing_optional_categorical_fields():
    report = generate(
        listing_price_kzt=40_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=[],
        nearest_air=None,
        crime=None,
        language="ru",
        use_llm=False,
    )
    assert report.final_verdict in {"excellent", "good", "questionable", "poor"}


def test_bus_stop_mentions_routes_in_ru(sample_pois, sample_air):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        language="ru",
        use_llm=False,
    )
    bus_text = " ".join(report.pros + report.cons)
    assert "автобусная остановка" in bus_text
    assert "автобусы 18" in bus_text
    assert "троллейбусы №6" in bus_text


def test_bus_stop_mentions_routes_in_en(sample_pois, sample_air):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        language="en",
        use_llm=False,
    )
    text = " ".join(report.pros + report.cons)
    assert "bus stop" in text
    assert "bus routes 18" in text
    assert "trolleybus routes 6" in text


class _FakeGeminiError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


def test_classify_llm_error_503_is_transient_overload():
    exc = _FakeGeminiError("503 UNAVAILABLE. This model is currently experiencing high demand.", 503)
    assert _classify_llm_error(exc) == "transient_overload"


def test_classify_llm_error_429_is_rate_limited():
    exc = _FakeGeminiError("429 RESOURCE_EXHAUSTED quota exceeded", 429)
    assert _classify_llm_error(exc) == "rate_limited"


def test_classify_llm_error_403_is_auth():
    exc = _FakeGeminiError("403 PERMISSION_DENIED invalid api key", 403)
    assert _classify_llm_error(exc) == "auth_error"


def test_classify_llm_error_string_match_when_no_code():
    exc = Exception("Model is OVERLOADED right now")
    assert _classify_llm_error(exc) == "transient_overload"


def test_strip_proper_noun_quotes_ascii():
    text = 'Convenient for families with kids, with "Школа-гимназия №51" (210 m) nearby.'
    cleaned = _strip_proper_noun_quotes(text)
    assert '"' not in cleaned
    assert "Школа-гимназия №51" in cleaned
    assert "(210 m)" in cleaned


def test_strip_proper_noun_quotes_russian_guillemets():
    text = "Рядом школа «Гимназия №14» в 320 м, а также детский сад «Мечта»."
    cleaned = _strip_proper_noun_quotes(text)
    assert "«" not in cleaned and "»" not in cleaned
    assert "Гимназия №14" in cleaned
    assert "Мечта" in cleaned


def test_strip_proper_noun_quotes_keeps_unmatched():
    text = 'There is an inch sign 5" tall — not a quote pair.'
    cleaned = _strip_proper_noun_quotes(text)
    assert cleaned == text


def test_fallback_reason_message_languages():
    ru = _fallback_reason_message("transient_overload", "ru")
    en = _fallback_reason_message("transient_overload", "en")
    assert "перегружен" in ru
    assert "overloaded" in en


def test_generate_offline_does_not_set_fallback_reason(sample_pois, sample_air):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        language="ru",
        use_llm=False,
    )
    assert report.source == "fallback"
    assert report.fallback_reason is None
    assert report.fallback_reason_code is None


def test_generate_sets_fallback_reason_when_llm_overloaded(sample_pois, sample_air):
    class StubFailingClient:
        class models:
            @staticmethod
            def generate_content(**_kwargs):
                raise _FakeGeminiError(
                    "503 UNAVAILABLE. This model is currently experiencing high demand.",
                    503,
                )

    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        language="ru",
        use_llm=True,
        llm_client=StubFailingClient(),
    )
    assert report.source == "fallback"
    assert report.fallback_reason_code == "transient_overload"
    assert "перегружен" in (report.fallback_reason or "")


def test_transport_hub_not_called_far_at_3_5km(sample_pois, sample_air):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        language="ru",
        use_llm=False,
    )
    transport_lines = [line for line in report.pros + report.cons if "транспортный узел" in line]
    assert transport_lines
    assert all("далековато" not in line for line in transport_lines)


def test_new_pois_appear_in_persona_notes(sample_pois, sample_air):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        language="ru",
        use_llm=False,
    )
    text = " ".join(report.pros + report.persona_notes)
    assert "парк" in text
    assert "супермаркет" in text
    assert "фитнес" in text


def test_derive_final_verdict_returns_known_label(sample_pois, sample_air):
    structured = build_structured_evaluation(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
    )
    verdict, score = derive_final_verdict(structured)
    assert verdict in {"excellent", "good", "questionable", "poor"}
    assert 0.0 <= score <= 1.0


def test_generate_deterministic_russian(sample_pois, sample_air):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        language="ru",
        use_llm=False,
    )
    assert report.language == "ru"
    assert report.lead
    assert any("м" in pro for pro in report.pros)
    assert any("ТЭЦ" in con or "энергообъект" in con for con in report.cons)
    assert report.final_verdict in {"excellent", "good", "questionable", "poor"}


def test_generate_deterministic_english(sample_pois, sample_air):
    report = generate(
        listing_price_kzt=38_000_000,
        predicted_price_kzt=42_000_000,
        nearby_pois=sample_pois,
        nearest_air=sample_air,
        language="en",
        use_llm=False,
    )
    assert report.language == "en"
    assert any("m" in pro for pro in report.pros)
    assert any("power" in con.lower() for con in report.cons)


def test_generate_handles_no_pois_and_no_air():
    report = generate(
        listing_price_kzt=40_000_000,
        predicted_price_kzt=40_000_000,
        nearby_pois=[],
        nearest_air=None,
        language="ru",
        use_llm=False,
    )
    assert report.final_verdict in {"excellent", "good", "questionable", "poor"}
    assert isinstance(report.pros, list)
    assert isinstance(report.cons, list)
