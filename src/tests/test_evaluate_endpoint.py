from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.main import app, get_apartment_evaluator


class StubEvaluator:
    def evaluate(self, listing: dict[str, Any], *, language: str = "ru", use_llm: bool | None = None):
        return {
            "model": "lightgbm",
            "model_dir": "stub",
            "predicted_price_kzt": 42_000_000.0,
            "listing_price_kzt": float(listing["listing_price_kzt"]),
            "delta_percent": 5.0,
            "price_interval": {"coverage": 0.9, "low_kzt": 38_000_000.0, "high_kzt": 46_000_000.0},
            "verdict": "fair",
            "final_verdict": "good",
            "narrative": {
                "language": language,
                "lead": "stub lead",
                "pros": ["stub pro"],
                "cons": [],
                "persona_notes": [],
                "final_verdict": "good",
                "final_verdict_text": "stub verdict",
            },
            "nearby_pois": [],
            "nearest_air_sensor": None,
        }


@pytest.fixture
def client(monkeypatch):
    def fake_parse_krisha(url: str) -> dict[str, Any]:
        return {
            "listing_id": "test-listing",
            "listing_price_kzt": 40_000_000,
            "area_m2": 55.5,
            "rooms": 2,
            "lat": 43.235,
            "lon": 76.917,
            "district": "Бостандыкский район",
            "house_type": "монолитный",
            "condition": "свежий ремонт",
            "bathroom_type": "совмещенный",
            "year_built": 2020,
            "floor_text": "5 из 12",
            "source_url": url,
        }

    monkeypatch.setattr(backend_main, "parse_krisha_listing_url", fake_parse_krisha)
    app.dependency_overrides[get_apartment_evaluator] = lambda: StubEvaluator()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_evaluate_endpoint_returns_narrative_and_interval(client):
    response = client.post(
        "/apartments/evaluate",
        json={"url": "https://krisha.kz/a/show/123", "language": "ru"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["source_url"] == "https://krisha.kz/a/show/123"
    assert body["parsed_listing"]["listing_id"] == "test-listing"
    evaluation = body["evaluation"]
    assert evaluation["final_verdict"] in {"excellent", "good", "questionable", "poor"}
    assert evaluation["narrative"]["lead"] == "stub lead"
    assert evaluation["price_interval"]["coverage"] == 0.9


def test_evaluate_endpoint_rejects_short_url(client):
    response = client.post("/apartments/evaluate", json={"url": "x"})
    assert response.status_code == 422
