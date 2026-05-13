from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import backend.main as backend_main
from backend.main import (
    app,
    get_apartment_evaluator,
    get_rent_evaluator,
)


class StubSaleEvaluator:
    def evaluate(self, listing: dict[str, Any], *, language: str = "ru", use_llm: bool | None = None):
        return {
            "model": "lightgbm",
            "model_dir": "stub",
            "predicted_price_kzt": 50_000_000.0,
            "listing_price_kzt": float(listing.get("listing_price_kzt", 0)),
            "delta_percent": 4.0,
            "price_interval": {"coverage": 0.9, "low_kzt": 46_000_000.0, "high_kzt": 54_000_000.0},
            "verdict": "fair",
            "final_verdict": "good",
            "narrative": {},
            "nearby_pois": [],
            "nearest_air_sensor": None,
            "crime": None,
        }


class StubRentEvaluator:
    def evaluate(self, listing: dict[str, Any]):
        return {
            "model": "lightgbm",
            "model_dir": "stub",
            "predicted_rent_kzt": 280_000.0,
            "rent_interval": {"coverage": 0.9, "low_kzt": 260_000.0, "high_kzt": 300_000.0},
            "nearby_pois": [
                {"category": "metro", "name": "Алатау", "distance_m": 600},
                {"category": "schools", "name": "Школа №14", "distance_m": 300},
            ],
            "nearest_air_sensor": {
                "distance_m": 800,
                "pm25_cold_day": 35.0,
                "pm25_warm_day": 18.0,
            },
            "crime": {
                "district": "Бостандыкский район",
                "year": 2025,
                "count": 2909.0,
                "index_min_max": 0.616,
            },
        }


@pytest.fixture
def client(monkeypatch):
    def fake_parse_krisha(url: str) -> dict[str, Any]:
        return {
            "listing_id": "test-invest",
            "listing_price_kzt": 48_000_000,
            "area_m2": 55,
            "rooms": 2,
            "lat": 43.24,
            "lon": 76.92,
            "district": "Бостандыкский район",
        }

    monkeypatch.setattr(backend_main, "parse_krisha_listing_url", fake_parse_krisha)
    app.dependency_overrides[get_apartment_evaluator] = lambda: StubSaleEvaluator()
    app.dependency_overrides[get_rent_evaluator] = lambda: StubRentEvaluator()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_investment_endpoint_returns_metrics_and_narrative(client):
    response = client.post(
        "/apartments/investment",
        json={"url": "https://krisha.kz/a/show/999", "language": "ru", "use_llm": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_url"] == "https://krisha.kz/a/show/999"
    assert body["sale_evaluation"]["listing_price_kzt"] == 48_000_000
    assert body["rent_evaluation"]["predicted_rent_kzt"] == 280_000
    inv = body["investment"]
    assert "gross_yield_pct" in inv
    assert "verdict" in inv
    assert inv["verdict"] in {"excellent", "good", "average", "poor"}
    assert body["narrative"]["verdict"] == inv["verdict"]
    assert "params" in inv
    # Default fallback (use_llm=False) → source must be fallback
    assert body["narrative"]["source"] == "fallback"


def test_investment_endpoint_respects_override_params(client):
    response = client.post(
        "/apartments/investment",
        json={
            "url": "https://krisha.kz/a/show/999",
            "language": "ru",
            "use_llm": False,
            "investment_params": {
                "vacancy_rate": 0.0,
                "repair_cost_pct": 0.0,
                "maintenance_pct": 0.0,
                "property_tax_pct": 0.0,
                "agent_commission_months": 0.0,
                "horizon_years": 5,
            },
        },
    )
    assert response.status_code == 200
    inv = response.json()["investment"]
    params = inv["params"]
    assert params["vacancy_rate"] == 0.0
    assert params["horizon_years"] == 5
    # With zero cost overrides, annual recurring costs must be zero
    assert inv["annual_recurring_costs_kzt"] == 0.0


def test_investment_endpoint_validates_short_url(client):
    response = client.post("/apartments/investment", json={"url": "x"})
    assert response.status_code == 422
