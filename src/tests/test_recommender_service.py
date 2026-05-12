from __future__ import annotations

from dataclasses import dataclass

import pytest

from ml_project.recommender.narrative import generate as generate_narrative
from ml_project.recommender.service import RecommendationService


@dataclass
class FakeNumericTarget:
    feature: str
    preference: str
    value: float
    weight: float
    evidence: str = ""


@dataclass
class FakeCategoricalTarget:
    feature: str
    value: str
    weight: float
    evidence: str = ""


@dataclass
class FakeBooleanTarget:
    feature: str
    value: bool
    weight: float
    evidence: str = ""


@dataclass
class FakeFeatures:
    intent: str = "buy"
    numeric_targets: list[FakeNumericTarget] = None
    categorical_targets: list[FakeCategoricalTarget] = None
    boolean_targets: list[FakeBooleanTarget] = None
    location_preferences: list = None

    def __post_init__(self):
        if self.numeric_targets is None:
            self.numeric_targets = []
        if self.categorical_targets is None:
            self.categorical_targets = []
        if self.boolean_targets is None:
            self.boolean_targets = []
        if self.location_preferences is None:
            self.location_preferences = []


@pytest.fixture(scope="module")
def service():
    return RecommendationService()


def test_recommend_respects_district_hard_filter(service):
    features = FakeFeatures(
        intent="buy",
        categorical_targets=[
            FakeCategoricalTarget(feature="district", value="Бостандыкский район", weight=1.0),
        ],
        numeric_targets=[
            FakeNumericTarget(
                feature="target_price_kzt",
                preference="at_most",
                value=80_000_000.0,
                weight=0.8,
            ),
        ],
    )
    result = service.recommend(features, limit=5)
    assert len(result["items"]) == 5
    for item in result["items"]:
        assert item["district"] == "Бостандыкский район"


def test_recommend_rooms_equal_hard_filter(service):
    features = FakeFeatures(
        intent="buy",
        numeric_targets=[
            FakeNumericTarget(feature="rooms", preference="equal", value=2.0, weight=1.0),
            FakeNumericTarget(feature="target_price_kzt", preference="at_most", value=60_000_000.0, weight=0.5),
        ],
    )
    result = service.recommend(features, limit=5)
    assert len(result["items"]) == 5
    for item in result["items"]:
        assert round(float(item["rooms"])) == 2


def test_recommend_at_most_price_soft_preference_keeps_below_target(service):
    features = FakeFeatures(
        intent="buy",
        numeric_targets=[
            FakeNumericTarget(feature="target_price_kzt", preference="at_most", value=30_000_000.0, weight=0.9),
            FakeNumericTarget(feature="rooms", preference="equal", value=2.0, weight=0.6),
        ],
    )
    result = service.recommend(features, limit=5)
    prices = [item["price_kzt"] for item in result["items"]]
    # Hard filter has 10% slack
    assert max(prices) <= 30_000_000.0 * 1.1


def test_recommend_mmr_diversity_across_districts(service):
    # No district preference → MMR should pick from different districts where possible
    features = FakeFeatures(
        intent="buy",
        numeric_targets=[
            FakeNumericTarget(feature="schools_nearest_dist_m", preference="at_most", value=500.0, weight=0.7),
            FakeNumericTarget(feature="target_price_kzt", preference="at_most", value=80_000_000.0, weight=0.5),
        ],
    )
    result = service.recommend(features, limit=5, mmr_lambda=0.3)  # lower lambda → more diversity
    districts = {item["district"] for item in result["items"] if item.get("district")}
    assert len(districts) >= 2


def test_recommend_empty_plan_raises(service):
    features = FakeFeatures()
    with pytest.raises(ValueError):
        service.recommend(features, limit=5)


def test_recommender_narrative_fallback_produces_text(service):
    features = FakeFeatures(
        intent="buy",
        numeric_targets=[
            FakeNumericTarget(feature="schools_nearest_dist_m", preference="at_most", value=500.0, weight=0.9),
            FakeNumericTarget(feature="target_price_kzt", preference="at_most", value=80_000_000.0, weight=0.7),
        ],
    )
    result = service.recommend(features, limit=3)
    narrative = generate_narrative(
        plan_payload=result["plan"],
        items=result["items"],
        language="ru",
        use_llm=False,
    )
    assert narrative.source == "fallback"
    assert narrative.summary
    assert len(narrative.items) == 3
    assert all(item.explanation for item in narrative.items)


def test_recommender_narrative_english(service):
    features = FakeFeatures(
        intent="buy",
        numeric_targets=[
            FakeNumericTarget(feature="target_price_kzt", preference="at_most", value=50_000_000.0, weight=0.8),
        ],
    )
    result = service.recommend(features, limit=3)
    narrative = generate_narrative(
        plan_payload=result["plan"],
        items=result["items"],
        language="en",
        use_llm=False,
    )
    assert "KZT" in narrative.summary or "M" in narrative.summary
