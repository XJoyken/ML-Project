from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

NumericPreference = Literal["at_most", "at_least", "equal", "prefer_low", "prefer_high"]

HARD_WEIGHT_THRESHOLD = 0.85
HARD_AT_MOST_SLACK = 0.10
HARD_AT_LEAST_SLACK = 0.10
ROOMS_FEATURE = "rooms"
PRICE_FEATURE = "target_price_kzt"
PRICE_EQUAL_TOLERANCE = 0.10


@dataclass(slots=True)
class NumericMatch:
    feature: str
    preference: NumericPreference
    target: float
    actual: float
    weight: float
    score: float
    hard_failed: bool = False


@dataclass(slots=True)
class CategoricalMatch:
    feature: str
    target: str
    actual: str | None
    weight: float
    score: float
    hard_failed: bool = False


def score_numeric(
    *,
    actual: float,
    target: float,
    preference: NumericPreference,
    feature: str,
) -> float:
    if math.isnan(actual):
        return 0.0
    if preference == "equal":
        return _score_equal(actual, target, feature)
    if preference == "at_most":
        return _score_at_most(actual, target)
    if preference == "at_least":
        return _score_at_least(actual, target)
    if preference == "prefer_low":
        return _score_at_most(actual, target)
    if preference == "prefer_high":
        return _score_at_least(actual, target)
    return 0.0


def hard_filter_passes(
    *,
    actual: float,
    target: float,
    preference: NumericPreference,
    feature: str,
) -> bool:
    if math.isnan(actual):
        return False
    if preference == "at_most" or preference == "prefer_low":
        slack_limit = target * (1.0 + HARD_AT_MOST_SLACK) if target > 0 else target
        return actual <= slack_limit
    if preference == "at_least" or preference == "prefer_high":
        slack_limit = target * (1.0 - HARD_AT_LEAST_SLACK) if target > 0 else target
        return actual >= slack_limit
    if preference == "equal":
        if feature == ROOMS_FEATURE:
            return round(actual) == round(target)
        if feature == PRICE_FEATURE:
            return abs(actual - target) <= max(target * PRICE_EQUAL_TOLERANCE, 1.0)
        return abs(actual - target) <= max(abs(target) * 0.10, 1.0)
    return True


def categorical_score(actual: str | None, target: str) -> float:
    if actual is None:
        return 0.0
    if str(actual).strip().lower() == str(target).strip().lower():
        return 1.0
    return 0.0


def _score_equal(actual: float, target: float, feature: str) -> float:
    if feature == ROOMS_FEATURE:
        diff = abs(round(actual) - round(target))
        return max(0.0, 1.0 - diff * 0.5)  # 0 diff=1.0, 1 diff=0.5, 2 diff=0.0
    if feature == PRICE_FEATURE:
        scale = max(abs(target) * PRICE_EQUAL_TOLERANCE, 1.0)
    else:
        scale = max(abs(target) * 0.20, 1.0)
    return max(0.0, 1.0 - abs(actual - target) / scale)


def _score_at_most(actual: float, target: float) -> float:
    if target <= 0:
        return 1.0 if actual <= 0 else 0.0
    if actual <= target:
        return 1.0
    overshoot = (actual - target) / target
    return max(0.0, 1.0 - overshoot)


def _score_at_least(actual: float, target: float) -> float:
    if target <= 0:
        return 1.0
    if actual >= target:
        return 1.0
    undershoot = (target - actual) / target
    return max(0.0, 1.0 - undershoot)
