from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DistanceTierName = Literal["excellent", "good", "acceptable", "far"]
AvoidTierName = Literal["too_close", "concerning", "safe"]
AirTierName = Literal["excellent", "good", "moderate", "poor", "very_poor"]


@dataclass(frozen=True, slots=True)
class DistanceTiers:
    excellent_m: float
    good_m: float
    acceptable_m: float

    def classify(self, distance_m: float) -> DistanceTierName:
        if distance_m <= self.excellent_m:
            return "excellent"
        if distance_m <= self.good_m:
            return "good"
        if distance_m <= self.acceptable_m:
            return "acceptable"
        return "far"


@dataclass(frozen=True, slots=True)
class AvoidDistanceTiers:
    too_close_m: float
    concerning_m: float

    def classify(self, distance_m: float) -> AvoidTierName:
        if distance_m <= self.too_close_m:
            return "too_close"
        if distance_m <= self.concerning_m:
            return "concerning"
        return "safe"


PROXIMITY_TIERS: dict[str, DistanceTiers] = {
    "schools": DistanceTiers(excellent_m=500, good_m=1000, acceptable_m=2000),
    "kindergartens": DistanceTiers(excellent_m=400, good_m=800, acceptable_m=1500),
    "universities": DistanceTiers(excellent_m=1000, good_m=2000, acceptable_m=4000),
    "metro": DistanceTiers(excellent_m=500, good_m=1000, acceptable_m=2000),
    "bus_stops": DistanceTiers(excellent_m=200, good_m=500, acceptable_m=1000),
    "transport": DistanceTiers(excellent_m=2000, good_m=4000, acceptable_m=8000),
    "clinics": DistanceTiers(excellent_m=700, good_m=1500, acceptable_m=3000),
    "polyclinics": DistanceTiers(excellent_m=1000, good_m=2000, acceptable_m=4000),
    "hospitals": DistanceTiers(excellent_m=1500, good_m=3000, acceptable_m=5000),
    "dentistry": DistanceTiers(excellent_m=700, good_m=1500, acceptable_m=3000),
    "medcenters": DistanceTiers(excellent_m=1000, good_m=2000, acceptable_m=4000),
    "restaurants_coffee": DistanceTiers(excellent_m=400, good_m=800, acceptable_m=1500),
    "parks": DistanceTiers(excellent_m=500, good_m=1000, acceptable_m=2000),
    "supermarkets": DistanceTiers(excellent_m=300, good_m=700, acceptable_m=1500),
    "fitness": DistanceTiers(excellent_m=500, good_m=1000, acceptable_m=2000),
}

AVOID_TIERS: dict[str, AvoidDistanceTiers] = {
    "energy": AvoidDistanceTiers(too_close_m=500, concerning_m=1000),
}

POI_PERSONA_HINTS = {
    "schools": ("families_with_kids",),
    "kindergartens": ("families_with_kids",),
    "universities": ("students",),
    "metro": ("commuters", "students"),
    "bus_stops": ("commuters", "elderly", "students"),
    "transport": ("commuters",),
    "clinics": ("elderly", "families_with_kids"),
    "polyclinics": ("elderly",),
    "hospitals": ("elderly",),
    "dentistry": ("everyone",),
    "medcenters": ("everyone",),
    "restaurants_coffee": ("singles", "young_professionals"),
    "energy": ("everyone",),
    "parks": ("families_with_kids", "elderly"),
    "supermarkets": ("everyone",),
    "fitness": ("young_professionals", "singles"),
}

AIR_PM25_TIERS = {
    "excellent": 15.0,
    "good": 25.0,
    "moderate": 40.0,
    "poor": 60.0,
}


def classify_pm25(pm25: float | None) -> AirTierName | None:
    if pm25 is None:
        return None
    if pm25 <= AIR_PM25_TIERS["excellent"]:
        return "excellent"
    if pm25 <= AIR_PM25_TIERS["good"]:
        return "good"
    if pm25 <= AIR_PM25_TIERS["moderate"]:
        return "moderate"
    if pm25 <= AIR_PM25_TIERS["poor"]:
        return "poor"
    return "very_poor"


CrimeTierName = Literal["low", "medium", "high"]


def classify_crime_index(crime_index: float | None) -> CrimeTierName | None:
    """crime_index is min-max normalised crime count per district in [0, 1]."""
    if crime_index is None:
        return None
    if crime_index <= 0.33:
        return "low"
    if crime_index <= 0.66:
        return "medium"
    return "high"


def score_crime_tier(tier: CrimeTierName | None) -> float | None:
    if tier is None:
        return None
    return {"low": 1.0, "medium": 0.55, "high": 0.15}[tier]


CRIME_TIER_LABELS_RU = {
    "low": "низкий уровень преступности",
    "medium": "средний уровень преступности",
    "high": "высокий уровень преступности",
}

CRIME_TIER_LABELS_EN = {
    "low": "low crime rate",
    "medium": "medium crime rate",
    "high": "high crime rate",
}


def crime_tier_label(tier: CrimeTierName, language: str) -> str:
    if language == "en":
        return CRIME_TIER_LABELS_EN[tier]
    return CRIME_TIER_LABELS_RU[tier]


PRICE_VERDICT_THRESHOLDS = {
    "great_deal": -0.10,
    "good_deal": -0.03,
    "fair": 0.03,
    "overpriced": 0.10,
}


def classify_price_delta(delta_fraction: float) -> str:
    if delta_fraction <= PRICE_VERDICT_THRESHOLDS["great_deal"]:
        return "great_deal"
    if delta_fraction <= PRICE_VERDICT_THRESHOLDS["good_deal"]:
        return "good_deal"
    if delta_fraction <= PRICE_VERDICT_THRESHOLDS["fair"]:
        return "fair"
    if delta_fraction <= PRICE_VERDICT_THRESHOLDS["overpriced"]:
        return "slightly_overpriced"
    return "overpriced"


FINAL_VERDICT_RULES = {
    "excellent": {"min_price_score": 0.7, "min_neighborhood_score": 0.7},
    "good": {"min_price_score": 0.5, "min_neighborhood_score": 0.55},
    "questionable": {"min_price_score": 0.3, "min_neighborhood_score": 0.0},
}


def score_distance_tier(tier: DistanceTierName) -> float:
    return {"excellent": 1.0, "good": 0.7, "acceptable": 0.4, "far": 0.1}[tier]


def score_avoid_tier(tier: AvoidTierName) -> float:
    return {"too_close": 0.0, "concerning": 0.4, "safe": 1.0}[tier]


def score_air_tier(tier: AirTierName | None) -> float | None:
    if tier is None:
        return None
    return {"excellent": 1.0, "good": 0.8, "moderate": 0.55, "poor": 0.3, "very_poor": 0.1}[tier]


def score_price_delta(delta_fraction: float) -> float:
    if delta_fraction <= -0.10:
        return 1.0
    if delta_fraction <= -0.03:
        return 0.85
    if delta_fraction <= 0.03:
        return 0.6
    if delta_fraction <= 0.10:
        return 0.35
    return 0.1


CATEGORY_LABELS_RU = {
    "schools": "школа",
    "kindergartens": "детский сад",
    "universities": "университет",
    "metro": "метро",
    "bus_stops": "автобусная остановка",
    "transport": "транспортный узел",
    "clinics": "клиника",
    "polyclinics": "поликлиника",
    "hospitals": "больница",
    "dentistry": "стоматология",
    "medcenters": "медцентр",
    "restaurants_coffee": "кафе/ресторан",
    "energy": "энергообъект",
    "parks": "парк",
    "supermarkets": "супермаркет",
    "fitness": "фитнес-клуб",
}

CATEGORY_LABELS_EN = {
    "schools": "school",
    "kindergartens": "kindergarten",
    "universities": "university",
    "metro": "metro station",
    "bus_stops": "bus stop",
    "transport": "transport hub (airport / railway / bus terminal)",
    "clinics": "clinic",
    "polyclinics": "polyclinic",
    "hospitals": "hospital",
    "dentistry": "dental clinic",
    "medcenters": "medical center",
    "restaurants_coffee": "café/restaurant",
    "energy": "power facility",
    "parks": "park",
    "supermarkets": "supermarket",
    "fitness": "fitness club",
}


def category_label(category: str, language: str) -> str:
    if language == "en":
        return CATEGORY_LABELS_EN.get(category, category)
    return CATEGORY_LABELS_RU.get(category, category)
