from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

NumericFeatureName = Literal[
    "target_price_kzt",
    "area_m2",
    "rooms",
    "year_built",
    "floor_current",
    "floors_total",
    "ceiling_height_m",
    "building_age",
    "area_per_room",
    "dist_to_center_km",
    "kindergartens_nearest_dist_m",
    "kindergartens_count_500m",
    "kindergartens_count_1000m",
    "schools_nearest_dist_m",
    "schools_count_500m",
    "schools_count_1000m",
    "metro_nearest_dist_m",
    "metro_count_500m",
    "metro_count_1000m",
    "transport_nearest_dist_m",
    "transport_count_500m",
    "transport_count_1000m",
    "clinics_nearest_dist_m",
    "clinics_count_500m",
    "clinics_count_1000m",
    "dentistry_nearest_dist_m",
    "dentistry_count_500m",
    "dentistry_count_1000m",
    "hospitals_nearest_dist_m",
    "hospitals_count_500m",
    "hospitals_count_1000m",
    "polyclinics_nearest_dist_m",
    "polyclinics_count_500m",
    "polyclinics_count_1000m",
    "medcenters_nearest_dist_m",
    "medcenters_count_500m",
    "medcenters_count_1000m",
    "universities_nearest_dist_m",
    "universities_count_500m",
    "universities_count_1000m",
    "restaurants_coffee_nearest_dist_m",
    "restaurants_coffee_count_500m",
    "restaurants_coffee_count_1000m",
    "energy_nearest_dist_m",
    "air_pm25_cold_day",
    "air_pm25_warm_day",
    "air_nearest_dist_m",
]

CategoricalFeatureName = Literal[
    "district",
    "house_type",
    "condition",
    "bathroom_type",
]

BooleanFeatureName = Literal[
    "has_complex_id",
    "has_photo",
    "has_microdistrict",
    "is_first_floor",
    "is_last_floor",
]

FEATURE_EXTRACTION_SYSTEM_PROMPT = """
You extract apartment recommendation features from short user prompts.

Domain:
- The product recommends apartments in Almaty.
- Return only preferences that the user explicitly wrote or strongly implied.
- Use the exact feature names from the response schema.
- Do not invent budget, rooms, district, or amenities.

Normalization rules:
- Price expressions:
  - "до X млн / до X миллионов / не дороже X млн / в пределах X млн / бюджет X млн" → target_price_kzt at_most X * 1_000_000.
  - "от X млн" → target_price_kzt at_least X * 1_000_000.
  - "от X до Y млн" → two targets: at_least X*1e6 AND at_most Y*1e6, same weight.
  - "X млн" without "до/от" → target_price_kzt at_most X * 1_000_000 (treat as cap).
  - "X тыс / X тысяч" → multiply by 1_000.
  - Always emit the price as plain digits, not strings.
- Building type and complex:
  - "ЖК", "жилой комплекс", "новостройка", "новострой", "в новостройке" → has_complex_id=true.
  - "ЖК с именем X" or "в ЖК «X»" or "Айнабулак", "Самал", "Назарбаев" (named complex) → also has_complex_id=true.
- Rooms (slang and numerals):
  - "студия", "студию" → rooms equal 1.
  - "однушка", "однушку", "1-к", "1-комнатная", "1 комн" → rooms equal 1.
  - "двушка", "двушку", "2-к", "2-комнатная", "2 комн", "две комнаты" → rooms equal 2.
  - "трешка", "трёшка", "трешку", "3-к", "3-комнатная", "3 комн", "три комнаты" → rooms equal 3.
  - "четырешка", "четырёшка", "4-к", "4-комнатная", "4 комн", "четыре комнаты" → rooms equal 4.
- Floor:
  - "не первый этаж", "не на первом" → is_first_floor=false.
  - "не последний этаж", "не на последнем" → is_last_floor=false.
- Distance to POI:
  - "рядом", "недалеко", "пешком", "в шаговой доступности" for any POI → *_nearest_dist_m at_most 500 meters.
  - "очень близко", "возле", "рядом с домом" without explicit distance → at_most 300 meters.
  - Explicit "в 500 м / в 1 км" → use that number directly.
  - "много POI поблизости" / "большой выбор" → *_count_1000m prefer_high (value 5).
- Center:
  - "в центре", "ближе к центру", "центр города" → dist_to_center_km prefer_low value 0.0.
  - "в пределах N км от центра" → dist_to_center_km at_most N.
  - Default value for "ближе к центру" without number: use 5.0 km cap (at_most preference).
- POI categories:
  - restaurants_coffee for cafés, coffee shops, restaurants, food places.
  - transport for bus stops / minibuses / public transport; metro only for explicit subway/metro.
  - parks, supermarkets, fitness — use named matches only.
- Air quality preferences → air_pm25_cold_day or air_pm25_warm_day with preference="prefer_low".
- District:
  - Named districts like "Бостандыкский", "Алмалинский" → categorical_targets feature=district value=full name.
  - "не Алмалинский" → location_preferences kind=district preference=exclude.

Weights (CRITICAL — affects hard filtering downstream):
- Use weight ≥ 0.9 for any preference that the user stated as a clear, non-negotiable requirement:
  - Specific number of rooms ("двушка", "2-комн"): weight 0.95.
  - Hard price cap ("до 40 млн", "не дороже X"): weight 0.95.
  - Explicit ЖК / new build requirement: weight 0.9.
  - "ближе к центру" or "в центре": weight 0.9 (treat as hard preference, not a soft wish).
  - Named district inclusion: weight 0.9.
  - "не первый этаж", "не последний этаж": weight 0.9.
- Use weight 0.6–0.8 for clear but flexible wishes ("желательно", "хотелось бы", "если есть").
- Use weight 0.3–0.5 only for vague nice-to-haves with hedging language.
- Default to the HIGHER end of the range when the user phrases it as a requirement.

Evidence:
- Every target must include the shortest original phrase that caused it.
- Put unsupported but explicit wishes into unmapped_preferences instead of guessing.
- Always include all arrays, even when they are empty.
""".strip()


class NumericFeatureTarget(BaseModel):
    feature: NumericFeatureName = Field(description="Canonical numeric recommender feature.")
    preference: Literal["at_most", "at_least", "equal", "prefer_low", "prefer_high"] = Field(
        description="How the recommender should compare this feature with value."
    )
    value: float = Field(description="Normalized numeric target value.")
    weight: float = Field(ge=0.0, le=1.0, description="Preference strength from 0 to 1.")
    evidence: str = Field(description="Shortest phrase from the user prompt.")


class CategoricalFeatureTarget(BaseModel):
    feature: CategoricalFeatureName = Field(description="Canonical categorical recommender feature.")
    value: str = Field(description="Normalized category value from the user prompt.")
    weight: float = Field(ge=0.0, le=1.0, description="Preference strength from 0 to 1.")
    evidence: str = Field(description="Shortest phrase from the user prompt.")


class BooleanFeatureTarget(BaseModel):
    feature: BooleanFeatureName = Field(description="Canonical boolean recommender feature.")
    value: bool = Field(description="Required boolean value.")
    weight: float = Field(ge=0.0, le=1.0, description="Preference strength from 0 to 1.")
    evidence: str = Field(description="Shortest phrase from the user prompt.")


class LocationPreference(BaseModel):
    kind: Literal["district", "microdistrict", "landmark", "city_area"] = Field(
        description="Location type named by the user."
    )
    value: str = Field(description="Location text normalized without inventing coordinates.")
    preference: Literal["include", "exclude"] = Field(description="Whether the user wants this location.")
    weight: float = Field(ge=0.0, le=1.0, description="Preference strength from 0 to 1.")
    evidence: str = Field(description="Shortest phrase from the user prompt.")


class ExtractedRecommendationFeatures(BaseModel):
    intent: Literal["buy", "rent", "unknown"] = Field(description="User intent.")
    numeric_targets: list[NumericFeatureTarget] = Field(
        description="Numeric targets for vector scoring or filtering."
    )
    categorical_targets: list[CategoricalFeatureTarget] = Field(
        description="Categorical targets for matching/filtering."
    )
    boolean_targets: list[BooleanFeatureTarget] = Field(
        description="Boolean targets for matching/filtering."
    )
    location_preferences: list[LocationPreference] = Field(
        description="Location text that may need later geocoding or raw-data filtering."
    )
    unmapped_preferences: list[str] = Field(
        description="Explicit wishes that do not map to supported features."
    )


def parse_features_json(raw_response: str) -> ExtractedRecommendationFeatures:
    return ExtractedRecommendationFeatures.model_validate_json(raw_response)


class GeminiFeatureExtractor:
    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        client: genai.Client | None = None,
    ):
        self.model = model
        self.client = client or genai.Client()

    def extract(self, user_prompt: str) -> ExtractedRecommendationFeatures:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=FEATURE_EXTRACTION_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_json_schema=ExtractedRecommendationFeatures.model_json_schema(),
            ),
        )
        return parse_features_json(response.text)
