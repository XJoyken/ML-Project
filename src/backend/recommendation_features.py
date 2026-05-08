from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

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
- Price limits map to target_price_kzt.
- "ЖК", "жилой комплекс", "новостройка в ЖК" map to boolean has_complex_id=true.
- "не первый этаж" maps to is_first_floor=false.
- "не последний этаж" maps to is_last_floor=false.
- "рядом", "недалеко", "пешком", "в шаговой доступности" for POI features map to
  *_nearest_dist_m with preference="at_most".
- If the user says a POI should be near but gives no distance, use 500 meters.
- If the user says "очень близко", "возле", or "рядом с домом" without a distance, use 300 meters.
- If the user asks for many nearby POIs, use *_count_1000m with preference="prefer_high".
- Use restaurants_coffee for cafes, coffee shops, restaurants, and food nearby.
- Use transport for bus stops and public transport; use metro only for explicit subway/metro.
- Air quality preferences map to air_pm25_cold_day or air_pm25_warm_day with preference="prefer_low".

Weights:
- Hard requirements: 1.0.
- Clear preferences: 0.7 to 0.9.
- Soft wishes: 0.4 to 0.6.

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
