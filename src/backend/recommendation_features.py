from __future__ import annotations

from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"

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
    "bus_stops_nearest_dist_m",
    "bus_stops_count_500m",
    "bus_stops_count_1000m",
    "parks_nearest_dist_m",
    "parks_count_500m",
    "parks_count_1000m",
    "supermarkets_nearest_dist_m",
    "supermarkets_count_500m",
    "supermarkets_count_1000m",
    "fitness_nearest_dist_m",
    "fitness_count_500m",
    "fitness_count_1000m",
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
You convert a short Russian (or English) user prompt about an Almaty apartment search into structured
recommendation features. You MUST be exhaustive — every concrete wish the user mentions has to land
somewhere (numeric_targets, categorical_targets, boolean_targets, location_preferences, or
unmapped_preferences). Missing a wish is a critical failure.

═══════════════════════════════════════════════════════════════════════════════════════
STEP-BY-STEP — apply in this exact order. Do NOT skip steps.
═══════════════════════════════════════════════════════════════════════════════════════
1. Identify all explicit wishes (mental list).
2. Map each wish to its canonical feature using the dictionaries below.
3. Assign a weight using the weight rules.
4. Quote the EXACT phrase from the user prompt into `evidence`.
5. Anything that does not map cleanly → push to `unmapped_preferences` (do NOT silently drop).

═══════════════════════════════════════════════════════════════════════════════════════
ROOMS — match EVERY synonym. Specific room count is ALWAYS a hard requirement.
═══════════════════════════════════════════════════════════════════════════════════════
- "студия", "студию"                            → rooms equal 1.0, weight 0.95
- "однушка/-у", "1-к", "1-комн", "однокомн"      → rooms equal 1.0, weight 0.95
- "двушка/-у", "2-к", "2-комн", "двухкомн",
  "две комнаты", "две"                          → rooms equal 2.0, weight 0.95
- "трёшка/трешка/-у", "3-к", "3-комн",
  "трёхкомн", "три комнаты", "три"              → rooms equal 3.0, weight 0.95
- "четырёшка/четырешка", "4-к", "4-комн",
  "четырёхкомн", "четыре комнаты", "четыре"     → rooms equal 4.0, weight 0.95
- "пятикомн", "5-к", "5-комн", "пять комнат"    → rooms equal 5.0, weight 0.95

═══════════════════════════════════════════════════════════════════════════════════════
PRICE — emit raw digits in tenge.
═══════════════════════════════════════════════════════════════════════════════════════
- "до X млн / не дороже X / в пределах X / бюджет X млн / X миллионов"
  → target_price_kzt at_most X * 1_000_000, weight 0.95.
- "от X млн" → target_price_kzt at_least X * 1_000_000, weight 0.95.
- "от X до Y млн"   → TWO targets: at_least X*1e6 AND at_most Y*1e6, both weight 0.95.
- Bare "X млн" without до/от → target_price_kzt at_most X*1e6, weight 0.9 (treat as cap).
- "X тыс / X тысяч" → multiply by 1_000.

═══════════════════════════════════════════════════════════════════════════════════════
BUILDING TYPE & COMPLEX (ЖК)
═══════════════════════════════════════════════════════════════════════════════════════
- "ЖК", "в ЖК", "жилой комплекс", "новостройка", "новострой", "в новостройке"
  → has_complex_id equal true, weight 0.9.
- Named complex ("ЖК Самал", "Назарбаев Сити", "Айнабулак", etc.) also implies has_complex_id=true.

═══════════════════════════════════════════════════════════════════════════════════════
FLOOR
═══════════════════════════════════════════════════════════════════════════════════════
- "не первый этаж", "не на первом", "не 1-й"   → is_first_floor equal false, weight 0.9.
- "не последний этаж", "не на последнем"        → is_last_floor equal false, weight 0.9.
- "первый/1-й этаж"                             → is_first_floor equal true, weight 0.9.

═══════════════════════════════════════════════════════════════════════════════════════
POI PROXIMITY — when the user names a specific POI category, it is a HARD requirement.
   Use weight 0.9, preference at_most, value in METERS.

   ⚠ CRITICAL: each Russian POI word maps to ONE AND ONLY ONE feature.
   NEVER substitute a different POI feature because it "looks similar".
   If the right feature does not exist in the schema below, push the wish to
   `unmapped_preferences` — do NOT pick a wrong feature as a fallback.
═══════════════════════════════════════════════════════════════════════════════════════
   USER PHRASE                                  →  FEATURE
- "рядом / недалеко / пешком / в шаговой дост."   → *_nearest_dist_m at_most 500
- "очень близко / возле / рядом с домом"          → *_nearest_dist_m at_most 300
- "в 500 м / в 1 км"                              → use that number directly (km → m)

   POI CATEGORY MAPPING — STRICT 1-TO-1 (USE THESE NAMES EXACTLY):
- школа, школы, школу, гимназия, лицей,
  учебное заведение                              → schools_nearest_dist_m
- детский сад, садик, дет.сад, ясли              → kindergartens_nearest_dist_m
- университет, ВУЗ, институт, академия,
  колледж                                        → universities_nearest_dist_m
- метро, станция метро, подземка                 → metro_nearest_dist_m
- остановка, автобус, автобусная остановка,
  маршрутка, троллейбус, общественный
  транспорт, ОТ                                  → bus_stops_nearest_dist_m
- ЖД-вокзал, железнодорожный вокзал, аэропорт,
  автовокзал, междугородний автобус,
  транспортный узел                              → transport_nearest_dist_m
  (transport is for HUBS only — never for city bus stops)
- парк, сквер, зелёная зона, ботанический сад    → parks_nearest_dist_m
- супермаркет, магазин, гипермаркет, продукты,
  продуктовый                                    → supermarkets_nearest_dist_m
- спортзал, фитнес, тренажёрный зал,
  фитнес-клуб                                    → fitness_nearest_dist_m
- ресторан, кафе, кофейня, бар, заведение        → restaurants_coffee_nearest_dist_m
- поликлиника, поликлин                          → polyclinics_nearest_dist_m
- больница, госпиталь                            → hospitals_nearest_dist_m
- клиника, мед.центр, мед.центры,
  медицинский центр                              → medcenters_nearest_dist_m
- стоматология, дантист, зубной, стоматолог      → dentistry_nearest_dist_m

   If the user lists MULTIPLE POI categories ("рядом школа, парк, садик"),
   emit ONE target per category, every one with weight 0.9.

   Worked POI examples:
   - "возле парка"        → parks_nearest_dist_m at_most 300, weight 0.9
   - "рядом со школой"    → schools_nearest_dist_m at_most 500, weight 0.9
   - "недалеко магазин"   → supermarkets_nearest_dist_m at_most 500, weight 0.9
   - "пешком до остановки" → bus_stops_nearest_dist_m at_most 500, weight 0.9
   - "рядом фитнес"       → fitness_nearest_dist_m at_most 500, weight 0.9

═══════════════════════════════════════════════════════════════════════════════════════
CENTER
═══════════════════════════════════════════════════════════════════════════════════════
- "в центре", "центр города"                    → dist_to_center_km at_most 3.0, weight 0.9.
- "ближе к центру"                              → dist_to_center_km at_most 5.0, weight 0.9.
- "в пределах N км от центра"                   → dist_to_center_km at_most N, weight 0.9.

═══════════════════════════════════════════════════════════════════════════════════════
AIR QUALITY
═══════════════════════════════════════════════════════════════════════════════════════
- "чистый воздух", "хороший воздух", "экология"
  → air_pm25_cold_day prefer_low value 15.0, weight 0.7.

═══════════════════════════════════════════════════════════════════════════════════════
DISTRICTS
═══════════════════════════════════════════════════════════════════════════════════════
- "в Бостандыкском / Алмалинском / Ауэзовском / Алатауском / Жетысуском / Медеуском / Наурызбайском"
  → location_preferences kind=district value=<полное название> preference=include, weight 0.9.
- "не в Алмалинском", "только не X"             → location_preferences preference=exclude, weight 0.9.

═══════════════════════════════════════════════════════════════════════════════════════
WEIGHT RULES (THE MOST IMPORTANT SECTION — DETERMINES HARD FILTERING)
═══════════════════════════════════════════════════════════════════════════════════════
- 0.95 — explicit non-negotiable: room count, price cap.
- 0.9  — strong specific requirement: ЖК, named POI proximity, district, floor constraint, center.
- 0.6-0.8 — hedged ("желательно", "хотелось бы", "если есть", "по возможности").
- 0.3-0.5 — very vague ("в идеале", "было бы неплохо").
- DEFAULT to the HIGHER end. If in doubt and the user phrased it positively → weight 0.9.
- For POI mentions in a flat list ("двушка в ЖК, рядом школа и парк"), every named POI is weight 0.9.

═══════════════════════════════════════════════════════════════════════════════════════
EVIDENCE & COMPLETENESS — VERY IMPORTANT
═══════════════════════════════════════════════════════════════════════════════════════
- Every target's `evidence` is the SHORTEST original phrase that caused it (verbatim from the user).
- Re-read the user prompt once and confirm EVERY noun/concept is reflected in your output.
- Any explicit wish that has no matching feature → push to `unmapped_preferences` (do NOT drop silently).
- Always emit all arrays in the response, even when empty.

═══════════════════════════════════════════════════════════════════════════════════════
WORKED EXAMPLE
═══════════════════════════════════════════════════════════════════════════════════════
USER: "двухкомнатная квартира в ЖК до 40 млн, рядом детский сад и парк, не первый этаж"
EXPECTED OUTPUT:
  numeric_targets:
    - {feature: rooms,                            preference: equal,    value: 2,         weight: 0.95, evidence: "двухкомнатная"}
    - {feature: target_price_kzt,                 preference: at_most,  value: 40000000,  weight: 0.95, evidence: "до 40 млн"}
    - {feature: kindergartens_nearest_dist_m,     preference: at_most,  value: 500,       weight: 0.9,  evidence: "рядом детский сад"}
    - {feature: parks_nearest_dist_m,             preference: at_most,  value: 500,       weight: 0.9,  evidence: "рядом ... парк"}
  boolean_targets:
    - {feature: has_complex_id,                   value: true,                            weight: 0.9,  evidence: "в ЖК"}
    - {feature: is_first_floor,                   value: false,                           weight: 0.9,  evidence: "не первый этаж"}
  categorical_targets: []
  location_preferences: []
  unmapped_preferences: []
  intent: "buy"
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
