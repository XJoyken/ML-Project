from __future__ import annotations

from pathlib import Path

EARTH_RADIUS_KM = 6_371.0088

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "ads_model_v1.csv"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_APARTMENT_MODEL_DIR = MODELS_DIR / "apartment_price_model"
DEFAULT_MODEL_NAME = "lightgbm"
MLFLOW_EXPERIMENT_NAME = "almaty-apartment-prices"

POI_SOURCE_FILES = {
    "metro": DATA_DIR / "almaty_metro.csv",
    "transport": DATA_DIR / "almaty_transport.csv",
    "bus_stops": DATA_DIR / "almaty_bus_stops.csv",
    "schools": DATA_DIR / "almaty_schools.csv",
    "kindergartens": DATA_DIR / "almaty_kindergartens.csv",
    "universities": DATA_DIR / "almaty_university_filtered.csv",
    "clinics": DATA_DIR / "almaty_clinics.csv",
    "dentistry": DATA_DIR / "almaty_dentistry.csv",
    "hospitals": DATA_DIR / "almaty_hospitals.csv",
    "polyclinics": DATA_DIR / "almaty_polyclinics.csv",
    "medcenters": DATA_DIR / "almaty_medcenters.csv",
    "energy": DATA_DIR / "almaty_energy.csv",
    "restaurants_coffee": DATA_DIR / "almaty_restaurants_coffee.csv",
    "parks": DATA_DIR / "almaty_parks.csv",
    "supermarkets": DATA_DIR / "almaty_supermarkets.csv",
    "fitness": DATA_DIR / "almaty_fitness.csv",
}

AIR_QUALITY_RAW_PATH = DATA_DIR / "air_quality.csv"
AIR_PM25_MAX_PLAUSIBLE = 500.0

CRIME_RAW_PATH = DATA_DIR / "almaty_crime_rate_by_district.csv"
CRIME_REFERENCE_YEAR = 2025

MACRO_INFLATION_PATH = DATA_DIR / "cleaned_macroeconomics.csv"
MACRO_MARKET_PATH = DATA_DIR / "cleaned_real_estate_market.csv"

ADS_COLUMNS = [
    "listing_id",
    "listing_category_alias",
    "listing_price_kzt",
    "listing_square_m2",
    "listing_rooms",
    "location_lat",
    "location_lon",
    "address_district",
    "address_microdistrict",
    "listing_complex_id",
    "summary_Год_постройки",
    "summary_Этаж",
    "summary_Тип_дома",
    "summary_Состояние_квартиры",
    "summary_Высота_потолков",
    "summary_Санузел",
    "listing_has_photo",
    "listing_photo_count",
]

TARGET_COLUMN = "target_price_kzt"

CATEGORICAL_FEATURES = [
    "district",
    "house_type",
    "condition",
    "bathroom_type",
]

BASE_NUMERIC_FEATURES = [
    "area_m2",
    "rooms",
    "lat",
    "lon",
    "year_built",
    "floor_current",
    "floors_total",
    "ceiling_height_m",
    "photo_count",
    "has_photo",
    "has_complex_id",
    "has_microdistrict",
    "complex_listing_count",
    "floor_ratio",
    "building_age",
    "area_per_room",
    "is_first_floor",
    "is_last_floor",
    "dist_to_center_km",
]

ALMATY_CENTER_LAT = 43.255
ALMATY_CENTER_LON = 76.935
REFERENCE_YEAR = 2026
PRICE_PER_M2_LOWER_QUANTILE = 0.005
PRICE_PER_M2_UPPER_QUANTILE = 0.995
AREA_M2_MIN = 15.0
AREA_M2_MAX = 500.0
ROOMS_MAX = 8
YEAR_BUILT_MIN = 1900
YEAR_BUILT_MAX = 2027

DEFAULT_EXPLANATION_CATEGORIES = (
    "schools",
    "universities",
    "kindergartens",
    "metro",
    "bus_stops",
    "transport",
    "clinics",
    "dentistry",
    "hospitals",
    "polyclinics",
    "medcenters",
    "energy",
    "restaurants_coffee",
    "parks",
    "supermarkets",
    "fitness",
)
EXPLANATION_RADIUS_M = 300.0

VERDICT_UNDERVALUED_THRESHOLD = 0.95
VERDICT_OVERPRICED_THRESHOLD = 1.05

POI_CATEGORY_LABELS = {
    "schools": "школа",
    "universities": "университет",
    "metro": "метро",
    "transport": "транспортный узел",
    "bus_stops": "автобусная остановка",
    "kindergartens": "детский сад",
    "clinics": "клиника",
    "dentistry": "стоматология",
    "hospitals": "больница",
    "polyclinics": "поликлиника",
    "medcenters": "медцентр",
    "energy": "энергообъект",
    "restaurants_coffee": "ресторан кофейня",
    "parks": "парк",
    "supermarkets": "супермаркет",
    "fitness": "фитнес-клуб",
}
