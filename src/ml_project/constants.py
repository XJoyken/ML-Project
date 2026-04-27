from __future__ import annotations

from pathlib import Path

EARTH_RADIUS_KM = 6_371.0088

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "ads_model_v1.csv"
DEFAULT_MODEL_NAME = "catboost"

POI_SOURCE_FILES = {
    "metro": DATA_DIR / "almaty_metro.csv",
    "transport": DATA_DIR / "almaty_transport.csv",
    "schools": DATA_DIR / "almaty_schools.csv",
    "kindergartens": DATA_DIR / "almaty_kindergartens.csv",
    "universities": DATA_DIR / "almaty_university_filtered.csv",
    "clinics": DATA_DIR / "almaty_clinics.csv",
    "dentistry": DATA_DIR / "almaty_dentistry.csv",
    "hospitals": DATA_DIR / "almaty_hospitals.csv",
    "polyclinics": DATA_DIR / "almaty_polyclinics.csv",
    "medcenters": DATA_DIR / "almaty_medcenters.csv",
    "energy": DATA_DIR / "almaty_energy.csv",
}

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
    "scraped_at",
    "listing_added_at",
]

TARGET_COLUMN = "target_price_kzt"
TARGET_LOG_COLUMN = "target_log_price"

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
    "days_since_added_to_scrape",
]

METADATA_COLUMNS = [
    "listing_id",
    TARGET_COLUMN,
    TARGET_LOG_COLUMN,
    "scraped_at",
    "listing_added_at",
]

DEFAULT_EXPLANATION_CATEGORIES = ("schools", "universities")
EXPLANATION_RADIUS_M = 300.0

VERDICT_UNDERVALUED_THRESHOLD = 0.95
VERDICT_OVERPRICED_THRESHOLD = 1.05

POI_CATEGORY_LABELS = {
    "schools": "школа",
    "universities": "университет",
    "metro": "метро",
    "transport": "транспорт",
    "kindergartens": "детский сад",
    "clinics": "клиника",
    "dentistry": "стоматология",
    "hospitals": "больница",
    "polyclinics": "поликлиника",
    "medcenters": "медцентр",
    "energy": "энергообъект",
}
