from __future__ import annotations

import pytest

from ml_project.air import load_air_catalog
from ml_project.constants import RENT_CATEGORICAL_FEATURES, RENT_TARGET_COLUMN
from ml_project.crime import load_crime_catalog
from ml_project.poi import load_poi_catalog
from ml_project.rent.data import load_rent_listings
from ml_project.rent.features import (
    build_rent_inference_frame,
    build_rent_processed_dataset,
)


@pytest.fixture(scope="module")
def rent_pipeline():
    listings = load_rent_listings()
    poi_catalog = load_poi_catalog()
    air_catalog = load_air_catalog()
    crime_catalog = load_crime_catalog()
    processed, schema = build_rent_processed_dataset(
        listings, poi_catalog, air_catalog, crime_catalog
    )
    return {
        "listings": listings,
        "poi": poi_catalog,
        "air": air_catalog,
        "crime": crime_catalog,
        "processed": processed,
        "schema": schema,
    }


def test_rent_schema_target_is_rent(rent_pipeline):
    assert rent_pipeline["schema"].target_column == RENT_TARGET_COLUMN


def test_rent_schema_contains_furnished_categorical(rent_pipeline):
    schema = rent_pipeline["schema"]
    assert "furnished" in schema.categorical_features
    assert set(RENT_CATEGORICAL_FEATURES).issubset(set(schema.categorical_features))


def test_rent_processed_has_no_nan_in_target(rent_pipeline):
    processed = rent_pipeline["processed"]
    assert processed[RENT_TARGET_COLUMN].notna().all()


def test_rent_processed_includes_air_and_crime_columns(rent_pipeline):
    processed = rent_pipeline["processed"]
    assert "air_pm25_cold_day" in processed.columns
    assert "district_crime_index" in processed.columns


def test_build_rent_inference_frame_accepts_sale_listing_dict(rent_pipeline):
    """The investment endpoint feeds a sale listing dict into the rent pipeline;
    the inference frame must accept it (target is missing → NaN allowed)."""
    listing = {
        "listing_id": "test-1",
        "lat": 43.24,
        "lon": 76.92,
        "area_m2": 55,
        "rooms": 2,
        "district": "Бостандыкский район",
        "house_type": "монолитный",
        "condition": "свежий ремонт",
        "bathroom_type": "совмещенный",
        # no furnished — model should default to "unknown"
    }
    processed, schema = build_rent_inference_frame(
        listing,
        reference_ads=rent_pipeline["listings"],
        poi_catalog=rent_pipeline["poi"],
        air_catalog=rent_pipeline["air"],
        crime_catalog=rent_pipeline["crime"],
    )
    assert len(processed) == 1
    assert "furnished" in processed.columns
    assert str(processed.iloc[0]["furnished"]) == "unknown"
    assert schema.target_column == RENT_TARGET_COLUMN
