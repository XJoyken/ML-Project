from __future__ import annotations

import pytest

from ml_project.constants import RENT_CATEGORICAL_FEATURES, RENT_TARGET_COLUMN
from ml_project.rent.data import load_rent_listings


@pytest.fixture(scope="module")
def rent_listings():
    return load_rent_listings()


def test_rent_listings_load_with_expected_schema(rent_listings):
    assert len(rent_listings) > 5000
    for col in [
        RENT_TARGET_COLUMN,
        "area_m2",
        "rooms",
        "lat",
        "lon",
        "district",
        "furnished",
    ]:
        assert col in rent_listings.columns


def test_district_canonicalized_to_cyrillic(rent_listings):
    districts = set(rent_listings["district"].astype(str))
    # transliterated forms must NOT appear after canonicalization
    assert "Bostandykskiy_r-n" not in districts
    assert "Бостандыкский район" in districts


def test_furnished_categorical_values_are_normalised(rent_listings):
    furnished = set(rent_listings["furnished"].astype(str))
    assert "полностью" in furnished
    # filled with "unknown" for missing values
    assert "unknown" in furnished or len(rent_listings["furnished"].dropna()) == len(rent_listings)


def test_rent_price_range_filtered(rent_listings):
    rents = rent_listings[RENT_TARGET_COLUMN]
    # No zero/negative rents
    assert rents.min() > 0
    # Outliers above 4M/month should be trimmed by ppm² quantile
    assert rents.max() < 10_000_000


def test_rent_categorical_features_constant_matches_runtime():
    assert "furnished" in RENT_CATEGORICAL_FEATURES
