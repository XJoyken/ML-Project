from __future__ import annotations

import math

import pandas as pd
import pytest

from ml_project.crime import CrimeCatalog, load_crime_catalog
from ml_project.macro import load_inflation_catalog, load_market_catalog


def test_load_crime_catalog_uses_reference_year():
    catalog = load_crime_catalog()
    assert catalog.year == 2025
    assert "Бостандыкский район" in catalog.by_district
    assert catalog.max_count > 0


def test_crime_catalog_lookup_returns_index():
    catalog = load_crime_catalog()
    info = catalog.lookup("Бостандыкский район")
    assert info is not None
    assert info["year"] == 2025
    assert 0.0 <= info["index_min_max"] <= 1.0


def test_crime_catalog_unknown_district_returns_none():
    catalog = CrimeCatalog(year=2025, by_district={"A": 100.0}, max_count=100.0)
    assert catalog.lookup("Несуществующий") is None
    assert catalog.lookup(None) is None


def test_crime_catalog_build_feature_frame_returns_two_columns():
    catalog = CrimeCatalog(year=2025, by_district={"A": 1000.0, "B": 500.0}, max_count=1000.0)
    listings = pd.DataFrame({"district": ["A", "B", "C"]})
    frame = catalog.build_feature_frame(listings)
    assert list(frame.columns) == ["district_crime_count", "district_crime_index"]
    assert frame["district_crime_count"].iloc[0] == pytest.approx(1000.0)
    assert frame["district_crime_index"].iloc[1] == pytest.approx(0.5)
    assert pd.isna(frame["district_crime_count"].iloc[2])


def test_inflation_catalog_monotonic_cumulative_index():
    catalog = load_inflation_catalog()
    years = sorted(catalog.by_year)
    for prev, curr in zip(years, years[1:]):
        assert catalog.by_year[curr]["cumulative_index"] >= catalog.by_year[prev]["cumulative_index"]


def test_inflation_discount_round_trip():
    catalog = load_inflation_catalog()
    later_year = sorted(catalog.by_year)[-1]
    earlier_year = sorted(catalog.by_year)[0]
    adjusted = catalog.discount_to_year(100.0, from_year=earlier_year, to_year=later_year)
    assert adjusted is not None
    assert adjusted > 100.0


def test_inflation_five_year_growth_is_positive():
    catalog = load_inflation_catalog()
    growth = catalog.expected_cpi_growth(horizon_years=5)
    assert growth > 0
    assert not math.isnan(growth)


def test_market_catalog_has_growth_rates():
    catalog = load_market_catalog()
    secondary_growth = catalog.average_growth("secondary_kzt_m2_thousand")
    rent_growth = catalog.average_growth("rent_kzt_m2_mo_thousand")
    assert secondary_growth > 0
    assert rent_growth > 0
