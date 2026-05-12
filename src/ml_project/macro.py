from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import MACRO_INFLATION_PATH, MACRO_MARKET_PATH


@dataclass(slots=True)
class InflationCatalog:
    by_year: dict[int, dict[str, float]]
    latest_year: int

    def for_year(self, year: int) -> dict[str, float] | None:
        return self.by_year.get(int(year))

    def cumulative_index(self, year: int) -> float | None:
        row = self.for_year(year)
        return None if row is None else float(row["cumulative_index"])

    def annual_pct(self, year: int) -> float | None:
        row = self.for_year(year)
        return None if row is None else float(row["annual_pct"])

    def discount_to_year(self, value: float, *, from_year: int, to_year: int) -> float | None:
        """Convert a nominal KZT value from `from_year` to nominal KZT in `to_year`.

        Multiplies by `index[to_year] / index[from_year]` (real-prices-fixed scaling)."""
        index_from = self.cumulative_index(from_year)
        index_to = self.cumulative_index(to_year)
        if index_from is None or index_to is None or index_from == 0:
            return None
        return value * (index_to / index_from)

    def expected_cpi_growth(self, *, horizon_years: int) -> float:
        """Geometric mean annual inflation over the last 5 known years."""
        recent_years = sorted(self.by_year)[-5:]
        annuals = [self.by_year[year]["annual_pct"] / 100.0 for year in recent_years]
        if not annuals:
            return 0.0
        product = 1.0
        for a in annuals:
            product *= 1.0 + a
        avg_growth = product ** (1 / len(annuals)) - 1.0
        return (1.0 + avg_growth) ** horizon_years - 1.0


def load_inflation_catalog(path: Path | None = None) -> InflationCatalog:
    frame = pd.read_csv(path or MACRO_INFLATION_PATH)
    by_year: dict[int, dict[str, float]] = {}
    for _, row in frame.iterrows():
        year = int(row["Year"])
        by_year[year] = {
            "annual_pct": float(row["Inflation_CPI_Pct_Dec_to_Dec"]),
            "cumulative_index": float(row["Cumulative_Index_Base_2015"]),
        }
    latest_year = max(by_year) if by_year else 0
    return InflationCatalog(by_year=by_year, latest_year=latest_year)


@dataclass(slots=True)
class RealEstateMarketCatalog:
    by_year: dict[int, dict[str, float]]
    latest_year: int

    def for_year(self, year: int) -> dict[str, float] | None:
        return self.by_year.get(int(year))

    def latest(self) -> dict[str, float]:
        return self.by_year[self.latest_year]

    def average_growth(self, column: str, *, lookback_years: int = 5) -> float:
        recent_years = sorted(self.by_year)[-lookback_years:]
        if len(recent_years) < 2:
            return 0.0
        first = self.by_year[recent_years[0]][column]
        last = self.by_year[recent_years[-1]][column]
        years = recent_years[-1] - recent_years[0]
        if first <= 0 or years <= 0:
            return 0.0
        return (last / first) ** (1 / years) - 1.0


def load_market_catalog(path: Path | None = None) -> RealEstateMarketCatalog:
    frame = pd.read_csv(path or MACRO_MARKET_PATH)
    by_year: dict[int, dict[str, float]] = {}
    for _, row in frame.iterrows():
        year = int(row["Year"])
        by_year[year] = {
            "primary_kzt_m2_thousand": float(row["Primary_Market_KZT_m2"]),
            "secondary_kzt_m2_thousand": float(row["Secondary_Market_KZT_m2"]),
            "rent_kzt_m2_mo_thousand": float(row["Rent_KZT_m2_mo"]),
            "rental_yield_annual_pct": float(row["Rental_Yield_Annual_Pct"]),
        }
    latest_year = max(by_year) if by_year else 0
    return RealEstateMarketCatalog(by_year=by_year, latest_year=latest_year)


def summarise(inflation: InflationCatalog, market: RealEstateMarketCatalog) -> dict[str, Any]:
    return {
        "inflation": {
            "latest_year": inflation.latest_year,
            "latest_annual_pct": inflation.annual_pct(inflation.latest_year),
            "five_year_growth_pct": inflation.expected_cpi_growth(horizon_years=5) * 100,
        },
        "market": {
            "latest_year": market.latest_year,
            "secondary_kzt_m2_thousand": market.latest()["secondary_kzt_m2_thousand"],
            "rent_kzt_m2_mo_thousand": market.latest()["rent_kzt_m2_mo_thousand"],
            "rental_yield_annual_pct": market.latest()["rental_yield_annual_pct"],
            "secondary_growth_annual_pct": market.average_growth("secondary_kzt_m2_thousand") * 100,
            "rent_growth_annual_pct": market.average_growth("rent_kzt_m2_mo_thousand") * 100,
        },
    }
