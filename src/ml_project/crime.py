from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import CRIME_RAW_PATH, CRIME_REFERENCE_YEAR

CRIME_FEATURE_COLUMNS = [
    "district_crime_count",
    "district_crime_index",
]


@dataclass(slots=True)
class CrimeCatalog:
    year: int
    by_district: dict[str, float]
    max_count: float

    def features_for_district(self, district: str | None) -> dict[str, float]:
        if not district or district not in self.by_district:
            return {
                "district_crime_count": np.nan,
                "district_crime_index": np.nan,
            }
        count = float(self.by_district[district])
        return {
            "district_crime_count": count,
            "district_crime_index": count / self.max_count if self.max_count > 0 else 0.0,
        }

    def build_feature_frame(self, listings: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(index=listings.index)
        districts = listings["district"].astype("string")
        counts = districts.map(self.by_district).astype("float32")
        indices = (counts / self.max_count) if self.max_count > 0 else pd.Series(0.0, index=districts.index)
        frame["district_crime_count"] = counts
        frame["district_crime_index"] = indices.astype("float32")
        return frame

    def lookup(self, district: str | None) -> dict[str, Any] | None:
        if not district or district not in self.by_district:
            return None
        count = float(self.by_district[district])
        return {
            "district": district,
            "year": self.year,
            "count": count,
            "index_min_max": count / self.max_count if self.max_count > 0 else 0.0,
        }


def load_crime_catalog(
    raw_path: Path | None = None,
    *,
    year: int = CRIME_REFERENCE_YEAR,
) -> CrimeCatalog:
    frame = pd.read_csv(raw_path or CRIME_RAW_PATH)
    yearly = frame.loc[frame["year"].eq(year), ["district", "count"]].dropna()
    by_district = {
        str(row["district"]).strip(): float(row["count"]) for _, row in yearly.iterrows()
    }
    max_count = max(by_district.values()) if by_district else 0.0
    return CrimeCatalog(year=year, by_district=by_district, max_count=max_count)
