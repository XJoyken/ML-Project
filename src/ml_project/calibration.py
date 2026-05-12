from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import PROCESSED_DATASET_PATH, TARGET_COLUMN
from .features import FeatureSchema

DEFAULT_COVERAGE = 0.9
CALIBRATION_SAMPLE = 4000
CALIBRATION_SEED = 7
CALIBRATION_FILENAME = "calibration.json"


@dataclass(slots=True)
class PriceCalibration:
    coverage: float
    log_residual_quantile: float
    sample_size: int

    def interval_for(self, predicted_price_kzt: float) -> tuple[float, float]:
        log_pred = np.log1p(max(predicted_price_kzt, 0.0))
        low = float(np.expm1(log_pred - self.log_residual_quantile))
        high = float(np.expm1(log_pred + self.log_residual_quantile))
        return max(low, 0.0), high

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "log_residual_quantile": self.log_residual_quantile,
            "sample_size": self.sample_size,
        }


def load_or_compute_calibration(
    *,
    model_dir: Path,
    model: Any,
    schema: FeatureSchema,
    processed_path: Path = PROCESSED_DATASET_PATH,
    coverage: float = DEFAULT_COVERAGE,
) -> PriceCalibration:
    calibration_path = Path(model_dir) / CALIBRATION_FILENAME
    if calibration_path.exists():
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        if data.get("coverage") == coverage:
            return PriceCalibration(
                coverage=float(data["coverage"]),
                log_residual_quantile=float(data["log_residual_quantile"]),
                sample_size=int(data["sample_size"]),
            )

    calibration = compute_calibration(
        model=model,
        schema=schema,
        processed_path=processed_path,
        coverage=coverage,
    )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(
        json.dumps(calibration.to_dict(), indent=2),
        encoding="utf-8",
    )
    return calibration


def compute_calibration(
    *,
    model: Any,
    schema: FeatureSchema,
    processed_path: Path,
    coverage: float,
    sample_size: int = CALIBRATION_SAMPLE,
    seed: int = CALIBRATION_SEED,
) -> PriceCalibration:
    processed = pd.read_csv(processed_path)
    if len(processed) > sample_size:
        processed = processed.sample(n=sample_size, random_state=seed)
    predictions = model.predict(processed, schema)
    log_actual = np.log1p(processed[TARGET_COLUMN].to_numpy(dtype=float))
    log_predicted = np.log1p(np.clip(predictions.astype(float), a_min=0.0, a_max=None))
    abs_residuals = np.abs(log_actual - log_predicted)
    quantile = float(np.quantile(abs_residuals, coverage))
    return PriceCalibration(
        coverage=coverage,
        log_residual_quantile=quantile,
        sample_size=int(len(processed)),
    )
