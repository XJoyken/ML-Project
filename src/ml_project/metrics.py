from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.clip(np.asarray(y_pred, dtype=float), a_min=0, a_max=None)
    rmse = float(np.sqrt(np.mean((y_true_array - y_pred_array) ** 2)))
    rmse_log = float(
        np.sqrt(np.mean((np.log1p(y_true_array) - np.log1p(y_pred_array)) ** 2))
    )
    return {
        "mae_kzt": float(mean_absolute_error(y_true_array, y_pred_array)),
        "mape": float(mean_absolute_percentage_error(y_true_array, y_pred_array)),
        "rmse_kzt": rmse,
        "rmse_log": rmse_log,
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }
