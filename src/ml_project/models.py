from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .constants import TARGET_COLUMN


class CatBoostPriceModel:
    model_name = "catboost"

    def __init__(self, **params: object):
        self.params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 1200,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "min_data_in_leaf": 30,
            "random_seed": 42,
            "allow_writing_files": False,
            "verbose": False,
            **params,
        }
        self.model = CatBoostRegressor(**self.params)

    def fit(self, train_frame: pd.DataFrame, schema):
        self.model.fit(
            train_frame[schema.feature_columns],
            np.log1p(train_frame[TARGET_COLUMN]),
            cat_features=schema.categorical_features,
            verbose=self.params.get("verbose", False),
        )

    def predict(self, frame: pd.DataFrame, schema) -> np.ndarray:
        return np.expm1(self.model.predict(frame[schema.feature_columns]))


MODEL_CLASSES = {CatBoostPriceModel.model_name: CatBoostPriceModel}


def create_model(model_name: str, **params: object):
    return MODEL_CLASSES[model_name](**params)


__all__ = [
    "CatBoostPriceModel",
    "create_model",
]
