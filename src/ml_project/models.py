from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .constants import TARGET_COLUMN


class CatBoostPriceModel:
    model_name = "catboost"

    def __init__(self, **params: object) -> None:
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
        self.model: CatBoostRegressor | None = None

    def fit(self, train_frame: pd.DataFrame, schema) -> None:
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            train_frame[schema.feature_columns],
            np.log1p(train_frame[TARGET_COLUMN]),
            cat_features=schema.categorical_features,
            verbose=self.params.get("verbose", False),
        )

    def predict(self, frame: pd.DataFrame, schema) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted or loaded before prediction.")
        return np.expm1(self.model.predict(frame[schema.feature_columns]))

    def save(self, artifact_dir: Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted CatBoost model.")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(artifact_dir / "model.cbm"))
        (artifact_dir / "params.json").write_text(
            json.dumps(self.params, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, artifact_dir: Path) -> "CatBoostPriceModel":
        params_path = artifact_dir / "params.json"
        params = {}
        if params_path.exists():
            params = json.loads(params_path.read_text(encoding="utf-8"))
        model = cls(**params)
        model.model = CatBoostRegressor()
        model.model.load_model(str(artifact_dir / "model.cbm"))
        return model

MODEL_CLASSES = {CatBoostPriceModel.model_name: CatBoostPriceModel}


def create_model(model_name: str, **params: object):
    try:
        model_cls = MODEL_CLASSES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {', '.join(MODEL_CLASSES)}"
        ) from exc
    return model_cls(**params) if params else model_cls()


def load_model(model_name: str, artifact_dir: Path):
    try:
        model_cls = MODEL_CLASSES[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {', '.join(MODEL_CLASSES)}"
        ) from exc
    return model_cls.load(artifact_dir)


__all__ = [
    "CatBoostPriceModel",
    "create_model",
    "load_model",
]
