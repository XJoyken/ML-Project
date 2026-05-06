from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import TARGET_COLUMN


class CatBoostPriceModel:
    model_name = "catboost"

    def __init__(self, **params: object):
        from catboost import CatBoostRegressor

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

    def log_to_mlflow(self, name: str, metadata: dict | None = None):
        import mlflow.catboost

        mlflow.catboost.log_model(self.model, name=name, metadata=metadata)


class LightGBMPriceModel:
    model_name = "lightgbm"

    def __init__(self, **params: object):
        from lightgbm import LGBMRegressor

        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "num_leaves": 127,
            "min_child_samples": 30,
            "reg_lambda": 5.0,
            "random_state": 42,
            "verbosity": -1,
            **params,
        }
        self.model = LGBMRegressor(**self.params)

    def fit(self, train_frame: pd.DataFrame, schema):
        features = prepare_features(train_frame, schema)
        self.model.fit(
            features,
            np.log1p(train_frame[TARGET_COLUMN]),
            categorical_feature=schema.categorical_features,
        )

    def predict(self, frame: pd.DataFrame, schema) -> np.ndarray:
        return np.expm1(self.model.predict(prepare_features(frame, schema)))

    def log_to_mlflow(self, name: str, metadata: dict | None = None):
        import mlflow.lightgbm

        mlflow.lightgbm.log_model(self.model, name=name, metadata=metadata)


class XGBoostPriceModel:
    model_name = "xgboost"

    def __init__(self, **params: object):
        from xgboost import XGBRegressor

        self.params = {
            "objective": "reg:squarederror",
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 30,
            "reg_lambda": 5.0,
            "random_state": 42,
            "tree_method": "hist",
            "enable_categorical": True,
            "verbosity": 0,
            **params,
        }
        self.model = XGBRegressor(**self.params)

    def fit(self, train_frame: pd.DataFrame, schema):
        features = prepare_features(train_frame, schema)
        self.model.fit(features, np.log1p(train_frame[TARGET_COLUMN]))

    def predict(self, frame: pd.DataFrame, schema) -> np.ndarray:
        return np.expm1(self.model.predict(prepare_features(frame, schema)))

    def log_to_mlflow(self, name: str, metadata: dict | None = None):
        import mlflow.xgboost

        mlflow.xgboost.log_model(self.model, name=name, metadata=metadata)


def prepare_features(frame: pd.DataFrame, schema) -> pd.DataFrame:
    features = frame[schema.feature_columns].copy()
    for column in schema.categorical_features:
        features[column] = features[column].astype("category")
    return features


MODEL_CLASSES = {
    CatBoostPriceModel.model_name: CatBoostPriceModel,
    LightGBMPriceModel.model_name: LightGBMPriceModel,
    XGBoostPriceModel.model_name: XGBoostPriceModel,
}


def create_model(model_name: str, **params: object):
    if model_name not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_CLASSES)}"
        )
    return MODEL_CLASSES[model_name](**params)


def wrap_loaded_model(model_name: str, loaded):
    cls = MODEL_CLASSES[model_name]
    instance = cls.__new__(cls)
    instance.model = loaded
    instance.params = {}
    return instance


def load_mlflow_model(model_name: str, model_uri: str):
    if model_name == "catboost":
        import mlflow.catboost

        loaded = mlflow.catboost.load_model(model_uri)
    elif model_name == "lightgbm":
        import mlflow.lightgbm

        loaded = mlflow.lightgbm.load_model(model_uri)
    elif model_name == "xgboost":
        import mlflow.xgboost

        loaded = mlflow.xgboost.load_model(model_uri)
    else:
        raise ValueError(f"Unknown model '{model_name}'")
    return wrap_loaded_model(model_name, loaded)


__all__ = [
    "CatBoostPriceModel",
    "LightGBMPriceModel",
    "XGBoostPriceModel",
    "create_model",
    "load_mlflow_model",
    "MODEL_CLASSES",
]
