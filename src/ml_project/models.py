from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EARLY_STOPPING_ROUNDS = 50
EVAL_SIZE = 0.1
EVAL_RANDOM_STATE = 0


class CatBoostPriceModel:
    model_name = "catboost"
    iterations_param = "iterations"

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
        inner, evaluation = split_for_early_stopping(train_frame)
        self.model.fit(
            inner[schema.feature_columns],
            np.log1p(inner[schema.target_column]),
            cat_features=schema.categorical_features,
            eval_set=(evaluation[schema.feature_columns], np.log1p(evaluation[schema.target_column])),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

    def predict(self, frame: pd.DataFrame, schema) -> np.ndarray:
        return np.expm1(self.model.predict(frame[schema.feature_columns]))

    def best_iteration(self) -> int:
        return int(self.model.tree_count_)

    def refit_no_es(self, train_frame: pd.DataFrame, schema):
        from catboost import CatBoostRegressor

        self.params = {**self.params, self.iterations_param: self.best_iteration()}
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            train_frame[schema.feature_columns],
            np.log1p(train_frame[schema.target_column]),
            cat_features=schema.categorical_features,
            verbose=False,
        )

    def log_to_mlflow(self, name: str, metadata: dict | None = None):
        import mlflow.catboost

        mlflow.catboost.log_model(self.model, name=name, metadata=metadata)


class LightGBMPriceModel:
    model_name = "lightgbm"
    iterations_param = "n_estimators"

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
        from lightgbm import early_stopping, log_evaluation

        self._category_dtypes = category_dtypes_from(train_frame, schema)
        inner, evaluation = split_for_early_stopping(train_frame)
        self.model.fit(
            prepare_features(inner, schema, self._category_dtypes),
            np.log1p(inner[schema.target_column]),
            eval_set=[
                (
                    prepare_features(evaluation, schema, self._category_dtypes),
                    np.log1p(evaluation[schema.target_column]),
                )
            ],
            categorical_feature=schema.categorical_features,
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(0),
            ],
        )

    def predict(self, frame: pd.DataFrame, schema) -> np.ndarray:
        dtypes = getattr(self, "_category_dtypes", None)
        return np.expm1(self.model.predict(prepare_features(frame, schema, dtypes)))

    def best_iteration(self) -> int:
        return int(self.model.best_iteration_)

    def refit_no_es(self, train_frame: pd.DataFrame, schema):
        from lightgbm import LGBMRegressor

        self.params = {**self.params, self.iterations_param: self.best_iteration()}
        self._category_dtypes = category_dtypes_from(train_frame, schema)
        self.model = LGBMRegressor(**self.params)
        self.model.fit(
            prepare_features(train_frame, schema, self._category_dtypes),
            np.log1p(train_frame[schema.target_column]),
            categorical_feature=schema.categorical_features,
        )

    def log_to_mlflow(self, name: str, metadata: dict | None = None):
        import mlflow.lightgbm

        mlflow.lightgbm.log_model(self.model, name=name, metadata=metadata)


class XGBoostPriceModel:
    model_name = "xgboost"
    iterations_param = "n_estimators"

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
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            **params,
        }
        self.model = XGBRegressor(**self.params)

    def fit(self, train_frame: pd.DataFrame, schema):
        self._category_dtypes = category_dtypes_from(train_frame, schema)
        inner, evaluation = split_for_early_stopping(train_frame)
        self.model.fit(
            prepare_features(inner, schema, self._category_dtypes),
            np.log1p(inner[schema.target_column]),
            eval_set=[
                (
                    prepare_features(evaluation, schema, self._category_dtypes),
                    np.log1p(evaluation[schema.target_column]),
                )
            ],
            verbose=False,
        )

    def predict(self, frame: pd.DataFrame, schema) -> np.ndarray:
        dtypes = getattr(self, "_category_dtypes", None)
        return np.expm1(self.model.predict(prepare_features(frame, schema, dtypes)))

    def best_iteration(self) -> int:
        return int(self.model.best_iteration) + 1

    def refit_no_es(self, train_frame: pd.DataFrame, schema):
        from xgboost import XGBRegressor

        new_params = {**self.params, self.iterations_param: self.best_iteration()}
        new_params.pop("early_stopping_rounds", None)
        self.params = new_params
        self._category_dtypes = category_dtypes_from(train_frame, schema)
        self.model = XGBRegressor(**self.params)
        self.model.fit(
            prepare_features(train_frame, schema, self._category_dtypes),
            np.log1p(train_frame[schema.target_column]),
            verbose=False,
        )

    def log_to_mlflow(self, name: str, metadata: dict | None = None):
        import mlflow.xgboost

        mlflow.xgboost.log_model(self.model, name=name, metadata=metadata)


def split_for_early_stopping(frame: pd.DataFrame):
    return train_test_split(frame, test_size=EVAL_SIZE, random_state=EVAL_RANDOM_STATE)


def category_dtypes_from(
    frame: pd.DataFrame, schema
) -> dict[str, pd.CategoricalDtype]:
    return {
        column: pd.CategoricalDtype(
            categories=sorted(frame[column].dropna().astype(str).unique())
        )
        for column in schema.categorical_features
    }


def prepare_features(
    frame: pd.DataFrame,
    schema,
    category_dtypes: dict[str, pd.CategoricalDtype] | None = None,
) -> pd.DataFrame:
    features = frame[schema.feature_columns].copy()
    for column in schema.categorical_features:
        if category_dtypes is not None and column in category_dtypes:
            features[column] = features[column].astype(str).astype(category_dtypes[column])
        else:
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
