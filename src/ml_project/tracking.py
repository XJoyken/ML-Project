from __future__ import annotations

from typing import Any

import mlflow
import mlflow.catboost
import pandas as pd

from .features import FeatureSchema


def log_training_run(
    *,
    experiment_name: str,
    model_name: str,
    model,
    schema: FeatureSchema,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    processed_rows: int,
    validation_rows: int,
    test_size: float,
    random_state: int,
):
    metrics = metrics_frame.drop(columns=["model"]).iloc[0].to_dict()
    params = {
        "model": model_name,
        "test_size": test_size,
        "random_state": random_state,
        "processed_rows": processed_rows,
        "validation_rows": validation_rows,
        "feature_count": len(schema.feature_columns),
        **model.params,
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        log_dataset(train_frame, name="processed_train", context="training")
        log_dataset(valid_frame, name="processed_validation", context="validation")
        mlflow.log_dict(schema.to_dict(), "schema.json")
        mlflow.catboost.log_model(
            model.model,
            name="model",
            metadata=run_metadata(schema),
        )


def log_dataset(frame: pd.DataFrame, *, name: str, context: str):
    dataset = mlflow.data.from_pandas(frame, name=name)
    mlflow.log_input(dataset, context=context)


def run_metadata(schema: FeatureSchema) -> dict[str, Any]:
    return {
        "target_column": schema.target_column,
        "numeric_features": schema.numeric_features,
        "categorical_features": schema.categorical_features,
    }
