from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from .features import FeatureSchema
from .models import load_mlflow_model


def log_training_run(
    *,
    experiment_name: str,
    model_name: str,
    model,
    schema: FeatureSchema,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame | None,
    metrics_frame: pd.DataFrame | None,
    processed_rows: int,
    validation_rows: int,
    test_size: float,
    random_state: int,
    full: bool = False,
):
    params = {
        "model": model_name,
        "test_size": test_size,
        "random_state": random_state,
        "processed_rows": processed_rows,
        "validation_rows": validation_rows,
        "feature_count": len(schema.feature_columns),
        "full_dataset": full,
        **model.params,
    }

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        if metrics_frame is not None:
            metrics = metrics_frame.drop(columns=["model"]).iloc[0].to_dict()
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        log_dataset(train_frame, name="processed_train", context="training")
        if valid_frame is not None:
            log_dataset(valid_frame, name="processed_validation", context="validation")
        mlflow.log_dict(schema.to_dict(), "schema.json")
        model.log_to_mlflow(name="model", metadata=run_metadata(schema))


def log_dataset(frame: pd.DataFrame, *, name: str, context: str):
    dataset = mlflow.data.from_pandas(frame, name=name)
    mlflow.log_input(dataset, context=context)


def run_metadata(schema: FeatureSchema) -> dict[str, Any]:
    return {
        "target_column": schema.target_column,
        "numeric_features": schema.numeric_features,
        "categorical_features": schema.categorical_features,
    }


def load_latest_model(experiment_name: str, model_name: str):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise FileNotFoundError(
            f"MLflow experiment '{experiment_name}' not found. Run train first."
        )
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model = '{model_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise FileNotFoundError(
            f"No runs with model='{model_name}' in experiment '{experiment_name}'."
        )
    run = runs[0]
    model = load_mlflow_model(model_name, f"runs:/{run.info.run_id}/model")
    schema_path = Path(client.download_artifacts(run.info.run_id, "schema.json"))
    schema_dict = json.loads(schema_path.read_text(encoding="utf-8"))
    schema = FeatureSchema(
        numeric_features=list(schema_dict["numeric_features"]),
        categorical_features=list(schema_dict["categorical_features"]),
        target_column=schema_dict["target_column"],
    )
    return run, model, schema
