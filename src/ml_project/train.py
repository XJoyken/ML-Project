from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import (
    MLFLOW_EXPERIMENT_NAME,
    POI_SOURCE_FILES,
    PROCESSED_DATASET_PATH,
    TARGET_COLUMN,
)
from .data import load_listings
from .features import build_processed_dataset
from .metrics import calculate_metrics
from .models import create_model
from .poi import load_poi_catalog
from .tracking import log_training_run


def train_model(
    model_name: str = "catboost",
    *,
    ads_path: Path | None = None,
    poi_sources: dict[str, Path] | None = None,
    processed_path: Path | None = PROCESSED_DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: dict[str, object] | None = None,
    save_processed: bool = True,
    log_to_mlflow: bool = True,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> dict[str, Any]:
    listings = load_listings(ads_path=ads_path)
    poi_catalog = load_poi_catalog(poi_sources=poi_sources or POI_SOURCE_FILES)
    processed, schema = build_processed_dataset(
        listings,
        poi_catalog,
        save_path=processed_path if save_processed else None,
    )

    train_frame, valid_frame = train_test_split(
        processed,
        test_size=test_size,
        random_state=random_state,
    )
    model = create_model(model_name, **(model_params or {}))
    model.fit(train_frame, schema)
    predictions = model.predict(valid_frame, schema)

    metrics_frame = pd.DataFrame(
        [
            {
                "model": model_name,
                **calculate_metrics(valid_frame[TARGET_COLUMN], predictions),
            }
        ]
    )

    if log_to_mlflow:
        log_training_run(
            experiment_name=experiment_name,
            model_name=model_name,
            model=model,
            schema=schema,
            train_frame=train_frame,
            valid_frame=valid_frame,
            metrics_frame=metrics_frame,
            processed_rows=len(processed),
            validation_rows=len(valid_frame),
            test_size=test_size,
            random_state=random_state,
        )

    return {
        "listings": listings,
        "poi_catalog": poi_catalog,
        "processed_frame": processed,
        "metrics_frame": metrics_frame,
        "model": model,
        "schema": schema,
    }


def evaluate_model(
    model_name: str = "catboost",
    *,
    ads_path: Path | None = None,
    poi_sources: dict[str, Path] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: dict[str, object] | None = None,
) -> dict[str, Any]:
    return train_model(
        model_name,
        ads_path=ads_path,
        poi_sources=poi_sources,
        processed_path=None,
        test_size=test_size,
        random_state=random_state,
        model_params=model_params,
        save_processed=False,
        log_to_mlflow=False,
    )
