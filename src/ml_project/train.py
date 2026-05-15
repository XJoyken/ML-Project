from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .air import load_air_catalog
from .constants import (
    AIR_QUALITY_RAW_PATH,
    CRIME_RAW_PATH,
    MLFLOW_EXPERIMENT_NAME,
    POI_SOURCE_FILES,
    PROCESSED_DATASET_PATH,
)
from .crime import load_crime_catalog
from .data import load_listings
from .features import build_processed_dataset
from .metrics import calculate_metrics
from .models import create_model
from .poi import load_poi_catalog
from .tracking import log_training_run

STRATIFY_BINS = 10


def stratify_bins(target: pd.Series, n_bins: int = STRATIFY_BINS) -> pd.Series:
    return pd.qcut(target, q=n_bins, labels=False, duplicates="drop")


def train_model(
    model_name: str = "lightgbm",
    *,
    ads_path: Path | None = None,
    poi_sources: dict[str, Path] | None = None,
    air_quality_path: Path | None = None,
    crime_path: Path | None = None,
    processed_path: Path | None = PROCESSED_DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: dict[str, object] | None = None,
    save_processed: bool = True,
    log_to_mlflow: bool = True,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    final: bool = False,
    full: bool = False,
) -> dict[str, Any]:
    if full and final:
        raise ValueError("--full and --final are mutually exclusive.")

    listings = load_listings(ads_path=ads_path)
    poi_catalog = load_poi_catalog(poi_sources=poi_sources or POI_SOURCE_FILES)
    air_catalog = load_air_catalog(raw_path=air_quality_path or AIR_QUALITY_RAW_PATH)
    crime_catalog = load_crime_catalog(raw_path=crime_path or CRIME_RAW_PATH)
    processed, schema = build_processed_dataset(
        listings,
        poi_catalog,
        air_catalog,
        crime_catalog,
        save_path=processed_path if save_processed else None,
    )

    model = create_model(model_name, **(model_params or {}))

    if full:
        # Two-pass on 100% of the data: ES finds best_iter, then refit on the full
        # dataset without ES. No held-out validation, so no metrics are computed.
        model.fit(processed, schema)
        best_iter = model.best_iteration()
        print(f"Pass 1 finished, best_iteration={best_iter}. Refitting on full dataset without ES.")
        model.refit_no_es(processed, schema)
        metrics_frame: pd.DataFrame | None = None
        train_frame = processed
        valid_frame: pd.DataFrame | None = None
        validation_rows = 0
    else:
        train_frame, valid_frame = train_test_split(
            processed,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_bins(processed[schema.target_column]),
        )
        model.fit(train_frame, schema)
        if final:
            best_iter = model.best_iteration()
            print(f"Pass 1 finished, best_iteration={best_iter}. Refitting on full train without ES.")
            model.refit_no_es(train_frame, schema)
        predictions = model.predict(valid_frame, schema)
        metrics_frame = pd.DataFrame(
            [
                {
                    "model": model_name,
                    **calculate_metrics(valid_frame[schema.target_column], predictions),
                }
            ]
        )
        validation_rows = len(valid_frame)

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
            validation_rows=validation_rows,
            test_size=0.0 if full else test_size,
            random_state=random_state,
            full=full,
        )

    return {
        "listings": listings,
        "poi_catalog": poi_catalog,
        "air_catalog": air_catalog,
        "crime_catalog": crime_catalog,
        "processed_frame": processed,
        "metrics_frame": metrics_frame,
        "model": model,
        "schema": schema,
    }


def evaluate_model(
    model_name: str = "lightgbm",
    *,
    ads_path: Path | None = None,
    poi_sources: dict[str, Path] | None = None,
    air_quality_path: Path | None = None,
    crime_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: dict[str, object] | None = None,
    final: bool = False,
) -> dict[str, Any]:
    return train_model(
        model_name,
        ads_path=ads_path,
        poi_sources=poi_sources,
        air_quality_path=air_quality_path,
        crime_path=crime_path,
        processed_path=None,
        test_size=test_size,
        random_state=random_state,
        model_params=model_params,
        save_processed=False,
        log_to_mlflow=False,
        final=final,
    )
