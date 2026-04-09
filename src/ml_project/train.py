from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import ARTIFACTS_DIR, POI_SOURCE_FILES, PROCESSED_DATASET_PATH, TARGET_COLUMN
from .data import load_listings, load_poi_catalog
from .features import FeatureSchema, build_processed_dataset
from .metrics import calculate_metrics
from .models import create_model


def train_model(
    model_name: str = "catboost",
    *,
    ads_path: Path | None = None,
    poi_sources: dict[str, Path] | None = None,
    processed_path: Path | None = PROCESSED_DATASET_PATH,
    artifact_root: Path | None = ARTIFACTS_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
    model_params: dict[str, object] | None = None,
    save_processed: bool = True,
    persist_artifacts: bool = True,
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

    artifact_dir_path = None
    if persist_artifacts:
        artifact_dir_path = save_artifacts(
            model_name,
            model,
            schema,
            metrics_frame,
            artifact_root=artifact_root,
            rows_processed=len(processed),
            validation_rows=len(valid_frame),
        )

    return {
        "processed_frame": processed,
        "metrics_frame": metrics_frame,
        "model": model,
        "schema": schema,
        "artifact_dir": artifact_dir_path,
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
        persist_artifacts=False,
    )


def save_artifacts(
    model_name: str,
    model,
    schema: FeatureSchema,
    metrics_frame: pd.DataFrame,
    *,
    artifact_root: Path | None = ARTIFACTS_DIR,
    rows_processed: int,
    validation_rows: int,
) -> Path:
    artifact_dir_path = artifact_dir(model_name, artifact_root=artifact_root)
    artifact_dir_path.mkdir(parents=True, exist_ok=True)
    model.save(artifact_dir_path)
    schema_path(model_name, artifact_root=artifact_root).write_text(
        json.dumps(schema.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path(model_name, artifact_root=artifact_root).write_text(
        json.dumps(
            {
                "model_name": model_name,
                "rows_processed": rows_processed,
                "validation_rows": validation_rows,
                "feature_columns": schema.feature_columns,
                "metrics": metrics_frame.round(6).to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_dir_path


def load_schema(
    model_name: str,
    *,
    artifact_root: Path | None = ARTIFACTS_DIR,
) -> FeatureSchema:
    payload = json.loads(schema_path(model_name, artifact_root=artifact_root).read_text(encoding="utf-8"))
    return FeatureSchema.from_dict(payload)


def artifact_dir(model_name: str, *, artifact_root: Path | None = ARTIFACTS_DIR) -> Path:
    return (artifact_root or ARTIFACTS_DIR) / model_name


def schema_path(model_name: str, *, artifact_root: Path | None = ARTIFACTS_DIR) -> Path:
    return artifact_dir(model_name, artifact_root=artifact_root) / "schema.json"


def metrics_path(model_name: str, *, artifact_root: Path | None = ARTIFACTS_DIR) -> Path:
    return artifact_dir(model_name, artifact_root=artifact_root) / "metrics.json"
