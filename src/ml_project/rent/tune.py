from __future__ import annotations

import json
import random
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..air import load_air_catalog
from ..constants import RENT_MLFLOW_EXPERIMENT_NAME, ROOT_DIR
from ..crime import load_crime_catalog
from ..poi import load_poi_catalog
from ..train import stratify_bins
from ..tune import (
    PARAM_GRIDS,
    _json_default,
    kfold_score,
    sample_params,
    score_on_split,
)
from .data import load_rent_listings
from .features import build_rent_processed_dataset


def prepare_rent_dataset() -> tuple[pd.DataFrame, Any]:
    listings = load_rent_listings()
    poi_catalog = load_poi_catalog()
    air_catalog = load_air_catalog()
    crime_catalog = load_crime_catalog()
    return build_rent_processed_dataset(listings, poi_catalog, air_catalog, crime_catalog)


def tune_rent_random(
    model_name: str,
    *,
    trials: int,
    seed: int,
    experiment_name: str,
) -> pd.DataFrame:
    processed, schema = prepare_rent_dataset()
    train_frame, valid_frame = train_test_split(
        processed,
        test_size=0.2,
        random_state=42,
        stratify=stratify_bins(processed[schema.target_column]),
    )
    grid = PARAM_GRIDS[model_name]
    rng = random.Random(seed)
    mlflow.set_experiment(experiment_name)

    history = []
    for trial in range(1, trials + 1):
        params = sample_params(grid, rng)
        started = time.time()
        with mlflow.start_run(run_name=f"{model_name}-trial-{trial}"):
            mlflow.log_params({"model": model_name, "mode": "random", "target": "rent", **params})
            metrics = score_on_split(model_name, params, train_frame, valid_frame, schema)
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            history.append({"trial": trial, **params, **metrics})
            print(
                f"trial {trial:>3}: mae={metrics['mae_kzt']:>12,.0f}  "
                f"r2={metrics['r2']:.3f}  time={time.time()-started:.0f}s"
            )

    return pd.DataFrame(history).sort_values("mae_kzt").reset_index(drop=True)


def tune_rent_nested(
    model_name: str,
    *,
    trials: int,
    outer_splits: int,
    inner_splits: int,
    seed: int,
    experiment_name: str,
) -> pd.DataFrame:
    processed, schema = prepare_rent_dataset()
    grid = PARAM_GRIDS[model_name]
    rng = random.Random(seed)
    outer_kf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_strata = stratify_bins(processed[schema.target_column])
    mlflow.set_experiment(experiment_name)

    rows: list[dict[str, Any]] = []
    started_total = time.time()

    with mlflow.start_run(run_name=f"{model_name}-rent-nested-cv"):
        mlflow.log_params(
            {
                "model": model_name,
                "mode": "nested-cv",
                "target": "rent",
                "trials_per_fold": trials,
                "outer_splits": outer_splits,
                "inner_splits": inner_splits,
                "seed": seed,
            }
        )
        for outer_fold, (otr_idx, ote_idx) in enumerate(
            outer_kf.split(processed, outer_strata), start=1
        ):
            outer_train = processed.iloc[otr_idx].reset_index(drop=True)
            outer_test = processed.iloc[ote_idx].reset_index(drop=True)

            best = {"inner_mae": float("inf"), "params": None}
            fold_started = time.time()

            for trial in range(1, trials + 1):
                params = sample_params(grid, rng)
                trial_started = time.time()
                inner_metrics = kfold_score(
                    model_name,
                    params,
                    outer_train,
                    schema,
                    n_splits=inner_splits,
                    random_state=seed + outer_fold,
                )
                if inner_metrics["mae_kzt"] < best["inner_mae"]:
                    best = {"inner_mae": inner_metrics["mae_kzt"], "params": params}
                print(
                    f"  fold {outer_fold} trial {trial:>3}: inner_mae={inner_metrics['mae_kzt']:>12,.0f}  "
                    f"time={time.time()-trial_started:.0f}s"
                )

            outer_metrics = score_on_split(
                model_name, best["params"], outer_train, outer_test, schema
            )
            with mlflow.start_run(run_name=f"outer-fold-{outer_fold}", nested=True):
                mlflow.log_params({"model": model_name, **best["params"]})
                mlflow.log_metric("inner_cv_mae", best["inner_mae"])
                mlflow.log_metrics(
                    {f"outer_{k}": float(v) for k, v in outer_metrics.items()}
                )
            rows.append(
                {
                    "fold": outer_fold,
                    **best["params"],
                    "inner_cv_mae": best["inner_mae"],
                    **{f"outer_{k}": v for k, v in outer_metrics.items()},
                }
            )
            print(
                f"== outer fold {outer_fold}/{outer_splits} done: "
                f"outer_mae={outer_metrics['mae_kzt']:,.0f}  "
                f"inner_cv_mae={best['inner_mae']:,.0f}  "
                f"time={time.time()-fold_started:.0f}s"
            )

        history = pd.DataFrame(rows)
        for key in ["mae_kzt", "mape", "rmse_kzt", "r2"]:
            mlflow.log_metric(f"mean_outer_{key}", float(history[f"outer_{key}"].mean()))
            mlflow.log_metric(f"std_outer_{key}", float(history[f"outer_{key}"].std()))

        best_row = history.loc[history["outer_mae_kzt"].idxmin()]
        best_params = {key: best_row[key] for key in PARAM_GRIDS[model_name].keys()}
        save_path = ROOT_DIR / "src" / "configs" / f"{model_name}_rent_nested_best.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(best_params, indent=2, default=_json_default), encoding="utf-8")
        mlflow.log_artifact(str(save_path))
        print(f"\nSaved best-fold params to {save_path}")
        print(f"Total nested-CV time: {(time.time()-started_total)/60:.1f} min")

    return history


def tune_rent_model(
    model_name: str,
    *,
    mode: str = "random",
    trials: int = 15,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 42,
    experiment_name: str = RENT_MLFLOW_EXPERIMENT_NAME + "-tune",
) -> pd.DataFrame:
    if mode == "random":
        return tune_rent_random(
            model_name,
            trials=trials,
            seed=seed,
            experiment_name=experiment_name,
        )
    if mode == "nested":
        return tune_rent_nested(
            model_name,
            trials=trials,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            seed=seed,
            experiment_name=experiment_name + "-nested",
        )
    raise ValueError(f"Unknown tune mode '{mode}'. Use 'random' or 'nested'.")


__all__ = ["prepare_rent_dataset", "tune_rent_model"]
