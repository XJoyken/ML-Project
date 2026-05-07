from __future__ import annotations

import argparse
import json
import random
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .air import load_air_catalog
from .constants import MLFLOW_EXPERIMENT_NAME, ROOT_DIR, TARGET_COLUMN
from .data import load_listings
from .features import build_processed_dataset
from .metrics import calculate_metrics
from .models import create_model
from .poi import load_poi_catalog
from .train import stratify_bins

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "catboost": {
        "iterations": [800, 1200, 1800],
        "learning_rate": [0.03, 0.05, 0.08],
        "depth": [6, 8, 10],
        "l2_leaf_reg": [3.0, 5.0, 8.0],
        "min_data_in_leaf": [20, 30, 50],
    },
    "lightgbm": {
        "n_estimators": [1500, 2500, 3500],
        "learning_rate": [0.01, 0.03, 0.08],
        "num_leaves": [127, 255, 400],
        "min_child_samples": [20, 30, 50],
        "reg_lambda": [5.0, 10.0, 15.0],
    },
    "xgboost": {
        "n_estimators": [800, 1500, 2500],
        "learning_rate": [0.03, 0.05, 0.08],
        "max_depth": [6, 8, 10],
        "min_child_weight": [10, 30, 50],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
}


def sample_params(grid: dict[str, list[Any]], rng: random.Random) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in grid.items()}


def prepare_dataset() -> tuple[pd.DataFrame, Any]:
    listings = load_listings()
    poi_catalog = load_poi_catalog()
    air_catalog = load_air_catalog()
    return build_processed_dataset(listings, poi_catalog, air_catalog)


def score_on_split(
    model_name: str,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    schema,
) -> dict[str, float]:
    model = create_model(model_name, **params)
    model.fit(train_frame, schema)
    preds = model.predict(valid_frame, schema)
    return calculate_metrics(valid_frame[TARGET_COLUMN], preds)


def kfold_score(
    model_name: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    schema,
    n_splits: int,
    random_state: int,
) -> dict[str, float]:
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strata = stratify_bins(frame[TARGET_COLUMN])
    metrics_list = []
    for train_idx, valid_idx in kf.split(frame, strata):
        train_fold = frame.iloc[train_idx].reset_index(drop=True)
        valid_fold = frame.iloc[valid_idx].reset_index(drop=True)
        metrics_list.append(score_on_split(model_name, params, train_fold, valid_fold, schema))
    return {key: float(np.mean([m[key] for m in metrics_list])) for key in metrics_list[0]}


def tune_random(
    model_name: str,
    *,
    trials: int,
    seed: int,
    experiment_name: str,
) -> pd.DataFrame:
    processed, schema = prepare_dataset()
    train_frame, valid_frame = train_test_split(
        processed,
        test_size=0.2,
        random_state=42,
        stratify=stratify_bins(processed[TARGET_COLUMN]),
    )
    grid = PARAM_GRIDS[model_name]
    rng = random.Random(seed)
    mlflow.set_experiment(experiment_name)

    history = []
    for trial in range(1, trials + 1):
        params = sample_params(grid, rng)
        started = time.time()
        with mlflow.start_run(run_name=f"{model_name}-trial-{trial}"):
            mlflow.log_params({"model": model_name, "mode": "random", **params})
            metrics = score_on_split(model_name, params, train_frame, valid_frame, schema)
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            history.append({"trial": trial, **params, **metrics})
            print(
                f"trial {trial:>3}: mae={metrics['mae_kzt']:>12,.0f}  "
                f"r2={metrics['r2']:.3f}  time={time.time()-started:.0f}s"
            )

    return pd.DataFrame(history).sort_values("mae_kzt").reset_index(drop=True)


def tune_nested(
    model_name: str,
    *,
    trials: int,
    outer_splits: int,
    inner_splits: int,
    seed: int,
    experiment_name: str,
) -> pd.DataFrame:
    processed, schema = prepare_dataset()
    grid = PARAM_GRIDS[model_name]
    rng = random.Random(seed)
    outer_kf = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    outer_strata = stratify_bins(processed[TARGET_COLUMN])
    mlflow.set_experiment(experiment_name)

    rows: list[dict[str, Any]] = []
    started_total = time.time()

    with mlflow.start_run(run_name=f"{model_name}-nested-cv"):
        mlflow.log_params(
            {
                "model": model_name,
                "mode": "nested-cv",
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
        best_params = {
            key: best_row[key]
            for key in PARAM_GRIDS[model_name].keys()
        }
        save_path = ROOT_DIR / "src" / "configs" / f"{model_name}_nested_best.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(best_params, indent=2, default=_json_default), encoding="utf-8")
        mlflow.log_artifact(str(save_path))
        print(f"\nSaved best-fold params to {save_path}")
        print(f"Total nested-CV time: {(time.time()-started_total)/60:.1f} min")

    return history


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def tune_model(
    model_name: str,
    *,
    mode: str = "random",
    trials: int = 15,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 42,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME + "-tune",
) -> pd.DataFrame:
    if mode == "random":
        return tune_random(
            model_name,
            trials=trials,
            seed=seed,
            experiment_name=experiment_name,
        )
    if mode == "nested":
        return tune_nested(
            model_name,
            trials=trials,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            seed=seed,
            experiment_name=experiment_name + "-nested",
        )
    raise ValueError(f"Unknown tune mode '{mode}'. Use 'random' or 'nested'.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter search.")
    parser.add_argument("--model", required=True, choices=list(PARAM_GRIDS.keys()))
    parser.add_argument("--mode", default="random", choices=["random", "nested"])
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    history = tune_model(
        args.model,
        mode=args.mode,
        trials=args.trials,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        seed=args.seed,
    )
    print()
    if args.mode == "random":
        print("Top 5 by MAE:")
        print(history.head(5).round(4).to_string(index=False))
    else:
        print("Per outer fold:")
        print(history.round(4).to_string(index=False))
        print()
        for key in ["outer_mae_kzt", "outer_mape", "outer_r2"]:
            print(f"{key}: mean={history[key].mean():.4f}, std={history[key].std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
