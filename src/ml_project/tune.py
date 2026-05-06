from __future__ import annotations

import argparse
import random
from typing import Any

import mlflow
import pandas as pd

from .constants import MLFLOW_EXPERIMENT_NAME
from .train import evaluate_model

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "catboost": {
        "iterations": [800, 1200, 1800],
        "learning_rate": [0.03, 0.05, 0.08],
        "depth": [6, 8, 10],
        "l2_leaf_reg": [3.0, 5.0, 8.0],
        "min_data_in_leaf": [20, 30, 50],
    },
    "lightgbm": {
        "n_estimators": [800, 1500, 2500],
        "learning_rate": [0.03, 0.05, 0.08],
        "num_leaves": [63, 127, 255],
        "min_child_samples": [20, 30, 50],
        "reg_lambda": [1.0, 5.0, 10.0],
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


def tune_model(
    model_name: str,
    *,
    trials: int = 15,
    seed: int = 42,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME + "-tune",
) -> pd.DataFrame:
    grid = PARAM_GRIDS[model_name]
    rng = random.Random(seed)
    mlflow.set_experiment(experiment_name)

    history = []
    for trial in range(1, trials + 1):
        params = sample_params(grid, rng)
        with mlflow.start_run(run_name=f"{model_name}-trial-{trial}"):
            mlflow.log_params({"model": model_name, **params})
            result = evaluate_model(model_name, model_params=params)
            metrics = result["metrics_frame"].drop(columns=["model"]).iloc[0].to_dict()
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            history.append({"trial": trial, **params, **metrics})
            print(f"trial {trial}: mae={metrics['mae_kzt']:,.0f}, r2={metrics['r2']:.3f}")

    return pd.DataFrame(history).sort_values("mae_kzt").reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Random search over hyperparameters.")
    parser.add_argument("--model", required=True, choices=list(PARAM_GRIDS.keys()))
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    history = tune_model(args.model, trials=args.trials, seed=args.seed)
    print()
    print("Top 5 by MAE:")
    print(history.head(5).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
