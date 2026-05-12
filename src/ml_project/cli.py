from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from .constants import MLFLOW_EXPERIMENT_NAME
from .predict import predict_price
from .train import evaluate_model, train_model
from .tune import tune_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Almaty apartment pricing pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    model_choices = ["catboost", "lightgbm", "xgboost"]

    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument("--model", default="lightgbm", choices=model_choices)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--experiment-name", default=MLFLOW_EXPERIMENT_NAME)
    train_parser.add_argument(
        "--params",
        help="Model hyperparameters as JSON (string or path to .json file).",
    )
    train_parser.add_argument(
        "--final",
        action="store_true",
        help="Two-pass: ES finds best_iter, then refit on full train without ES.",
    )

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a model.")
    evaluate_parser.add_argument("--model", default="lightgbm", choices=model_choices)
    evaluate_parser.add_argument("--test-size", type=float, default=0.2)
    evaluate_parser.add_argument("--random-state", type=int, default=42)
    evaluate_parser.add_argument(
        "--params",
        help="Model hyperparameters as JSON (string or path to .json file).",
    )
    evaluate_parser.add_argument(
        "--final",
        action="store_true",
        help="Two-pass: ES finds best_iter, then refit on full train without ES.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict fair price and explanation.")
    predict_parser.add_argument("--model", default="lightgbm", choices=model_choices)
    predict_parser.add_argument(
        "--params",
        help="Model hyperparameters as JSON (string or path to .json file).",
    )
    predict_parser.add_argument(
        "--from-mlflow",
        action="store_true",
        help="Skip retraining; load latest matching model from MLflow.",
    )
    predict_parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment to load from when --from-mlflow is set.",
    )
    source_group = predict_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--listing-id")
    source_group.add_argument("--input-json")
    source_group.add_argument("--input-csv")

    tune_parser = subparsers.add_parser("tune", help="Hyperparameter search (random or nested-CV).")
    tune_parser.add_argument("--model", required=True, choices=model_choices)
    tune_parser.add_argument("--mode", default="random", choices=["random", "nested"])
    tune_parser.add_argument("--trials", type=int, default=15)
    tune_parser.add_argument("--outer-splits", type=int, default=5)
    tune_parser.add_argument("--inner-splits", type=int, default=3)
    tune_parser.add_argument("--seed", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {None, "train"}:
        experiment_name = args.experiment_name if args.command else MLFLOW_EXPERIMENT_NAME
        model_params = _load_model_params(getattr(args, "params", None)) if args.command else None
        result = train_model(
            args.model if args.command else "lightgbm",
            test_size=args.test_size if args.command else 0.2,
            random_state=args.random_state if args.command else 42,
            experiment_name=experiment_name,
            model_params=model_params,
            final=getattr(args, "final", False),
        )
        print(f"Processed rows: {len(result['processed_frame']):,}")
        print(f"MLflow experiment: {experiment_name}")
        print(result["metrics_frame"].round(4).to_string(index=False))
        return 0

    if args.command == "evaluate":
        result = evaluate_model(
            args.model,
            test_size=args.test_size,
            random_state=args.random_state,
            model_params=_load_model_params(args.params),
            final=args.final,
        )
        print(result["metrics_frame"].round(4).to_string(index=False))
        return 0

    if args.command == "tune":
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

    if args.command == "predict":
        model_params = _load_model_params(args.params)
        common_kwargs = {
            "model_name": args.model,
            "model_params": model_params,
            "load_from_mlflow": args.from_mlflow,
            "experiment_name": args.experiment_name,
        }
        if args.listing_id is not None:
            result = predict_price(listing_id=args.listing_id, **common_kwargs)
        else:
            payload = _load_payload(args.input_json or args.input_csv)
            if args.input_csv is not None:
                payload = _parse_csv_payload(payload)
            else:
                payload = json.loads(payload)
            result = predict_price(listing=payload, **common_kwargs)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.error("Unknown command.")
    return 2


def _load_payload(value: str) -> str:
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


def _load_model_params(value: str | None) -> dict | None:
    if not value:
        return None
    return json.loads(_load_payload(value))


def _parse_csv_payload(payload: str) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(payload))
    return next(reader)
