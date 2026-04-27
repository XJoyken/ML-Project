from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

from .predict import predict_price
from .train import evaluate_model, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Almaty apartment pricing pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument("--model", default="catboost")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a model.")
    evaluate_parser.add_argument("--model", default="catboost")
    evaluate_parser.add_argument("--test-size", type=float, default=0.2)
    evaluate_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict fair price and explanation.")
    predict_parser.add_argument("--model", default="catboost")
    source_group = predict_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--listing-id")
    source_group.add_argument("--input-json")
    source_group.add_argument("--input-csv")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {None, "train"}:
        result = train_model(
            args.model if args.command else "catboost",
            test_size=args.test_size if args.command else 0.2,
            random_state=args.random_state if args.command else 42,
        )
        print(f"Processed rows: {len(result['processed_frame']):,}")
        print(result["metrics_frame"].round(4).to_string(index=False))
        return 0

    if args.command == "evaluate":
        result = evaluate_model(
            args.model,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(result["metrics_frame"].round(4).to_string(index=False))
        return 0

    if args.command == "predict":
        if args.listing_id is not None:
            result = predict_price(model_name=args.model, listing_id=args.listing_id)
        else:
            payload = _load_payload(args.input_json or args.input_csv)
            if args.input_csv is not None:
                payload = _parse_csv_payload(payload)
            else:
                payload = json.loads(payload)
            result = predict_price(model_name=args.model, listing=payload)

        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.error("Unknown command.")
    return 2


def _load_payload(value: str) -> str:
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return value


def _parse_csv_payload(payload: str) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(payload))
    return next(reader)
