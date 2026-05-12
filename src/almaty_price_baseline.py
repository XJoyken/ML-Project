#!/usr/bin/env python3

from __future__ import annotations

from ml_project.cli import main as cli_main
from ml_project.train import train_model


def run_pipeline(
    model_name: str = "lightgbm",
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    result = train_model(
        model_name,
        test_size=test_size,
        random_state=random_state,
        save_processed=True,
    )
    return result["processed_frame"], result["metrics_frame"], result["model"]


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
