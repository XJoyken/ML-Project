"""Investment ROI / yield / payback distributional analysis.

For every listing in the processed feature frame we:
  1) take the listed sale price (target_price_kzt),
  2) predict the monthly rent using the live LightGBM rent model,
  3) run `compute_investment_metrics` with the default parameters,
  4) collect gross/net yield, payback (nominal + inflation-adjusted), NPV, IRR,
     and the categorical verdict.

The output is a distribution snapshot: mean/median/std/percentiles for every
metric, plus the share of each verdict tier. We also rerun the whole batch
with a few `risk_premium_pct` overrides to see how the verdict distribution
shifts when the user dials the risk premium up or down.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _percentiles(values: list[float], qs: tuple[float, ...]) -> dict[str, float]:
    if not values:
        return {f"p{int(q*100)}": 0.0 for q in qs}
    arr = np.asarray(values, dtype=float)
    return {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in qs}


def _describe(values: list[float], unit: str = "") -> dict[str, Any]:
    """Compute summary stats for a numeric series, ignoring NaN/None."""
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return {"n": 0, "unit": unit}
    return {
        "n": len(clean),
        "unit": unit,
        "mean": round(statistics.fmean(clean), 4),
        "median": round(statistics.median(clean), 4),
        "std": round(statistics.pstdev(clean), 4) if len(clean) > 1 else 0.0,
        "min": round(min(clean), 4),
        "max": round(max(clean), 4),
        **{k: round(v, 4) for k, v in _percentiles(clean, (0.05, 0.25, 0.5, 0.75, 0.95)).items()},
    }


@dataclass(slots=True)
class ScenarioRun:
    inflation_rate_pct: float
    risk_premium_pct: float
    n_listings: int
    metrics: dict[str, Any]
    verdict_counts: dict[str, int]
    verdict_shares: dict[str, float]


def _run_one_scenario(
    *,
    sale_prices: np.ndarray,
    rents: np.ndarray,
    inflation_rate_pct: float,
    risk_premium_pct: float,
) -> ScenarioRun:
    from ml_project.macro import load_inflation_catalog, load_market_catalog
    from ml_project.rent.investment import InvestmentParams, compute_investment_metrics

    inflation = load_inflation_catalog()
    market = load_market_catalog()
    params = InvestmentParams(
        inflation_rate_pct=inflation_rate_pct,
        risk_premium_pct=risk_premium_pct,
    )

    gross, net, payback_n, payback_r, npv, irr = [], [], [], [], [], []
    verdicts: list[str] = []
    for sale, rent in zip(sale_prices, rents):
        if sale <= 0 or rent <= 0 or np.isnan(sale) or np.isnan(rent):
            continue
        try:
            res = compute_investment_metrics(
                sale_price_kzt=float(sale),
                monthly_rent_kzt=float(rent),
                params=params,
                inflation=inflation,
                market=market,
            )
        except ValueError:
            continue
        gross.append(res.gross_yield_pct)
        net.append(res.net_yield_pct)
        payback_n.append(res.payback_years_nominal)
        payback_r.append(res.payback_years_inflation_adjusted)
        npv.append(res.npv_kzt)
        irr.append(res.irr_pct)
        verdicts.append(res.verdict)

    counts = dict(Counter(verdicts))
    total = sum(counts.values())
    shares = {k: round(v / total, 4) if total else 0.0 for k, v in counts.items()}
    return ScenarioRun(
        inflation_rate_pct=inflation_rate_pct,
        risk_premium_pct=risk_premium_pct,
        n_listings=len(verdicts),
        metrics={
            "gross_yield_pct":                  _describe(gross,      unit="%"),
            "net_yield_pct":                    _describe(net,        unit="%"),
            "payback_years_nominal":            _describe([v for v in payback_n if v is not None], unit="years"),
            "payback_years_inflation_adjusted": _describe([v for v in payback_r if v is not None], unit="years"),
            "npv_kzt":                          _describe(npv,        unit="kzt"),
            "irr_pct":                          _describe([v for v in irr if v is not None],       unit="%"),
        },
        verdict_counts=counts,
        verdict_shares=shares,
    )


def evaluate_full_dataset(
    *,
    output_path: Path,
    risk_sensitivity: tuple[float, ...] = (0.0, 1.5, 3.0, 4.5, 6.0),
) -> dict[str, Any]:
    """Score every listing in the processed feature frame, plus a risk-premium sweep.

    Sale price is taken straight from `target_price_kzt`; monthly rent is the
    LightGBM prediction. Both numbers are in tenge.
    """
    from ml_project.constants import PROCESSED_DATASET_PATH
    from ml_project.rent.evaluation import RentEvaluationService

    features = pd.read_csv(PROCESSED_DATASET_PATH)
    sale_prices = features["target_price_kzt"].to_numpy(dtype=float)

    rent_service = RentEvaluationService()
    rent_model = rent_service.model
    rent_schema = rent_service.schema

    # The sale-side processed frame doesn't carry the rent-only `furnished` flag.
    # Default any missing schema column to "unknown" (categorical, valid training value)
    # or 0.0 (numeric). This keeps the prediction shape compatible.
    rent_input = features.copy()
    for col in rent_schema.categorical_features:
        if col not in rent_input.columns:
            rent_input[col] = "unknown"
    for col in rent_schema.numeric_features:
        if col not in rent_input.columns:
            rent_input[col] = 0.0

    rents = np.asarray(rent_model.predict(rent_input, rent_schema), dtype=float)

    base = _run_one_scenario(
        sale_prices=sale_prices,
        rents=rents,
        inflation_rate_pct=12.3,
        risk_premium_pct=3.0,
    )

    sensitivity = []
    for rp in risk_sensitivity:
        run = _run_one_scenario(
            sale_prices=sale_prices,
            rents=rents,
            inflation_rate_pct=12.3,
            risk_premium_pct=rp,
        )
        sensitivity.append({
            "inflation_rate_pct": run.inflation_rate_pct,
            "risk_premium_pct":   run.risk_premium_pct,
            "discount_rate_pct":  round(run.inflation_rate_pct + run.risk_premium_pct, 2),
            "n_listings":         run.n_listings,
            "verdict_counts":     run.verdict_counts,
            "verdict_shares":     run.verdict_shares,
            "median_real_payback_years": run.metrics["payback_years_inflation_adjusted"].get("median"),
            "median_gross_yield_pct":     run.metrics["gross_yield_pct"].get("median"),
            "median_net_yield_pct":       run.metrics["net_yield_pct"].get("median"),
        })

    payload = {
        "description":
            "Distribution of investment economics across every listing in the processed feature frame. "
            "Sale price = listing price as-is. Monthly rent = LightGBM rent-model prediction. "
            "Base scenario uses the production defaults (vacancy 8%, repair 5%, agent 0.5 mo, turnover "
            "1.5 yr, maintenance 5%, property tax 0.3%, inflation 12.3%, risk premium 3 pp, horizon 10 yr).",
        "base_scenario": {
            "params": {
                "inflation_rate_pct": base.inflation_rate_pct,
                "risk_premium_pct":   base.risk_premium_pct,
                "discount_rate_pct":  round(base.inflation_rate_pct + base.risk_premium_pct, 2),
            },
            "n_listings":     base.n_listings,
            "metrics":        base.metrics,
            "verdict_counts": base.verdict_counts,
            "verdict_shares": base.verdict_shares,
        },
        "risk_premium_sensitivity": sensitivity,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


__all__ = ["ScenarioRun", "evaluate_full_dataset"]
