# Metrics

Internal evaluation snapshots — not exposed to the user. Re-generate any of
these files by running the matching CLI command (see below).

## Files

| File | What it holds | How to regenerate |
|------|---------------|--------------------|
| `sale_model_nested_cv.json`       | Sale-price LightGBM nested 5×3 CV: MAE, MAPE, RMSE, R² (mean ± std) plus best hyperparameters | `python src/almaty_price_baseline.py tune       --model lightgbm --mode nested` |
| `rent_model_nested_cv.json`       | Rent LightGBM nested 5×3 CV (same shape as above)                                            | `python src/almaty_price_baseline.py tune-rent  --model lightgbm --mode nested` |
| `eval_queries.json`               | 25 hand-curated `prompt → relevance-rule` pairs feeding Precision@K. Hand-edit to extend.    | edit manually |
| `recommender_precision.json`      | Offline Precision@K of the recommender at K=5 and K=10 (mean / median / hit-rate + per-query breakdown) | `python src/almaty_price_baseline.py eval-recommender` |
| `investment_roi.json`             | Distribution of gross/net yield, payback (nominal + real), NPV, IRR across every listing in the processed feature frame + risk-premium sensitivity sweep | `python src/almaty_price_baseline.py eval-investment` |

## Model 1 (sale price) — current best

Nested 5×3 cross-validation, LightGBM with the hyperparameters baked into
`src/configs/lightgbm_best.json`. (or `src/configs/lightgbm_nested_best.json`)

| Metric | Mean        | Std         |
|--------|-------------|-------------|
| MAE    | 4,859,485 ₸ | 96,922 ₸    |
| MAPE   | 7.42%       | 0.015%      |
| RMSE   | 11,938,614 ₸| 1,131,041 ₸ |
| R²     | 0.9470      | 0.0093      |

## Model 3 (rent price) — current best

Nested 5×5 cross-validation, LightGBM.

| Metric | Mean      | Std       |
|--------|-----------|-----------|
| MAE    | 48,354 ₸  | 2,011 ₸   |
| MAPE   | 12.26%    | 0.29%     |
| RMSE   | 101,750 ₸ | 8,166 ₸   |
| R²     | 0.8664    | 0.0182    |

## Recommender — Precision@K

Method: for each hand-written prompt in `eval_queries.json` we run the full
recommender stack (Gemini feature extraction → hard-filter → scoring → MMR)
with `limit=K` and `mmr_lambda=0.7`. A returned listing counts as **relevant**
if it mechanically satisfies the explicit `relevance` rule (rooms / price cap /
POI distance cap / district / floor flags). The rule is applied directly to
the processed feature frame, so the metric is fully reproducible offline.

Aggregates we report:

- `mean_precision`   — mean of (relevant in top-K / K) across queries.
- `median_precision` — robust to one or two failing queries.
- `hit_rate`         — share of queries with ≥1 relevant in top-K (this is
  more forgiving and roughly tracks user satisfaction).
- `per_query`        — for every query: precision, hit, returned listing
  count, number of listings in the entire index that satisfy the rule
  (`candidates_passing_rule`) — useful to spot queries where the recommender
  is starving on a very tight rule.

Errors during Gemini extraction (e.g. free-tier 429 quota) are recorded as
`unmapped_preferences: ["__error__: …"]` with precision=0; they're included
in the aggregate denominator. Re-run later if the free-tier daily quota
reset matters.

## Investment ROI — distribution snapshot

`investment_roi.json` scores every listing in the processed feature frame:

1. Sale price = `target_price_kzt` from the listing.
2. Monthly rent = LightGBM rent-model prediction (the same one served by
   the API).
3. `compute_investment_metrics` runs with the production defaults
   (`InvestmentParams()` — vacancy 8%, repair 5%, agent 0.5 mo, turnover
   1.5 yr, maintenance 5%, property tax 0.3%, inflation 12.3%, risk-premium
   3 pp, horizon 10 yr).

For every metric (`gross_yield_pct`, `net_yield_pct`, `payback_years_nominal`,
`payback_years_inflation_adjusted`, `npv_kzt`, `irr_pct`) we save mean,
median, std, min, max, and the 5/25/50/75/95th percentiles.

We also save `verdict_counts` and `verdict_shares` — what fraction of the
dataset falls into each verdict tier under the base scenario.

`risk_premium_sensitivity` re-scores the dataset with `risk_premium_pct ∈
{0, 1.5, 3, 4.5, 6}` (inflation fixed at 12.3%). This shows how the verdict
mix shifts when the discount rate moves — the main intuition pump for the
"why is everything 'poor'?" question.
