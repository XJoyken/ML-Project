# ML-Project

End-to-end ML system for Almaty apartments: predicts sale price (Model 1),
recommends listings from a natural-language prompt (Model 2), and evaluates
rental investments — rent prediction + ROI/yield/payback/NPV (Model 3).

## Structure

Sale-price pipeline (Model 1):
- `src/ml_project/data.py` — listing loader + normalisation.
- `src/ml_project/poi.py` / `air.py` / `crime.py` — POI / PM2.5 / district crime catalogs.
- `src/ml_project/features.py` — feature schema + processed dataset builder.
- `src/ml_project/models.py` — CatBoost / LightGBM / XGBoost classes.
- `src/ml_project/train.py` / `tune.py` / `predict.py` — train / hyperparam search / inference.
- `src/ml_project/calibration.py` — split-conformal price intervals.
- `src/ml_project/evaluation.py` — `ApartmentEvaluationService` for single-listing inference.
- `src/ml_project/narrative.py` — Gemini-based human report + deterministic fallback.

Recommender (Model 2):
- `src/ml_project/recommender/scoring.py` — per-preference scoring (at_most / at_least / equal / prefer_low / prefer_high).
- `src/ml_project/recommender/mmr.py` — MMR diversification.
- `src/ml_project/recommender/service.py` — `RecommendationService` with hard filters + MMR.
- `src/ml_project/recommender/narrative.py` — recommender LLM report.

Rent + investment (Model 3):
- `src/ml_project/rent/data.py` — rent-ad loader + `furnished` categorical.
- `src/ml_project/rent/features.py` — rent feature schema (shares POI/air/crime with sale).
- `src/ml_project/rent/train.py` / `tune.py` / `predict.py` — mirror Model 1 for rent.
- `src/ml_project/rent/evaluation.py` — `RentEvaluationService` with calibration.
- `src/ml_project/rent/investment.py` — pure-math: gross/net yield, payback, NPV, IRR, verdict.
- `src/ml_project/rent/narrative.py` — investment LLM report with deterministic fallback.

Shared:
- `src/ml_project/macro.py` — inflation + Almaty real-estate yearly aggregates.
- `src/ml_project/tracking.py` — MLflow run/artifact helpers.
- `src/backend/main.py` — FastAPI with `/apartments/evaluate`, `/apartments/investment`, `/recommendations`, `/recommendations/features`.

## Setup

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cp .env.example .env  # then add your GEMINI_API_KEY
```

## CLI

The CLI lives at `src/almaty_price_baseline.py`. Subcommands:

**Sale-price (Model 1):**
```bash
src/almaty_price_baseline.py train     --model lightgbm --final
src/almaty_price_baseline.py evaluate  --model lightgbm
src/almaty_price_baseline.py predict   --model lightgbm --listing-id 1009196093
src/almaty_price_baseline.py tune      --model lightgbm --mode random --trials 30
```

Train logs to MLflow experiment `almaty-apartment-prices`.

**Rent (Model 3):**
```bash
src/almaty_price_baseline.py train-rent     --model lightgbm --final
src/almaty_price_baseline.py evaluate-rent  --model lightgbm
src/almaty_price_baseline.py tune-rent      --model lightgbm --mode random --trials 30
```

Train logs to MLflow experiment `almaty-apartment-rent`. Available models:
`lightgbm` (default), `catboost`, `xgboost`. After training, copy the latest
MLflow artifact into `models/apartment_rent_model/` to make it available to
the API service. See the project history for the helper snippet.

**MLflow UI** (browse both experiments):
```bash
mlflow ui
```

## API

```powershell
$env:PYTHONPATH = "src"
uvicorn backend.main:app --reload --port 8000
```

Swagger UI at `http://localhost:8000/docs`. Three endpoints:

- `POST /apartments/evaluate` — sale-price verdict from a krisha.kz URL.
  Body: `{ "url": "...", "language": "ru" | "en", "use_llm": true | false | null }`.

- `POST /apartments/investment` — full investment analysis from a krisha.kz URL.
  Predicts sale price (Model 1) + monthly rent (Model 3), computes yield/payback/NPV/IRR,
  returns a structured narrative with verdict.
  Body:
  ```json
  {
    "url": "https://krisha.kz/a/show/...",
    "language": "ru",
    "use_llm": null,
    "investment_params": {
      "vacancy_rate": null,
      "repair_cost_pct": null,
      "agent_commission_months": null,
      "tenant_turnover_years": null,
      "maintenance_pct": null,
      "property_tax_pct": null,
      "discount_rate_pct": null,
      "horizon_years": null
    }
  }
  ```
  Any `investment_params` field set to `null` uses the default. Defaults:
  vacancy 8%, repair 5% of sale price, agent 0.5 month per turnover, tenant turnover
  1.5 years, maintenance 5% of rent, property tax 0.3% of sale price/year,
  discount rate = latest inflation + 3 pp risk premium, horizon 10 years.

- `POST /recommendations` — Gemini-extracted preferences → ranked list of listings
  with per-item explanation and overall summary.

## Tests

```bash
python -m pytest src/tests -v
```

89+ tests cover: krisha parser, narrative generation, recommender scoring/MMR/service,
rent data/features, investment math, and FastAPI endpoint behaviour.

## Available pricing models

- `lightgbm` (default, served by the API for both sale and rent)
- `catboost`
- `xgboost`
