# ML-Project
Ml project is related to determining the pricing of apartments in Almaty and determining the profitability of ads on the krisha.kz, based on the area, floor, distances to key objects, and so on. It also includes a recommendation system and the flexible use of the ML model through the processing of requests using DL.

## Structure
- `src/ml_project/data.py`: loading and normalization of listings.
- `src/ml_project/poi.py`: loading POI csv files, spatial index, and POI distance/count features.
- `src/ml_project/features.py`: feature schema, base listing features, POI features, processed dataset builders.
- `src/ml_project/models.py`: model implementations and model factory.
- `src/ml_project/train.py`: train/evaluate functions and artifact saving.
- `src/ml_project/predict.py`: prediction, verdict, and human-readable explanation.
- `src/almaty_price_baseline.py`: thin wrapper and CLI entrypoint.
- `artifacts/<model_name>`: trained model files, schema, and metrics.

## CLI
Activate the project virtualenv first:

```bash
source .venv/bin/activate
```

Then run commands directly:

```bash
src/almaty_price_baseline.py train --model catboost
src/almaty_price_baseline.py evaluate --model catboost
src/almaty_price_baseline.py predict --model catboost --listing-id 1009196093
```

Predict on a custom listing with inline JSON:

```bash
src/almaty_price_baseline.py predict --model catboost --input-json '{"lat":43.2,"lon":76.9,"area_m2":50,"rooms":2,"district":"Бостандыкский район","house_type":"монолитный","condition":"хорошее","bathroom_type":"совмещенный","listing_price_kzt":4000000}'
```

Available models:
- `catboost`

`datasets/processed/ads_model_v1.csv` is kept as an optional debug dataset. Trained models are stored under `artifacts/`.
