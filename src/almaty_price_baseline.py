from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6_371.0088

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DATASET_PATH = PROCESSED_DIR / "ads_model_v1.csv"
METRICS_PATH = PROCESSED_DIR / "baseline_metrics_v1.json"

POI_SOURCES = {
    "metro": DATA_DIR / "almaty_metro.csv",
    "transport": DATA_DIR / "almaty_transport.csv",
    "schools": DATA_DIR / "almaty_schools.csv",
    "kindergartens": DATA_DIR / "almaty_kindergartens.csv",
    "universities": DATA_DIR / "almaty_university_filtered.csv",
    "clinics": DATA_DIR / "almaty_clinics.csv",
    "dentistry": DATA_DIR / "almaty_dentistry.csv",
    "hospitals": DATA_DIR / "almaty_hospitals.csv",
    "polyclinics": DATA_DIR / "almaty_polyclinics.csv",
    "medcenters": DATA_DIR / "almaty_medcenters.csv",
    "energy": DATA_DIR / "almaty_energy.csv",
}

ADS_COLUMNS = [
    "listing_id",
    "listing_category_alias",
    "listing_price_kzt",
    "listing_square_m2",
    "listing_rooms",
    "location_lat",
    "location_lon",
    "address_district",
    "address_microdistrict",
    "listing_complex_id",
    "summary_Год_постройки",
    "summary_Этаж",
    "summary_Тип_дома",
    "summary_Состояние_квартиры",
    "summary_Высота_потолков",
    "summary_Санузел",
    "listing_has_photo",
    "listing_photo_count",
    "scraped_at",
    "listing_added_at",
]

CATEGORICAL_FEATURES = [
    "district",
    "house_type",
    "condition",
    "bathroom_type",
]

BASE_NUMERIC_FEATURES = [
    "area_m2",
    "rooms",
    "lat",
    "lon",
    "year_built",
    "floor_current",
    "floors_total",
    "ceiling_height_m",
    "photo_count",
    "has_photo",
    "has_complex_id",
    "has_microdistrict",
    "complex_listing_count",
    "days_since_added_to_scrape",
]

TARGET_COLUMN = "target_price_kzt"


def extract_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", ".", regex=False)
        .str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_floor_columns(series: pd.Series) -> pd.DataFrame:
    parts = series.astype("string").str.extract(
        r"(?P<floor_current>\d+)\s*из\s*(?P<floors_total>\d+)"
    )
    return parts.apply(pd.to_numeric, errors="coerce")


def binary_from_text(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "да": 1.0,
        "нет": 0.0,
    }
    return normalized.map(mapping)


def load_ads_dataframe() -> pd.DataFrame:
    ads = pd.read_csv(DATA_DIR / "ads.csv", usecols=ADS_COLUMNS, low_memory=False)
    ads = ads.loc[ads["listing_category_alias"].eq("kvartiry")].copy()

    ads = ads.rename(
        columns={
            "listing_id": "listing_id",
            "listing_price_kzt": TARGET_COLUMN,
            "listing_square_m2": "area_m2",
            "listing_rooms": "rooms",
            "location_lat": "lat",
            "location_lon": "lon",
            "address_district": "district",
            "address_microdistrict": "microdistrict",
            "listing_complex_id": "complex_id",
            "summary_Год_постройки": "year_built",
            "summary_Этаж": "floor_text",
            "summary_Тип_дома": "house_type",
            "summary_Состояние_квартиры": "condition",
            "summary_Высота_потолков": "ceiling_height_text",
            "summary_Санузел": "bathroom_type",
            "listing_has_photo": "has_photo",
            "listing_photo_count": "photo_count",
        }
    )

    numeric_columns = [TARGET_COLUMN, "area_m2", "rooms", "lat", "lon", "photo_count"]
    for column in numeric_columns:
        ads[column] = pd.to_numeric(ads[column], errors="coerce")

    ads["year_built"] = pd.to_numeric(ads["year_built"], errors="coerce")
    ads["ceiling_height_m"] = extract_numeric(ads["ceiling_height_text"])
    ads["has_photo"] = binary_from_text(ads["has_photo"]).fillna(0.0)

    floor_parts = parse_floor_columns(ads["floor_text"])
    ads[["floor_current", "floors_total"]] = floor_parts

    ads["scraped_at"] = (
        pd.to_datetime(ads["scraped_at"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    ads["listing_added_at"] = (
        pd.to_datetime(ads["listing_added_at"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    ads["days_since_added_to_scrape"] = (
        ads["scraped_at"] - ads["listing_added_at"]
    ).dt.days

    ads["has_complex_id"] = ads["complex_id"].notna().astype(np.int8)
    ads["has_microdistrict"] = ads["microdistrict"].notna().astype(np.int8)

    complex_listing_count = ads["complex_id"].value_counts(dropna=True)
    ads["complex_listing_count"] = (
        ads["complex_id"].map(complex_listing_count).fillna(0).astype(np.int32)
    )

    for column in CATEGORICAL_FEATURES:
        ads[column] = ads[column].fillna("unknown").astype("string")

    ads = ads.loc[
        ads[TARGET_COLUMN].gt(0)
        & ads["area_m2"].gt(0)
        & ads["rooms"].notna()
        & ads["lat"].between(43.0, 44.0)
        & ads["lon"].between(76.0, 77.2)
    ].copy()

    ads["target_log_price"] = np.log1p(ads[TARGET_COLUMN])
    return ads.reset_index(drop=True)


def load_poi_coordinates(path: Path) -> np.ndarray:
    poi = pd.read_csv(path, usecols=["lat", "lon"])
    poi["lat"] = pd.to_numeric(poi["lat"], errors="coerce")
    poi["lon"] = pd.to_numeric(poi["lon"], errors="coerce")
    poi = poi.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["lat", "lon"])
    return np.radians(poi[["lat", "lon"]].to_numpy())


def build_poi_features(ads: pd.DataFrame) -> pd.DataFrame:
    ads_coordinates = np.radians(ads[["lat", "lon"]].to_numpy())
    poi_features = pd.DataFrame(index=ads.index)

    for source_name, path in POI_SOURCES.items():
        poi_coordinates = load_poi_coordinates(path)
        if len(poi_coordinates) == 0:
            continue

        tree = BallTree(poi_coordinates, metric="haversine")
        neighbors_count = min(5, len(poi_coordinates))
        distances, _ = tree.query(ads_coordinates, k=neighbors_count)
        distances_km = distances * EARTH_RADIUS_KM

        poi_features[f"{source_name}_dist_km_1"] = distances_km[:, 0].astype(np.float32)
        poi_features[f"{source_name}_mean_dist_km_3"] = (
            distances_km[:, : min(3, neighbors_count)].mean(axis=1).astype(np.float32)
        )
        poi_features[f"{source_name}_mean_dist_km_5"] = (
            distances_km.mean(axis=1).astype(np.float32)
        )
        poi_features[f"{source_name}_count_500m"] = tree.query_radius(
            ads_coordinates,
            r=0.5 / EARTH_RADIUS_KM,
            count_only=True,
        ).astype(np.int16)
        poi_features[f"{source_name}_count_1000m"] = tree.query_radius(
            ads_coordinates,
            r=1.0 / EARTH_RADIUS_KM,
            count_only=True,
        ).astype(np.int16)

    return poi_features


def build_processed_dataset(save: bool = True) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ads = load_ads_dataframe()
    poi_features = build_poi_features(ads)
    processed = pd.concat([ads, poi_features], axis=1)

    poi_feature_columns = sorted(poi_features.columns.tolist())
    selected_columns = [
        "listing_id",
        TARGET_COLUMN,
        "target_log_price",
        "scraped_at",
        "listing_added_at",
        *BASE_NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        *poi_feature_columns,
    ]
    processed = processed[selected_columns].copy()

    if save:
        processed.to_csv(PROCESSED_DATASET_PATH, index=False)

    return processed


def get_feature_columns(processed: pd.DataFrame) -> tuple[list[str], list[str]]:
    poi_feature_columns = sorted(
        [
            column
            for column in processed.columns
            if any(column.startswith(f"{name}_") for name in POI_SOURCES)
        ]
    )
    numeric_features = BASE_NUMERIC_FEATURES + poi_feature_columns
    return numeric_features, CATEGORICAL_FEATURES


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.clip(np.asarray(y_pred, dtype=float), a_min=0, a_max=None)
    rmse = float(np.sqrt(np.mean((y_true_array - y_pred_array) ** 2)))
    rmse_log = float(
        np.sqrt(np.mean((np.log1p(y_true_array) - np.log1p(y_pred_array)) ** 2))
    )
    return {
        "mae_kzt": float(mean_absolute_error(y_true_array, y_pred_array)),
        "mape": float(mean_absolute_percentage_error(y_true_array, y_pred_array)),
        "rmse_kzt": rmse,
        "rmse_log": rmse_log,
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }


def predict_group_median_price(
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
) -> np.ndarray:
    train_reference = train_frame.copy()
    train_reference["price_per_m2"] = (
        train_reference[TARGET_COLUMN] / train_reference["area_m2"]
    )

    group_columns = ["district", "rooms"]
    group_median = (
        train_reference.groupby(group_columns)["price_per_m2"].median().to_dict()
    )
    district_median = train_reference.groupby("district")["price_per_m2"].median().to_dict()
    rooms_median = train_reference.groupby("rooms")["price_per_m2"].median().to_dict()
    global_median = float(train_reference["price_per_m2"].median())

    predicted_price_per_m2 = []
    for _, row in valid_frame.iterrows():
        key = (row["district"], row["rooms"])
        price_per_m2 = group_median.get(key)
        if price_per_m2 is None:
            price_per_m2 = district_median.get(row["district"])
        if price_per_m2 is None:
            price_per_m2 = rooms_median.get(row["rooms"])
        if price_per_m2 is None:
            price_per_m2 = global_median
        predicted_price_per_m2.append(price_per_m2)

    return np.asarray(predicted_price_per_m2) * valid_frame["area_m2"].to_numpy()


def train_baseline_models(
    processed: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, CatBoostRegressor]:
    numeric_features, categorical_features = get_feature_columns(processed)
    feature_columns = numeric_features + categorical_features

    train_frame, valid_frame = train_test_split(
        processed,
        test_size=test_size,
        random_state=random_state,
    )

    heuristic_predictions = predict_group_median_price(train_frame, valid_frame)
    train_target_log = np.log1p(train_frame[TARGET_COLUMN])
    valid_target_log = np.log1p(valid_frame[TARGET_COLUMN])

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5.0,
        min_data_in_leaf=30,
        random_seed=random_state,
        verbose=False,
    )

    model.fit(
        train_frame[feature_columns],
        train_target_log,
        cat_features=categorical_features,
        eval_set=(valid_frame[feature_columns], valid_target_log),
        use_best_model=True,
    )
    model_predictions = np.expm1(model.predict(valid_frame[feature_columns]))

    metrics = pd.DataFrame(
        [
            {"model": "district_room_median", **calculate_metrics(valid_frame[TARGET_COLUMN], heuristic_predictions)},
            {"model": "catboost_log_target", **calculate_metrics(valid_frame[TARGET_COLUMN], model_predictions)},
        ]
    )

    payload = {
        "rows_processed": int(len(processed)),
        "validation_rows": int(len(valid_frame)),
        "models": metrics.round(6).to_dict(orient="records"),
        "feature_columns": feature_columns,
    }
    METRICS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics, model


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, CatBoostRegressor]:
    processed = build_processed_dataset(save=True)
    metrics, model = train_baseline_models(processed)
    return processed, metrics, model


def main() -> None:
    processed, metrics, _ = run_pipeline()
    print(f"Processed rows: {len(processed):,}")
    print(f"Saved dataset: {PROCESSED_DATASET_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
