"""
src/data_pipeline.py
Feature engineering, preprocessing pipelines, train/test splits.
Temporal split: train on 2022-2023, test on 2024.
"""

import os, sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

RAW_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "f1_laps.csv")
PROC_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

COMPOUND_ORDER = [["WET", "INTER", "HARD", "MEDIUM", "SOFT"]]

NUMERIC_FEATURES = [
    "tyre_life",
    "fuel_remaining_laps",
    "fuel_laps_completed",
    "track_temp",
    "air_temp",
    "humidity",
    "is_raining",
    "safety_car_laps_ago",
    "circuit_deg_factor",
    "lap_number",
    "total_laps",
    "prev_lap_time",
    "tyre_life_sq",
    "temp_tyre_interaction",
    "lap_pct",
    "is_high_deg_circuit",
    "compound_encoded",
]

CATEGORICAL_FEATURES = [
    "circuit",
    "team",
    "driver",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET       = "lap_time_s"
SECTOR_TARGETS = ["sector1_s", "sector2_s", "sector3_s"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Remove pit stop laps and formation laps for clean lap time modelling
    df = df[df["is_pit_lap"] == 0].copy()
    df = df[df["is_outlier"] == 0].copy()
    df = df[df["prev_lap_time"].notna()].copy()

    # Polynomial tyre life (captures non-linear degradation)
    df["tyre_life_sq"] = df["tyre_life"] ** 2

    # Track temp × tyre interaction (hot track + soft = bad)
    compound_heat_sensitivity = {"SOFT": 1.5, "MEDIUM": 1.0, "HARD": 0.6, "INTER": 0.4, "WET": 0.2}
    df["temp_tyre_interaction"] = df["track_temp"] * df["compound"].map(compound_heat_sensitivity)

    # Lap percentage through race (captures fuel/strategic context)
    df["lap_pct"] = df["lap_number"] / df["total_laps"]

    # High degradation circuit flag
    df["is_high_deg_circuit"] = (df["circuit_deg_factor"] > 1.2).astype(int)

    # Ordinal compound encoding (WET=0 ... SOFT=4)
    compound_map = {"WET": 0, "INTER": 1, "HARD": 2, "MEDIUM": 3, "SOFT": 4}
    df["compound_encoded"] = df["compound"].map(compound_map)

    # Rolling driver form (last 5 laps avg delta vs circuit mean)
    df = df.sort_values(["season", "circuit", "driver", "lap_number"])
    circuit_mean = df.groupby(["season", "circuit"])["lap_time_s"].transform("mean")
    df["lap_delta_from_mean"] = df["lap_time_s"] - circuit_mean
    df["driver_rolling_form"] = (
        df.groupby(["season", "circuit", "driver"])["lap_delta_from_mean"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Add to numeric features if not there
    if "driver_rolling_form" not in NUMERIC_FEATURES:
        NUMERIC_FEATURES.append("driver_rolling_form")

    return df.reset_index(drop=True)


def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    use_num = [f for f in NUMERIC_FEATURES if f != "compound_encoded" or True]

    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")
    return preprocessor


def get_feature_names(preprocessor):
    cat_enc  = preprocessor.named_transformers_["cat"]["encoder"]
    cat_names= list(cat_enc.feature_names_in_)
    return NUMERIC_FEATURES + cat_names


def load_and_split():
    df  = load_raw()
    df  = engineer_features(df)

    # Temporal split — train 2022-2023, test 2024
    train_df = df[df["season"].isin([2022, 2023])].copy()
    test_df  = df[df["season"] == 2024].copy()

    cols = ALL_FEATURES + (["driver_rolling_form"] if "driver_rolling_form" in df.columns
                           and "driver_rolling_form" not in ALL_FEATURES else [])

    # Safely pick only cols that exist
    feat_cols = [c for c in ALL_FEATURES + ["driver_rolling_form"] if c in df.columns]
    feat_cols = list(dict.fromkeys(feat_cols))  # dedupe

    X_train = train_df[feat_cols]
    X_test  = test_df[feat_cols]
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]

    # Sector targets
    y_s1_train, y_s1_test = train_df["sector1_s"], test_df["sector1_s"]
    y_s2_train, y_s2_test = train_df["sector2_s"], test_df["sector2_s"]
    y_s3_train, y_s3_test = train_df["sector3_s"], test_df["sector3_s"]

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Train seasons: {train_df['season'].unique()}  |  Test season: {test_df['season'].unique()}")

    return {
        "main":    (X_train, X_test, y_train, y_test),
        "sector1": (X_train, X_test, y_s1_train, y_s1_test),
        "sector2": (X_train, X_test, y_s2_train, y_s2_test),
        "sector3": (X_train, X_test, y_s3_train, y_s3_test),
        "full_df": df,
        "train_df": train_df,
        "test_df":  test_df,
        "feat_cols": feat_cols,
    }


if __name__ == "__main__":
    splits = load_and_split()
    X_train, X_test, y_train, y_test = splits["main"]
    print(f"Target mean: {y_train.mean():.2f}s  std: {y_train.std():.2f}s")
