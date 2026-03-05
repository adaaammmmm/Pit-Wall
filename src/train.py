"""
src/train.py
Trains:
  - Stage 1: XGBoost lap time regressor (full lap)
  - Stage 2: LightGBM sector regressors (S1, S2, S3 separately)
  - Stage 3: XGBoost pit stop strategy classifier (should_pit? yes/no)

Run: python src/train.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, f1_score, classification_report,
)
from sklearn.model_selection import KFold, cross_val_score, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.data_pipeline import load_and_split, build_preprocessor, get_feature_names

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


def reg_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # Convert to milliseconds for interpretability
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae*1000:.1f} ms  ({mae:.4f}s)")
    print(f"    RMSE : {rmse*1000:.1f} ms")
    print(f"    R²   : {r2:.4f}")
    return {"model": name, "mae_ms": round(mae*1000, 2),
            "rmse_ms": round(rmse*1000, 2), "r2": round(r2, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Lap Time Regressor
# ─────────────────────────────────────────────────────────────────────────────
def train_lap_time_model(X_train, X_test, y_train, y_test, feat_cols):
    section("STAGE 1 — LAP TIME REGRESSOR")

    prep = build_preprocessor()
    X_tr = prep.fit_transform(X_train)
    X_te = prep.transform(X_test)

    results = []

    # Baseline: Ridge
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr, y_train)
    results.append(reg_metrics("Ridge (baseline)", y_test, ridge.predict(X_te)))

    # Tune XGBoost with Optuna
    print("\n  [XGBoost] Tuning with Optuna (60 trials)...")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 4, 10),
            "learning_rate":    trial.suggest_float("lr", 0.01, 0.2, log=True),
            "subsample":        trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("col", 0.65, 1.0),
            "reg_alpha":        trial.suggest_float("alpha", 1e-4, 5.0, log=True),
            "reg_lambda":       trial.suggest_float("lambda", 1e-4, 5.0, log=True),
            "min_child_weight": trial.suggest_int("mcw", 1, 10),
            "random_state": 42,
        }
        model = xgb.XGBRegressor(**params)
        kf    = KFold(n_splits=5, shuffle=True, random_state=42)
        score = cross_val_score(model, X_tr, y_train, cv=kf,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
        return score.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60, show_progress_bar=False)

    best = study.best_params
    best["random_state"] = 42
    print(f"  Best CV MAE: {-study.best_value*1000:.1f} ms")

    xgb_model = xgb.XGBRegressor(**best)
    xgb_model.fit(X_tr, y_train,
                  eval_set=[(X_te, y_test)],
                  verbose=False)

    y_pred = xgb_model.predict(X_te)
    results.append(reg_metrics("XGBoost (Tuned)", y_test, y_pred))

    # Save
    joblib.dump(prep,       os.path.join(MODELS_DIR, "lap_preprocessor.pkl"))
    joblib.dump(xgb_model,  os.path.join(MODELS_DIR, "xgb_lap_model.pkl"))
    fnames = get_feature_names(prep)
    joblib.dump(fnames,     os.path.join(MODELS_DIR, "feature_names.pkl"))

    return xgb_model, prep, results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Sector Time Regressors
# ─────────────────────────────────────────────────────────────────────────────
def train_sector_models(X_train, X_test, splits):
    section("STAGE 2 — SECTOR TIME REGRESSORS (S1, S2, S3)")

    sector_results = {}
    for sector_key, label in [("sector1", "S1"), ("sector2", "S2"), ("sector3", "S3")]:
        _, _, y_tr, y_te = splits[sector_key]

        prep_s = build_preprocessor()
        X_tr_s = prep_s.fit_transform(X_train)
        X_te_s = prep_s.transform(X_test)

        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.06, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, random_state=42, verbose=-1,
        )
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        m = reg_metrics(f"LightGBM {label}", y_te, y_pred)
        sector_results[sector_key] = m

        joblib.dump(prep_s, os.path.join(MODELS_DIR, f"{sector_key}_preprocessor.pkl"))
        joblib.dump(model,  os.path.join(MODELS_DIR, f"lgb_{sector_key}_model.pkl"))

    # Summed sectors vs direct prediction
    print("\n  [Sector sum evaluation]")
    preds_sum = np.zeros(len(splits["main"][3]))
    for sector_key in ["sector1", "sector2", "sector3"]:
        model = joblib.load(os.path.join(MODELS_DIR, f"lgb_{sector_key}_model.pkl"))
        prep  = joblib.load(os.path.join(MODELS_DIR, f"{sector_key}_preprocessor.pkl"))
        X_te_s = prep.transform(X_test)
        preds_sum += model.predict(X_te_s)

    y_te_total = splits["main"][3]
    mae_sum = mean_absolute_error(y_te_total, preds_sum)
    print(f"    Summed sector MAE: {mae_sum*1000:.1f} ms")

    return sector_results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Pit Stop Classifier
# ─────────────────────────────────────────────────────────────────────────────
def train_pit_classifier(full_df):
    section("STAGE 3 — PIT STOP STRATEGY CLASSIFIER")

    df = full_df.copy()

    # Target: will the driver pit on the NEXT lap?
    df = df.sort_values(["season", "circuit", "driver", "lap_number"])
    df["pit_next_lap"] = df.groupby(["season", "circuit", "driver"])["is_pit_lap"].shift(-1).fillna(0).astype(int)

    # Only clean (non-pit, non-outlier) laps as training data
    df = df[(df["is_pit_lap"] == 0) & (df["is_outlier"] == 0)].copy()

    # Feature set for pit decision
    pit_features = [
        "tyre_life", "tyre_life_sq", "lap_pct", "fuel_remaining_laps",
        "compound_encoded", "track_temp", "is_raining",
        "safety_car_laps_ago", "circuit_deg_factor",
        "driver_rolling_form", "lap_delta_from_mean",
        "prev_lap_time",
    ]
    pit_features = [f for f in pit_features if f in df.columns]

    train = df[df["season"].isin([2022, 2023])]
    test  = df[df["season"] == 2024]

    X_tr = train[pit_features].fillna(0)
    y_tr = train["pit_next_lap"]
    X_te = test[pit_features].fillna(0)
    y_te = test["pit_next_lap"]

    scale_pos = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"  Class balance — pit: {y_tr.mean():.1%}, no pit: {(1-y_tr).mean():.1%}")
    print(f"  scale_pos_weight: {scale_pos:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42, eval_metric="auc",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    print(f"\n  ROC-AUC: {roc_auc_score(y_te, y_prob):.4f}")
    print(f"  F1:      {f1_score(y_te, y_pred):.4f}")
    print(f"  Accuracy:{accuracy_score(y_te, y_pred):.4f}")
    print(classification_report(y_te, y_pred, target_names=["No Pit", "Pit"]))

    joblib.dump(model,       os.path.join(MODELS_DIR, "xgb_pit_classifier.pkl"))
    joblib.dump(pit_features,os.path.join(MODELS_DIR, "pit_features.pkl"))

    return model, {
        "roc_auc":  round(roc_auc_score(y_te, y_prob), 4),
        "f1":       round(f1_score(y_te, y_pred), 4),
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n🏎️  F1 Lap Time Predictor — Model Training")

    splits = load_and_split()
    X_train, X_test, y_train, y_test = splits["main"]
    feat_cols = splits["feat_cols"]
    full_df   = splits["full_df"]

    lap_model, prep, lap_results   = train_lap_time_model(X_train, X_test, y_train, y_test, feat_cols)
    sector_results                  = train_sector_models(X_train, X_test, splits)
    pit_model, pit_results          = train_pit_classifier(full_df)

    summary = {
        "lap_time_results": lap_results,
        "sector_results":   sector_results,
        "pit_results":      pit_results,
    }
    with open(os.path.join(MODELS_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    section("TRAINING COMPLETE ✅")
    best_lap = min(r["mae_ms"] for r in lap_results if "mae_ms" in r)
    print(f"  Best lap MAE:    {best_lap:.0f} ms")
    print(f"  Pit classifier:  ROC-AUC {pit_results['roc_auc']}")
    print(f"  Models saved  →  {MODELS_DIR}/")


if __name__ == "__main__":
    main()
