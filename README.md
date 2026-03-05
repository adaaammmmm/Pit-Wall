# 🏎️ F1 Lap Time Predictor

An end-to-end machine learning project predicting Formula 1 lap times using a three-stage model architecture — lap time regression, sector decomposition, and pit stop strategy classification — with full SHAP explainability and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)

---

## 📁 Project Structure

```
f1_laptime_predictor/
├── data/
│   ├── raw/                      # Generated lap data (100k+ laps)
│   └── generate_data.py          # Synthetic F1 data generator
├── notebooks/
│   ├── 01_eda.ipynb              # Tyre deg, weather, team pace EDA
│   └── 02_model_analysis.ipynb  # Model eval, SHAP, residual analysis
├── src/
│   ├── data_pipeline.py          # Feature engineering + sklearn pipelines
│   ├── train.py                  # All 3 model stages
│   └── explain.py                # SHAP plots
├── app/
│   └── streamlit_app.py          # 3-tab dashboard
├── models/                       # Saved models (after training)
├── outputs/                      # Generated plots
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate data (~4s)
python data/generate_data.py

# 3. Train all models (~3-4 min)
python src/train.py

# 4. Generate SHAP plots (~1 min)
python src/explain.py

# 5. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## 🏗️ Model Architecture

### Stage 1 — Lap Time Regressor (XGBoost)
- Predicts full lap time in seconds
- Tuned with Optuna (60 trials, KFold CV)
- Target MAE: **< 200ms**

### Stage 2 — Sector Regressors (LightGBM × 3)
- Separate models for Sector 1, 2, and 3
- Reveals *where* on track conditions matter most
- Sector sum is compared to direct prediction

### Stage 3 — Pit Stop Classifier (XGBoost)
- Binary: should this driver pit on the next lap?
- Class imbalance handled via `scale_pos_weight`
- Target metric: ROC-AUC (majority of laps are non-pit)

---

## 🔧 Feature Engineering

| Feature | Description | Why it matters |
|---|---|---|
| `tyre_life` + `tyre_life²` | Laps on current set | Non-linear deg curve |
| `compound_encoded` | WET=0 → SOFT=4 | Ordinal compound speed |
| `temp_tyre_interaction` | track_temp × compound sensitivity | Hot track punishes SOFT more |
| `circuit_deg_factor` | Circuit-specific multiplier | Qatar deg is 2× Monaco |
| `fuel_remaining_laps` | Fuel load proxy | ~0.032s/lap weight effect |
| `driver_rolling_form` | Rolling delta vs circuit avg | Recent form captures car setup |
| `safety_car_laps_ago` | Laps since SC restart | Cold tyre penalty (decays over 5 laps) |
| `lap_pct` | Lap / total laps | Strategic context |

---

## 📊 Key Findings

### Tyre Degradation
- **SOFT**: ~0.06s/lap degradation × circuit factor — cliff at ~lap 18-22
- **MEDIUM**: ~0.035s/lap — linear and predictable
- **HARD**: ~0.018s/lap — minimal, but 0.8s slower baseline

### Weather
- Wet conditions add **8-20s** on slick tyres (wrong compound penalty)
- Track temp above 48°C increases SOFT degradation by ~40%

### Circuit Characteristics
- **Qatar** highest degradation (factor 1.6) — explains the multiple strategic interventions
- **Monaco** lowest (factor 0.6) — track position over tyre strategy always wins

### Driver Analysis
- SHAP residual analysis reveals drivers consistently faster/slower than the car's pace model predicts — a proxy for pure driver skill

---

## 🖥️ Dashboard

| Tab | Features |
|---|---|
| 🎯 Lap Predictor | Input any race condition → predicted lap time + sector breakdown + SHAP breakdown |
| 🔀 Strategy Simulator | Compare two full race strategies lap-by-lap |
| 📊 Circuit Explorer | Per-circuit tyre curves, compound comparison, driver ranking |

---

## 🚀 Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → Connect repo
3. Main file: `app/streamlit_app.py`
4. Deploy (free tier)

---

## 📚 Stack

`fastf1` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `lightgbm` · `shap` · `optuna` · `streamlit` · `plotly` · `matplotlib` · `seaborn`

---

## 📄 License

MIT
