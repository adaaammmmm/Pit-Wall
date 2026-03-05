"""
src/explain.py
Generates all SHAP explanation plots and saves to outputs/.

Run: python src/explain.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

from src.data_pipeline import load_and_split, get_feature_names

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

BG      = "#0d1117"
SURFACE = "#161b22"
ACCENT  = "#58a6ff"
DANGER  = "#f85149"
SUCCESS = "#3fb950"
YELLOW  = "#f0a500"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"

COMPOUND_COLORS = {
    "SOFT": "#f85149", "MEDIUM": "#f0a500",
    "HARD": "#e6edf3", "INTER": "#3fb950", "WET": "#58a6ff",
}


def style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(MUTED)


def load_models():
    clf   = joblib.load(os.path.join(MODELS_DIR, "xgb_lap_model.pkl"))
    prep  = joblib.load(os.path.join(MODELS_DIR, "lap_preprocessor.pkl"))
    fnames= joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return clf, prep, fnames


# ── Plot 1: SHAP Bar Feature Importance ──────────────────────────────────────
def plot_bar_importance(shap_values, feature_names, top_n=20):
    print("  [1/5] Bar importance…")
    mean_sv = np.abs(shap_values).mean(axis=0)
    idx     = np.argsort(mean_sv)[-top_n:]
    vals    = mean_sv[idx]
    names   = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    style_ax(ax, fig)
    colors = [ACCENT if i >= top_n - 5 else MUTED for i in range(top_n)]
    ax.barh(range(top_n), vals, color=colors, alpha=0.9, height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=9, color=TEXT)
    ax.set_xlabel("Mean |SHAP value| — seconds impact on lap time", color=TEXT)
    ax.set_title("Feature Importance (SHAP) — Lap Time Predictor", fontsize=13, color=TEXT, pad=12)
    ax.grid(axis="x", alpha=0.15, color=MUTED)
    ax.axvline(0, color=MUTED, linewidth=0.5)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "shap_bar_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"     → {path}")


# ── Plot 2: SHAP Beeswarm ──────────────────────────────────────────────────
def plot_beeswarm(shap_values, X_sample, feature_names, top_n=18):
    print("  [2/5] Beeswarm…")
    mean_sv  = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_sv)[-top_n:]
    sv_top   = shap_values[:, top_idx]
    fn_top   = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(11, 8))
    style_ax(ax, fig)

    for row_i, sv_col in enumerate(sv_top.T):
        jitter = np.random.normal(0, 0.09, len(sv_col))
        c = [DANGER if v > 0 else SUCCESS for v in sv_col]
        ax.scatter(sv_col, row_i + jitter, c=c, alpha=0.3, s=7, zorder=2)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(fn_top, fontsize=9, color=TEXT)
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (seconds added to lap time)", color=TEXT)
    ax.set_title("SHAP Beeswarm — Lap Time Model", fontsize=13, color=TEXT, pad=12)
    ax.grid(axis="x", alpha=0.12, color=MUTED)

    slow_p  = mpatches.Patch(color=DANGER,  label="Increases lap time (slower)")
    fast_p  = mpatches.Patch(color=SUCCESS, label="Decreases lap time (faster)")
    ax.legend(handles=[slow_p, fast_p], facecolor=SURFACE, labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"     → {path}")


# ── Plot 3: Tyre Degradation Curves ──────────────────────────────────────────
def plot_tyre_degradation(full_df):
    print("  [3/5] Tyre degradation curves…")
    df = full_df[(full_df["is_outlier"] == 0) & (full_df["is_pit_lap"] == 0)].copy()

    # Normalise within circuit so we see pure degradation
    df["lap_time_norm"] = df.groupby(["season", "circuit"])["lap_time_s"].transform(
        lambda x: x - x.min()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax in axes:
        style_ax(ax, fig)

    # Left: mean degradation per compound
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        sub = df[df["compound"] == compound]
        agg = sub.groupby("tyre_life")["lap_time_norm"].mean()
        agg = agg[agg.index <= 40]
        axes[0].plot(agg.index, agg.values, color=COMPOUND_COLORS[compound],
                     linewidth=2.5, label=compound, marker="o", markersize=3, alpha=0.9)

    axes[0].set_xlabel("Tyre Age (laps)")
    axes[0].set_ylabel("Lap time delta from best (seconds)")
    axes[0].set_title("Tyre Degradation by Compound", fontsize=12)
    axes[0].legend(facecolor=SURFACE, labelcolor=TEXT, fontsize=10)
    axes[0].grid(alpha=0.15, color=MUTED)

    # Right: degradation by compound per high/low deg circuit
    for is_high, label, ls in [(1, "High Degradation", "-"), (0, "Low Degradation", "--")]:
        for compound in ["SOFT", "MEDIUM", "HARD"]:
            sub = df[(df["compound"] == compound) & (df["is_high_deg_circuit"] == is_high)]
            agg = sub.groupby("tyre_life")["lap_time_norm"].mean()
            agg = agg[agg.index <= 35]
            axes[1].plot(agg.index, agg.values, color=COMPOUND_COLORS[compound],
                         linewidth=1.8, linestyle=ls, alpha=0.8,
                         label=f"{compound} ({label})" if is_high else None)

    axes[1].set_xlabel("Tyre Age (laps)")
    axes[1].set_ylabel("Lap time delta (s)")
    axes[1].set_title("Degradation: High vs Low Deg Circuits", fontsize=12)
    axes[1].grid(alpha=0.15, color=MUTED)
    solid = mpatches.Patch(color=MUTED, label="Solid = High Deg")
    dashed = mpatches.Patch(color=MUTED, label="Dashed = Low Deg")
    soft_p = mpatches.Patch(color=COMPOUND_COLORS["SOFT"],   label="SOFT")
    med_p  = mpatches.Patch(color=COMPOUND_COLORS["MEDIUM"], label="MEDIUM")
    hard_p = mpatches.Patch(color=COMPOUND_COLORS["HARD"],   label="HARD")
    axes[1].legend(handles=[soft_p, med_p, hard_p, solid, dashed],
                   facecolor=SURFACE, labelcolor=TEXT, fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "tyre_degradation_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"     → {path}")


# ── Plot 4: SHAP Dependence — Tyre Life ──────────────────────────────────────
def plot_dependence_tyre(shap_values, X_sample, feature_names):
    print("  [4/5] Dependence plot (tyre life)…")
    try:
        ti = feature_names.index("tyre_life")
    except ValueError:
        print("  tyre_life not found, skipping.")
        return

    tyre_vals = X_sample[:, ti]
    tyre_shap = shap_values[:, ti]

    fig, ax = plt.subplots(figsize=(9, 5))
    style_ax(ax, fig)

    sc = ax.scatter(tyre_vals, tyre_shap, c=tyre_shap,
                    cmap="RdYlGn_r", s=8, alpha=0.5, vmin=-0.5, vmax=2.0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("SHAP value (s)", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)
    cbar.ax.set_facecolor(SURFACE)

    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Tyre Age (laps)")
    ax.set_ylabel("SHAP value (seconds added to lap time)")
    ax.set_title("Tyre Age → Lap Time Impact (SHAP Dependence)", fontsize=12)
    ax.grid(alpha=0.12, color=MUTED)

    # Annotation
    ax.annotate("Degradation cliff\n(≈lap 20 on SOFT)",
                xy=(20, 0.8), xytext=(30, 1.5),
                color=DANGER, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=DANGER))

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "shap_dependence_tyre.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"     → {path}")


# ── Plot 5: Driver vs Model — Residual Analysis ───────────────────────────────
def plot_driver_residuals(model, prep, full_df, feature_names):
    print("  [5/5] Driver residual analysis…")

    test_df = full_df[(full_df["season"] == 2024) &
                      (full_df["is_outlier"] == 0) &
                      (full_df["is_pit_lap"] == 0)].copy()

    feat_cols = [c for c in feature_names if c in test_df.columns]
    # Use only numeric cols that are in test_df
    num_cols = [c for c in feat_cols]

    X_te = prep.transform(test_df[[c for c in test_df.columns
                                   if c in full_df.columns
                                   and c not in ["lap_time_s","sector1_s","sector2_s","sector3_s",
                                                 "is_outlier","is_pit_lap","pit_next_lap",
                                                 "lap_delta_from_mean","season","compound",
                                                 "lap_delta"]]])
    y_pred  = model.predict(X_te)
    residuals = test_df["lap_time_s"].values - y_pred
    test_df["residual"] = residuals

    # Mean residual per driver (negative = consistently faster than model predicts)
    driver_res = test_df.groupby("driver")["residual"].agg(["mean","std"]).reset_index()
    driver_res.columns = ["driver", "mean_res", "std_res"]
    driver_res = driver_res.sort_values("mean_res")

    fig, ax = plt.subplots(figsize=(10, 6))
    style_ax(ax, fig)

    colors = [SUCCESS if v < 0 else DANGER for v in driver_res["mean_res"]]
    bars = ax.barh(range(len(driver_res)), driver_res["mean_res"],
                   xerr=driver_res["std_res"] * 0.3,
                   color=colors, alpha=0.85, height=0.7,
                   error_kw={"ecolor": MUTED, "capsize": 3})

    ax.set_yticks(range(len(driver_res)))
    ax.set_yticklabels(driver_res["driver"], fontsize=9, color=TEXT)
    ax.axvline(0, color=MUTED, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Mean residual (actual − predicted) in seconds\nNegative = faster than model expects")
    ax.set_title("Driver Skill vs Model Prediction\n(2024 Season — residual analysis)", fontsize=12)
    ax.grid(axis="x", alpha=0.15, color=MUTED)

    fast_p = mpatches.Patch(color=SUCCESS, label="Faster than predicted")
    slow_p = mpatches.Patch(color=DANGER,  label="Slower than predicted")
    ax.legend(handles=[fast_p, slow_p], facecolor=SURFACE, labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "driver_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"     → {path}")


# ── Model comparison bar chart ────────────────────────────────────────────────
def plot_model_comparison():
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    if not os.path.exists(summary_path):
        return
    with open(summary_path) as f:
        summary = json.load(f)

    results = summary.get("lap_time_results", [])
    if not results:
        return

    models = [r["model"] for r in results]
    mae    = [r.get("mae_ms", 0) for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    style_ax(ax, fig)
    colors = [ACCENT if i == len(models)-1 else MUTED for i in range(len(models))]
    ax.bar(models, mae, color=colors, alpha=0.85, width=0.5)
    ax.set_ylabel("Mean Absolute Error (milliseconds)")
    ax.set_title("Model Comparison — Lap Time MAE", fontsize=12)
    ax.grid(axis="y", alpha=0.15, color=MUTED)
    for i, v in enumerate(mae):
        ax.text(i, v + 2, f"{v:.0f}ms", ha="center", color=TEXT, fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Model comparison → {path}")


def main():
    print("\n🔍 Generating SHAP & Analysis Plots\n")

    clf, prep, fnames = load_models()
    splits = load_and_split()
    X_train, X_test, y_train, y_test = splits["main"]
    full_df = splits["full_df"]

    X_test_t = prep.transform(X_test)

    print("  Computing SHAP values (TreeExplainer)…")
    explainer   = shap.TreeExplainer(clf)
    X_sample    = X_test_t[:800]
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plot_bar_importance(shap_values, fnames)
    plot_beeswarm(shap_values, X_sample, fnames)
    plot_tyre_degradation(full_df)
    plot_dependence_tyre(shap_values, X_sample, fnames)
    plot_driver_residuals(clf, prep, full_df, fnames)
    plot_model_comparison()

    print(f"\n✅ All plots saved → {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
