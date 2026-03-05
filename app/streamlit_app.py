"""
app/streamlit_app.py
F1 Lap Time Predictor — Interactive Dashboard
Three tabs: Predict, Strategy Simulator, Circuit Explorer

Run: streamlit run app/streamlit_app.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏎️ F1 Lap Time Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
  h1,h2,h3 { color: #e10600; }
  .stTabs [data-baseweb="tab-list"] { gap: 6px; }
  .stTabs [data-baseweb="tab"] {
    background-color: #161b22; border-radius: 6px 6px 0 0;
    padding: 8px 20px; color: #8b949e; font-weight: 600;
  }
  .stTabs [aria-selected="true"] {
    background-color: #e10600 !important; color: white !important;
  }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 18px; text-align: center;
  }
  .metric-val   { font-size: 2rem; font-weight: 700; }
  .metric-label { font-size: 0.8rem; color: #8b949e; margin-top: 3px; }
  .time-fast  { color: #3fb950; }
  .time-mid   { color: #f0a500; }
  .time-slow  { color: #f85149; }
  div[data-testid="stSidebar"] { background: #161b22; }
  .compound-soft   { color: #f85149; font-weight: 700; }
  .compound-medium { color: #f0a500; font-weight: 700; }
  .compound-hard   { color: #e6edf3; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "raw", "f1_laps.csv")

CIRCUITS = [
    "Bahrain","Saudi Arabia","Australia","Azerbaijan","Miami","Monaco","Spain",
    "Canada","Austria","Britain","Hungary","Belgium","Netherlands","Italy",
    "Singapore","Japan","Qatar","USA","Mexico","Brazil","Las Vegas","Abu Dhabi",
]
CIRCUIT_META = {
    "Bahrain": {"laps":57,"base":93.5,"deg":1.3}, "Saudi Arabia":{"laps":50,"base":91.0,"deg":0.9},
    "Australia":{"laps":58,"base":81.0,"deg":1.1},"Azerbaijan":{"laps":51,"base":104.0,"deg":0.8},
    "Miami":{"laps":57,"base":91.5,"deg":1.2},    "Monaco":{"laps":78,"base":74.0,"deg":0.6},
    "Spain":{"laps":66,"base":80.0,"deg":1.4},    "Canada":{"laps":70,"base":74.5,"deg":1.1},
    "Austria":{"laps":71,"base":66.0,"deg":1.2},  "Britain":{"laps":52,"base":90.0,"deg":1.3},
    "Hungary":{"laps":70,"base":79.0,"deg":1.2},  "Belgium":{"laps":44,"base":106.0,"deg":0.9},
    "Netherlands":{"laps":72,"base":72.0,"deg":1.1},"Italy":{"laps":53,"base":82.0,"deg":0.8},
    "Singapore":{"laps":62,"base":97.0,"deg":0.9},"Japan":{"laps":53,"base":93.0,"deg":1.2},
    "Qatar":{"laps":57,"base":84.0,"deg":1.6},    "USA":{"laps":56,"base":97.0,"deg":1.1},
    "Mexico":{"laps":71,"base":79.0,"deg":1.0},   "Brazil":{"laps":71,"base":72.0,"deg":1.2},
    "Las Vegas":{"laps":50,"base":96.0,"deg":0.8},"Abu Dhabi":{"laps":58,"base":87.0,"deg":1.0},
}

DRIVERS = [
    "Verstappen","Perez","Hamilton","Russell","Leclerc","Sainz",
    "Norris","Piastri","Alonso","Stroll","Ocon","Gasly",
    "Albon","Sargeant","Tsunoda","De Vries","Bottas","Zhou",
    "Magnussen","Hulkenberg",
]
TEAMS = [
    "Red Bull","Mercedes","Ferrari","McLaren","Aston Martin",
    "Alpine","Williams","AlphaTauri","Alfa Romeo","Haas",
]
COMPOUNDS = ["SOFT","MEDIUM","HARD","INTER","WET"]
COMPOUND_COLORS = {
    "SOFT":"#f85149","MEDIUM":"#f0a500","HARD":"#e6edf3","INTER":"#3fb950","WET":"#58a6ff"
}
DRIVER_SKILL = {
    "Verstappen":-0.25,"Perez":0.10,"Hamilton":-0.15,"Russell":-0.05,
    "Leclerc":-0.10,"Sainz":0.05,"Norris":-0.08,"Piastri":0.02,
    "Alonso":-0.12,"Stroll":0.20,"Ocon":0.05,"Gasly":0.00,
    "Albon":0.02,"Sargeant":0.25,"Tsunoda":0.08,"De Vries":0.15,
    "Bottas":0.03,"Zhou":0.12,"Magnussen":0.05,"Hulkenberg":0.03,
}
TEAM_PACE = {
    "Red Bull":0.00,"Mercedes":0.55,"Ferrari":0.40,"McLaren":0.65,
    "Aston Martin":0.90,"Alpine":1.40,"Williams":1.60,
    "AlphaTauri":1.50,"Alfa Romeo":1.30,"Haas":1.45,
}
DRIVER_TEAM = {
    "Verstappen":"Red Bull","Perez":"Red Bull","Hamilton":"Mercedes","Russell":"Mercedes",
    "Leclerc":"Ferrari","Sainz":"Ferrari","Norris":"McLaren","Piastri":"McLaren",
    "Alonso":"Aston Martin","Stroll":"Aston Martin","Ocon":"Alpine","Gasly":"Alpine",
    "Albon":"Williams","Sargeant":"Williams","Tsunoda":"AlphaTauri","De Vries":"AlphaTauri",
    "Bottas":"Alfa Romeo","Zhou":"Alfa Romeo","Magnussen":"Haas","Hulkenberg":"Haas",
}


# ── Model loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        lap_model  = joblib.load(os.path.join(MODELS_DIR, "xgb_lap_model.pkl"))
        lap_prep   = joblib.load(os.path.join(MODELS_DIR, "lap_preprocessor.pkl"))
        pit_model  = joblib.load(os.path.join(MODELS_DIR, "xgb_pit_classifier.pkl"))
        pit_feats  = joblib.load(os.path.join(MODELS_DIR, "pit_features.pkl"))
        fnames     = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
        s1_model   = joblib.load(os.path.join(MODELS_DIR, "lgb_sector1_model.pkl"))
        s1_prep    = joblib.load(os.path.join(MODELS_DIR, "sector1_preprocessor.pkl"))
        s2_model   = joblib.load(os.path.join(MODELS_DIR, "lgb_sector2_model.pkl"))
        s2_prep    = joblib.load(os.path.join(MODELS_DIR, "sector2_preprocessor.pkl"))
        s3_model   = joblib.load(os.path.join(MODELS_DIR, "lgb_sector3_model.pkl"))
        s3_prep    = joblib.load(os.path.join(MODELS_DIR, "sector3_preprocessor.pkl"))
        return dict(lap=lap_model, lap_prep=lap_prep, pit=pit_model, pit_feats=pit_feats,
                    fnames=fnames, s1=s1_model, s1_prep=s1_prep, s2=s2_model, s2_prep=s2_prep,
                    s3=s3_model, s3_prep=s3_prep), True
    except FileNotFoundError as e:
        return {}, False


@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        return None


# ── Feature builder ───────────────────────────────────────────────────────────
COMPOUND_MAP = {"WET":0,"INTER":1,"HARD":2,"MEDIUM":3,"SOFT":4}
COMPOUND_HEAT = {"SOFT":1.5,"MEDIUM":1.0,"HARD":0.6,"INTER":0.4,"WET":0.2}

def build_row(circuit, driver, compound, tyre_life, lap_number, total_laps,
              track_temp, air_temp, humidity, is_raining, sc_laps_ago,
              prev_lap_time, team=None):
    team = team or DRIVER_TEAM.get(driver, "Red Bull")
    cm = CIRCUIT_META.get(circuit, {"laps":57,"base":90.0,"deg":1.1})
    deg = cm["deg"]
    return pd.DataFrame([{
        "circuit":             circuit,
        "team":                team,
        "driver":              driver,
        "tyre_life":           tyre_life,
        "tyre_life_sq":        tyre_life**2,
        "fuel_remaining_laps": total_laps - lap_number,
        "fuel_laps_completed": lap_number,
        "track_temp":          track_temp,
        "air_temp":            air_temp,
        "humidity":            humidity,
        "is_raining":          int(is_raining),
        "safety_car_laps_ago": sc_laps_ago,
        "circuit_deg_factor":  deg,
        "lap_number":          lap_number,
        "total_laps":          total_laps,
        "prev_lap_time":       prev_lap_time,
        "lap_pct":             lap_number / total_laps,
        "is_high_deg_circuit": int(deg > 1.2),
        "compound_encoded":    COMPOUND_MAP.get(compound, 2),
        "temp_tyre_interaction": track_temp * COMPOUND_HEAT.get(compound, 1.0),
        "driver_rolling_form": DRIVER_SKILL.get(driver, 0.0),
    }])


def predict_lap(models, row_df):
    X = models["lap_prep"].transform(row_df)
    return models["lap"].predict(X)[0]


def predict_sectors(models, row_df):
    s = {}
    for key in ["s1","s2","s3"]:
        prep  = models[f"{key}_prep"]
        model = models[key]
        X = prep.transform(row_df)
        s[key] = model.predict(X)[0]
    return s


# ── Lap time gauge ────────────────────────────────────────────────────────────
def lap_gauge(lap_time_s, base_time):
    delta  = lap_time_s - base_time
    pct    = min(100, max(0, (delta + 5) / 20 * 100))
    color  = "#3fb950" if delta < 0.5 else ("#f0a500" if delta < 3 else "#f85149")
    mins   = int(lap_time_s // 60)
    secs   = lap_time_s - mins * 60
    label  = f"{mins}:{secs:06.3f}"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(lap_time_s, 3),
        number={"suffix": "s", "font": {"size": 36, "color": color},
                "valueformat": ".3f"},
        title={"text": f"Predicted Lap Time<br><span style='font-size:1.3rem;color:{color}'>{label}</span>",
               "font": {"size": 15, "color": "#e6edf3"}},
        gauge={
            "axis": {"range": [base_time - 3, base_time + 20],
                     "tickcolor": "#8b949e", "tickfont": {"color": "#8b949e"}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#161b22", "borderwidth": 0,
            "steps": [
                {"range": [base_time-3, base_time+2],  "color": "#0d2818"},
                {"range": [base_time+2, base_time+8],  "color": "#1f1900"},
                {"range": [base_time+8, base_time+20], "color": "#200a0a"},
            ],
        },
    ))
    fig.update_layout(height=260, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      margin=dict(l=30,r=30,t=55,b=10))
    return fig, label


# ── Strategy simulator ────────────────────────────────────────────────────────
def simulate_strategy(models, circuit, driver, strategy_stints,
                      track_temp, air_temp, humidity, is_raining):
    cm = CIRCUIT_META.get(circuit, {"laps":57,"base":90.0,"deg":1.1})
    total_laps = cm["laps"]

    # Build pit lap lookup
    pit_lookup = {}
    for i, (start_lap, compound) in enumerate(strategy_stints):
        end_lap = strategy_stints[i+1][0] - 1 if i+1 < len(strategy_stints) else total_laps
        for lap in range(start_lap, end_lap+1):
            tyre_age = lap - start_lap + 1
            pit_lookup[lap] = (compound, tyre_age)

    laps_data = []
    prev_lap = cm["base"] + TEAM_PACE.get(DRIVER_TEAM.get(driver,"Red Bull"), 0.5)

    for lap in range(1, total_laps+1):
        compound, tyre_age = pit_lookup.get(lap, ("MEDIUM", lap))
        row = build_row(circuit, driver, compound, tyre_age, lap, total_laps,
                        track_temp, air_temp, humidity, is_raining, 50, prev_lap)
        try:
            lap_t = predict_lap(models, row)
        except Exception:
            lap_t = cm["base"] + 1.0
        prev_lap = lap_t
        laps_data.append({"lap": lap, "compound": compound, "tyre_age": tyre_age,
                          "lap_time": lap_t})

    return pd.DataFrame(laps_data)


def strategy_fig(df_a, df_b, label_a, label_b):
    fig = go.Figure()
    for df, name in [(df_a, label_a), (df_b, label_b)]:
        fig.add_trace(go.Scatter(
            x=df["lap"], y=df["lap_time"],
            mode="lines", name=name, line={"width": 2.5},
        ))
        # Pit stop markers
        pit_laps = df[df["tyre_age"] == 1]["lap"]
        if len(pit_laps) > 0:
            fig.add_trace(go.Scatter(
                x=pit_laps, y=df.loc[df["lap"].isin(pit_laps), "lap_time"],
                mode="markers", name=f"{name} — Pit",
                marker={"symbol": "triangle-down", "size": 12, "color": "#f0a500"},
                showlegend=False,
            ))
    fig.update_layout(
        title="Strategy Comparison — Lap Times",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3", height=380,
        xaxis=dict(title="Lap", gridcolor="#30363d"),
        yaxis=dict(title="Lap Time (s)", gridcolor="#30363d"),
        legend=dict(bgcolor="#161b22"),
    )
    return fig


# ── Circuit EDA ───────────────────────────────────────────────────────────────
def circuit_eda(df, circuit):
    sub = df[(df["circuit"] == circuit) & (df["is_outlier"]==0) & (df["is_pit_lap"]==0)].copy()
    if len(sub) == 0:
        st.warning("No data for this circuit.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Tyre deg curve
        fig = go.Figure()
        for compound in ["SOFT","MEDIUM","HARD"]:
            c_sub = sub[sub["compound"]==compound]
            if len(c_sub) == 0:
                continue
            agg = c_sub.groupby("tyre_life")["lap_time_s"].mean()
            agg = agg[agg.index <= 40]
            norm = agg - agg.min()
            fig.add_trace(go.Scatter(
                x=norm.index, y=norm.values, name=compound,
                line={"color": COMPOUND_COLORS[compound], "width": 2.5},
            ))
        fig.update_layout(
            title=f"{circuit} — Tyre Degradation",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=320,
            xaxis=dict(title="Tyre Age (laps)", gridcolor="#30363d"),
            yaxis=dict(title="Delta from best (s)", gridcolor="#30363d"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Compound boxplot
        fig = go.Figure()
        for compound in ["SOFT","MEDIUM","HARD"]:
            c_sub = sub[(sub["compound"]==compound) & (sub["tyre_life"] <= 20)]
            if len(c_sub) == 0:
                continue
            fig.add_trace(go.Box(
                y=c_sub["lap_time_s"], name=compound,
                marker_color=COMPOUND_COLORS[compound],
                line_color=COMPOUND_COLORS[compound],
                fillcolor="rgba(0,0,0,0)",
            ))
        fig.update_layout(
            title=f"{circuit} — Compound Distribution (≤20 laps age)",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=320,
            yaxis=dict(title="Lap Time (s)", gridcolor="#30363d"),
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Driver lap time distribution
        driver_agg = sub.groupby("driver")["lap_time_s"].median().sort_values()
        fig = go.Figure(go.Bar(
            x=driver_agg.values, y=driver_agg.index,
            orientation="h",
            marker_color=[COMPOUND_COLORS["SOFT"] if v == driver_agg.min()
                         else "#8b949e" for v in driver_agg.values],
        ))
        fig.update_layout(
            title=f"{circuit} — Median Lap Time by Driver",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=360,
            xaxis=dict(title="Median Lap Time (s)", gridcolor="#30363d"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Track temp vs lap time scatter
        fig = px.scatter(
            sub.sample(min(500, len(sub))),
            x="track_temp", y="lap_time_s", color="compound",
            color_discrete_map=COMPOUND_COLORS,
            title=f"{circuit} — Track Temp vs Lap Time",
            opacity=0.5, trendline="lowess",
        )
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=360,
            xaxis=dict(gridcolor="#30363d"),
            yaxis=dict(gridcolor="#30363d"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    models, ready = load_models()
    df = load_data()

    # Header
    st.markdown("""
    <div style='display:flex;align-items:center;gap:16px;margin-bottom:8px'>
      <span style='font-size:2.4rem'>🏎️</span>
      <div>
        <h1 style='margin:0;font-size:1.9rem;color:#e10600'>F1 Lap Time Predictor</h1>
        <p style='margin:0;color:#8b949e;font-size:0.9rem'>
          XGBoost · LightGBM · SHAP · 3-season training data
        </p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not ready:
        st.error("⚠️  Models not found. Run `python src/train.py` first.")

    tab_pred, tab_strat, tab_circuit, tab_about = st.tabs([
        "🎯 Lap Predictor", "🔀 Strategy Simulator", "📊 Circuit Explorer", "ℹ️ About"
    ])

    # ═══ TAB 1: LAP PREDICTOR ════════════════════════════════════════════════
    with tab_pred:
        st.markdown("### Configure Lap Conditions")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Race Setup**")
            circuit  = st.selectbox("Circuit", CIRCUITS, key="p_circuit")
            driver   = st.selectbox("Driver", DRIVERS, key="p_driver")
            compound = st.selectbox("Tyre Compound", COMPOUNDS, key="p_compound")
            tyre_life= st.slider("Tyre Age (laps)", 1, 50, 8)

        with c2:
            st.markdown("**Race State**")
            cm       = CIRCUIT_META.get(circuit, {"laps":57})
            lap_num  = st.slider("Lap Number", 1, cm["laps"], 20)
            sc_ago   = st.slider("Laps since Safety Car", 0, 50, 50,
                                 help="50 = no recent SC")
            is_rain  = st.checkbox("Raining", value=False)
            prev_lap = st.number_input("Previous Lap Time (s)", 60.0, 130.0, 92.0, 0.1)

        with c3:
            st.markdown("**Weather**")
            track_t  = st.slider("Track Temp (°C)", 15.0, 60.0, 38.0, 0.5)
            air_t    = st.slider("Air Temp (°C)",    8.0, 45.0, 28.0, 0.5)
            humidity = st.slider("Humidity (%)",     20,  100,  55)

        st.markdown("---")
        if st.button("🏁 Predict Lap Time", type="primary", use_container_width=True):
            if not ready:
                st.error("Models not loaded.")
            else:
                row = build_row(circuit, driver, compound, tyre_life, lap_num,
                                cm["laps"], track_t, air_t, humidity, is_rain,
                                sc_ago, prev_lap)
                lap_t = predict_lap(models, row)
                sectors = predict_sectors(models, row)

                base_t = CIRCUIT_META[circuit]["base"]

                g1, g2, g3, g4 = st.columns(4)
                fig_g, label = lap_gauge(lap_t, base_t)
                g1.plotly_chart(fig_g, use_container_width=True)

                delta = lap_t - base_t
                dc = "time-fast" if delta < 0.5 else ("time-mid" if delta < 3 else "time-slow")
                with g2:
                    st.markdown(f"""
                    <div class='metric-card' style='margin-top:40px'>
                      <div class='metric-val {dc}'>{label}</div>
                      <div class='metric-label'>Lap Time</div>
                    </div>""", unsafe_allow_html=True)
                with g3:
                    sign = "+" if delta >= 0 else ""
                    st.markdown(f"""
                    <div class='metric-card' style='margin-top:40px'>
                      <div class='metric-val {dc}'>{sign}{delta:.3f}s</div>
                      <div class='metric-label'>vs Circuit Best</div>
                    </div>""", unsafe_allow_html=True)
                with g4:
                    s_total = sum(sectors.values())
                    st.markdown(f"""
                    <div class='metric-card' style='margin-top:40px'>
                      <div class='metric-val' style='color:#58a6ff'>{s_total:.3f}s</div>
                      <div class='metric-label'>Sector Sum</div>
                    </div>""", unsafe_allow_html=True)

                # Sector breakdown
                st.markdown("#### Sector Breakdown")
                sc1, sc2, sc3 = st.columns(3)
                for col, key, label_s in [(sc1,"s1","Sector 1"),(sc2,"s2","Sector 2"),(sc3,"s3","Sector 3")]:
                    with col:
                        mins = int(sectors[key]//60)
                        secs = sectors[key] - mins*60
                        t_label = f"{mins}:{secs:06.3f}" if mins > 0 else f"{secs:.3f}s"
                        st.markdown(f"""
                        <div class='metric-card'>
                          <div class='metric-val' style='color:#f0a500;font-size:1.6rem'>{t_label}</div>
                          <div class='metric-label'>{label_s}</div>
                        </div>""", unsafe_allow_html=True)

                # Key factors
                st.markdown("#### 🔑 Key Factors in This Prediction")
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    deg_pen = CIRCUIT_META[circuit]["deg"] * 0.035 * tyre_life
                    st.info(f"🔴 **Tyre Deg**: +{deg_pen:.2f}s penalty at age {tyre_life}")
                with f2:
                    fuel_pen = (cm["laps"] - lap_num) * 0.032
                    st.info(f"⛽ **Fuel Load**: +{fuel_pen:.2f}s for {cm['laps']-lap_num} laps remaining")
                with f3:
                    skill = DRIVER_SKILL.get(driver, 0)
                    sc = "🟢" if skill < 0 else "🟡"
                    st.info(f"{sc} **Driver**: {driver} ({skill:+.2f}s skill rating)")
                with f4:
                    if sc_ago < 5:
                        st.warning(f"🟡 **Safety Car**: Only {sc_ago} laps ago — tyres cold")
                    else:
                        st.success("🟢 **Safety Car**: No recent SC effect")

    # ═══ TAB 2: STRATEGY SIMULATOR ══════════════════════════════════════════
    with tab_strat:
        st.markdown("### Compare Two Pit Strategies")
        st.markdown("*Configure each strategy's stint breakdown then simulate.*")

        cc1, cc2, cc3 = st.columns([2, 2, 1])
        with cc1:
            s_circuit= st.selectbox("Circuit", CIRCUITS, key="s_circuit")
            s_driver = st.selectbox("Driver",  DRIVERS,  key="s_driver")
        with cc2:
            s_track_t = st.slider("Track Temp (°C)", 20.0, 55.0, 38.0, key="s_tt")
            s_rain    = st.checkbox("Wet Race", key="s_rain")
        with cc3:
            s_cm = CIRCUIT_META.get(s_circuit, {"laps":57})
            st.metric("Total Laps", s_cm["laps"])

        st.markdown("---")
        sa_col, sb_col = st.columns(2)

        with sa_col:
            st.markdown("#### 🔵 Strategy A")
            a_c1 = st.selectbox("Stint 1 Compound", COMPOUNDS, index=0, key="a1c")
            a_p1 = st.slider("Pit Stop 1 — Lap", 5, s_cm["laps"]-5, int(s_cm["laps"]*0.35), key="a1p")
            a_c2 = st.selectbox("Stint 2 Compound", COMPOUNDS, index=1, key="a2c")
            a_two_stop = st.checkbox("Two-stop Strategy A?", key="a2s")
            if a_two_stop:
                a_p2 = st.slider("Pit Stop 2 — Lap", a_p1+3, s_cm["laps"]-3,
                                  min(a_p1+15, s_cm["laps"]-5), key="a2p")
                a_c3 = st.selectbox("Stint 3 Compound", COMPOUNDS, index=0, key="a3c")
                strat_a = [(1, a_c1), (a_p1, a_c2), (a_p2, a_c3)]
            else:
                strat_a = [(1, a_c1), (a_p1, a_c2)]

        with sb_col:
            st.markdown("#### 🔴 Strategy B")
            b_c1 = st.selectbox("Stint 1 Compound", COMPOUNDS, index=1, key="b1c")
            b_p1 = st.slider("Pit Stop 1 — Lap", 5, s_cm["laps"]-5, int(s_cm["laps"]*0.50), key="b1p")
            b_c2 = st.selectbox("Stint 2 Compound", COMPOUNDS, index=0, key="b2c")
            b_two_stop = st.checkbox("Two-stop Strategy B?", key="b2s")
            if b_two_stop:
                b_p2 = st.slider("Pit Stop 2 — Lap", b_p1+3, s_cm["laps"]-3,
                                  min(b_p1+15, s_cm["laps"]-5), key="b2p")
                b_c3 = st.selectbox("Stint 3 Compound", COMPOUNDS, index=1, key="b3c")
                strat_b = [(1, b_c1), (b_p1, b_c2), (b_p2, b_c3)]
            else:
                strat_b = [(1, b_c1), (b_p1, b_c2)]

        if st.button("🏎️ Simulate Both Strategies", type="primary", use_container_width=True):
            if not ready:
                st.error("Models not loaded.")
            else:
                with st.spinner("Simulating full race…"):
                    df_a = simulate_strategy(models, s_circuit, s_driver, strat_a,
                                             s_track_t, s_track_t-8, 55, s_rain)
                    df_b = simulate_strategy(models, s_circuit, s_driver, strat_b,
                                             s_track_t, s_track_t-8, 55, s_rain)

                total_a = df_a["lap_time"].sum()
                total_b = df_b["lap_time"].sum()
                winner  = "A" if total_a < total_b else "B"
                gap     = abs(total_a - total_b)

                r1, r2, r3 = st.columns(3)
                r1.metric("Strategy A Total",  f"{total_a:.1f}s")
                r2.metric("Strategy B Total",  f"{total_b:.1f}s")
                r3.metric(f"Strategy {winner} wins by", f"{gap:.2f}s")

                st.plotly_chart(strategy_fig(df_a, df_b, "Strategy A", "Strategy B"),
                                use_container_width=True)

    # ═══ TAB 3: CIRCUIT EXPLORER ════════════════════════════════════════════
    with tab_circuit:
        if df is None:
            st.warning("Data not found. Run `python data/generate_data.py` first.")
        else:
            # Overview metrics
            st.markdown("### All-Circuit Overview")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Laps", f"{len(df):,}")
            m2.metric("Circuits",   df["circuit"].nunique())
            m3.metric("Drivers",    df["driver"].nunique())
            m4.metric("Seasons",    df["season"].nunique())

            # Circuit degradation heatmap
            clean = df[(df["is_outlier"]==0) & (df["is_pit_lap"]==0)]
            deg_by_cmp = clean.groupby(["circuit","compound"])["lap_time_s"].mean().unstack("compound")
            if "HARD" in deg_by_cmp.columns and "SOFT" in deg_by_cmp.columns:
                deg_by_cmp["compound_spread"] = deg_by_cmp["SOFT"] - deg_by_cmp["HARD"]
                cmap_df = deg_by_cmp["compound_spread"].dropna().sort_values(ascending=False)

                fig = go.Figure(go.Bar(
                    x=cmap_df.values, y=cmap_df.index, orientation="h",
                    marker_color=[COMPOUND_COLORS["SOFT"] if v > 1.5 else
                                  "#f0a500" if v > 0.8 else "#3fb950" for v in cmap_df.values],
                    text=[f"{v:.2f}s" for v in cmap_df.values], textposition="outside",
                ))
                fig.update_layout(
                    title="SOFT vs HARD Compound Spread per Circuit (larger = more tyre-dependent)",
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font_color="#e6edf3", height=500,
                    xaxis=dict(title="Lap time difference SOFT→HARD (s)", gridcolor="#30363d"),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### Deep Dive by Circuit")
            sel_circuit = st.selectbox("Select Circuit", CIRCUITS, key="eda_circuit")
            circuit_eda(df, sel_circuit)

    # ═══ TAB 4: ABOUT ═══════════════════════════════════════════════════════
    with tab_about:
        st.markdown("""
### About This Project

An end-to-end F1 data science project with three ML models and an interactive dashboard.

#### Model Architecture

| Stage | Model | Target | Metric |
|---|---|---|---|
| 1 | XGBoost (Optuna-tuned) | Full lap time (s) | MAE in ms |
| 2 | LightGBM × 3 | Sector 1, 2, 3 times separately | MAE in ms |
| 3 | XGBoost classifier | Should pit next lap? | ROC-AUC |

#### Temporal Validation
Train: **2022–2023** seasons → Test: **2024** season.  
No data leakage — the model never sees future race laps during training.

#### Key Features

| Feature | Insight |
|---|---|
| `tyre_life` + `tyre_life²` | Non-linear degradation |
| `compound_encoded` | Ordinal: WET → SOFT |
| `temp_tyre_interaction` | Hot track punishes SOFT more |
| `circuit_deg_factor` | Circuit-specific degradation multiplier |
| `fuel_remaining_laps` | Fuel load effect (~0.03s/lap) |
| `driver_rolling_form` | Recent delta vs circuit average |
| `safety_car_laps_ago` | Cold tyre restart effect |

#### How to Run
```bash
pip install -r requirements.txt
python data/generate_data.py
python src/train.py
python src/explain.py
streamlit run app/streamlit_app.py
```

#### Stack
`fastf1` · `pandas` · `xgboost` · `lightgbm` · `shap` · `optuna` · `streamlit` · `plotly`
        """)


if __name__ == "__main__":
    main()
