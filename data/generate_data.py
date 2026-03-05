"""
data/generate_data.py
Generates a realistic synthetic F1 lap dataset (2022-2024 seasons).
Mimics the exact schema you'd get from the fastf1 library.

Run: python data/generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os

np.random.seed(42)

# ── Circuit definitions ───────────────────────────────────────────────────────
CIRCUITS = {
    "Bahrain":          {"laps": 57, "base_time": 93.5,  "deg_factor": 1.3, "overtake": 0.7},
    "Saudi Arabia":     {"laps": 50, "base_time": 91.0,  "deg_factor": 0.9, "overtake": 0.4},
    "Australia":        {"laps": 58, "base_time": 81.0,  "deg_factor": 1.1, "overtake": 0.5},
    "Azerbaijan":       {"laps": 51, "base_time": 104.0, "deg_factor": 0.8, "overtake": 0.8},
    "Miami":            {"laps": 57, "base_time": 91.5,  "deg_factor": 1.2, "overtake": 0.6},
    "Monaco":           {"laps": 78, "base_time": 74.0,  "deg_factor": 0.6, "overtake": 0.1},
    "Spain":            {"laps": 66, "base_time": 80.0,  "deg_factor": 1.4, "overtake": 0.5},
    "Canada":           {"laps": 70, "base_time": 74.5,  "deg_factor": 1.1, "overtake": 0.7},
    "Austria":          {"laps": 71, "base_time": 66.0,  "deg_factor": 1.2, "overtake": 0.7},
    "Britain":          {"laps": 52, "base_time": 90.0,  "deg_factor": 1.3, "overtake": 0.6},
    "Hungary":          {"laps": 70, "base_time": 79.0,  "deg_factor": 1.2, "overtake": 0.3},
    "Belgium":          {"laps": 44, "base_time": 106.0, "deg_factor": 0.9, "overtake": 0.7},
    "Netherlands":      {"laps": 72, "base_time": 72.0,  "deg_factor": 1.1, "overtake": 0.4},
    "Italy":            {"laps": 53, "base_time": 82.0,  "deg_factor": 0.8, "overtake": 0.8},
    "Singapore":        {"laps": 62, "base_time": 97.0,  "deg_factor": 0.9, "overtake": 0.3},
    "Japan":            {"laps": 53, "base_time": 93.0,  "deg_factor": 1.2, "overtake": 0.5},
    "Qatar":            {"laps": 57, "base_time": 84.0,  "deg_factor": 1.6, "overtake": 0.5},
    "USA":              {"laps": 56, "base_time": 97.0,  "deg_factor": 1.1, "overtake": 0.7},
    "Mexico":           {"laps": 71, "base_time": 79.0,  "deg_factor": 1.0, "overtake": 0.6},
    "Brazil":           {"laps": 71, "base_time": 72.0,  "deg_factor": 1.2, "overtake": 0.7},
    "Las Vegas":        {"laps": 50, "base_time": 96.0,  "deg_factor": 0.8, "overtake": 0.8},
    "Abu Dhabi":        {"laps": 58, "base_time": 87.0,  "deg_factor": 1.0, "overtake": 0.5},
}

# ── Teams + drivers ───────────────────────────────────────────────────────────
TEAMS = {
    "Red Bull":      {"pace": 0.00,  "drivers": ["Verstappen", "Perez"]},
    "Mercedes":      {"pace": 0.55,  "drivers": ["Hamilton", "Russell"]},
    "Ferrari":       {"pace": 0.40,  "drivers": ["Leclerc", "Sainz"]},
    "McLaren":       {"pace": 0.65,  "drivers": ["Norris", "Piastri"]},
    "Aston Martin":  {"pace": 0.90,  "drivers": ["Alonso", "Stroll"]},
    "Alpine":        {"pace": 1.40,  "drivers": ["Ocon", "Gasly"]},
    "Williams":      {"pace": 1.60,  "drivers": ["Albon", "Sargeant"]},
    "AlphaTauri":    {"pace": 1.50,  "drivers": ["Tsunoda", "De Vries"]},
    "Alfa Romeo":    {"pace": 1.30,  "drivers": ["Bottas", "Zhou"]},
    "Haas":          {"pace": 1.45,  "drivers": ["Magnussen", "Hulkenberg"]},
}

DRIVER_SKILL = {
    "Verstappen": -0.25, "Perez": 0.10,
    "Hamilton":   -0.15, "Russell": -0.05,
    "Leclerc":    -0.10, "Sainz": 0.05,
    "Norris":     -0.08, "Piastri": 0.02,
    "Alonso":     -0.12, "Stroll": 0.20,
    "Ocon":        0.05, "Gasly": 0.00,
    "Albon":       0.02, "Sargeant": 0.25,
    "Tsunoda":     0.08, "De Vries": 0.15,
    "Bottas":      0.03, "Zhou": 0.12,
    "Magnussen":   0.05, "Hulkenberg": 0.03,
}

COMPOUNDS = {
    "SOFT":   {"base_delta": -1.2, "deg_rate": 0.060, "optimal_temp_range": (38, 50)},
    "MEDIUM": {"base_delta":  0.0, "deg_rate": 0.035, "optimal_temp_range": (35, 55)},
    "HARD":   {"base_delta":  0.8, "deg_rate": 0.018, "optimal_temp_range": (30, 60)},
    "INTER":  {"base_delta":  3.0, "deg_rate": 0.012, "optimal_temp_range": (15, 35)},
    "WET":    {"base_delta":  8.0, "deg_rate": 0.008, "optimal_temp_range": (10, 25)},
}

SEASONS = [2022, 2023, 2024]


# ── Weather generator ─────────────────────────────────────────────────────────
def generate_race_weather(circuit, season):
    """Return (track_temp, air_temp, humidity, is_raining) for a race."""
    hot_circuits    = {"Bahrain", "Saudi Arabia", "Qatar", "UAE", "Singapore", "Miami"}
    cold_circuits   = {"Britain", "Belgium", "Netherlands", "Canada"}
    rain_circuits   = {"Britain", "Belgium", "Brazil", "Japan", "Canada"}

    if circuit in hot_circuits:
        track_temp = np.random.uniform(38, 55)
        air_temp   = np.random.uniform(28, 42)
    elif circuit in cold_circuits:
        track_temp = np.random.uniform(18, 35)
        air_temp   = np.random.uniform(14, 28)
    else:
        track_temp = np.random.uniform(28, 45)
        air_temp   = np.random.uniform(20, 35)

    humidity  = np.random.uniform(30, 90)
    rain_prob = 0.30 if circuit in rain_circuits else 0.08
    is_raining= bool(np.random.random() < rain_prob)
    if is_raining:
        track_temp = max(15, track_temp - 10)

    return round(track_temp, 1), round(air_temp, 1), round(humidity, 1), is_raining


# ── Pit strategy generator ────────────────────────────────────────────────────
def generate_pit_strategy(total_laps, is_raining):
    """Return list of (stint_start_lap, compound)."""
    if is_raining:
        # Wet start, possible switch to inters then slicks
        stints = [(1, "WET"), (np.random.randint(8, 18), "INTER")]
        if np.random.random() > 0.4:
            stints.append((np.random.randint(25, 40), "MEDIUM"))
        return stints

    strategies = [
        # 1-stop
        [(1, "MEDIUM"), (int(total_laps * np.random.uniform(0.38, 0.52)), "HARD")],
        [(1, "HARD"),   (int(total_laps * np.random.uniform(0.45, 0.58)), "MEDIUM")],
        [(1, "SOFT"),   (int(total_laps * np.random.uniform(0.28, 0.42)), "HARD")],
        # 2-stop
        [(1, "SOFT"),   (int(total_laps * 0.28), "MEDIUM"), (int(total_laps * 0.60), "SOFT")],
        [(1, "MEDIUM"), (int(total_laps * 0.33), "HARD"),   (int(total_laps * 0.65), "SOFT")],
    ]
    return np.random.choice(strategies)


# ── Lap time computation ──────────────────────────────────────────────────────
def compute_lap_time(base_time, team_pace, driver_skill, compound, tyre_life,
                     fuel_laps_completed, total_laps, track_temp, is_raining,
                     safety_car_laps_ago, circuit_deg_factor, traffic_penalty,
                     lap_number):
    t = base_time

    # Team + driver
    t += team_pace + driver_skill

    # Compound base delta
    c = COMPOUNDS[compound]
    t += c["base_delta"]

    # Tyre degradation (non-linear — accelerates with age)
    deg = c["deg_rate"] * circuit_deg_factor
    t += deg * tyre_life + 0.002 * (tyre_life ** 1.6)

    # Temperature effect on tyres
    opt_lo, opt_hi = c["optimal_temp_range"]
    if track_temp < opt_lo:
        t += (opt_lo - track_temp) * 0.04
    elif track_temp > opt_hi:
        t += (track_temp - opt_hi) * 0.06

    # Fuel load (heavier car = slower; ~0.035s per lap of fuel burned)
    fuel_remaining_laps = total_laps - fuel_laps_completed
    t += fuel_remaining_laps * 0.032

    # Rain penalty
    if is_raining and compound not in ("WET", "INTER"):
        t += np.random.uniform(5, 20)  # massive penalty for wrong tyres
    elif is_raining and compound == "INTER":
        t += np.random.uniform(0, 4)

    # Safety car restart — tyres cold, 3–5 laps recovery
    if 0 < safety_car_laps_ago <= 5:
        t += max(0, (5 - safety_car_laps_ago) * 0.18)

    # Traffic
    t += traffic_penalty

    # Out-lap / in-lap noise
    if tyre_life == 1:
        t += np.random.uniform(0.3, 1.2)   # out-lap, tyres warming
    if tyre_life == 0:
        t += np.random.uniform(0.5, 2.0)   # pit-out

    # Sector splits (roughly 20/40/40 split of lap time with noise)
    noise = np.random.normal(0, 0.08)
    s1 = t * 0.20 + np.random.normal(0, 0.04)
    s2 = t * 0.40 + np.random.normal(0, 0.06)
    s3 = t * 0.40 + np.random.normal(0, 0.05)
    total = s1 + s2 + s3 + noise

    return round(total, 3), round(s1, 3), round(s2, 3), round(s3, 3)


# ── Main generation loop ──────────────────────────────────────────────────────
def generate_dataset():
    rows = []

    for season in SEASONS:
        for circuit, cdata in CIRCUITS.items():
            total_laps  = cdata["laps"]
            base_time   = cdata["base_time"]
            deg_factor  = cdata["deg_factor"]

            track_temp, air_temp, humidity, is_raining = generate_race_weather(circuit, season)

            # Safety car events
            sc_laps = set()
            if np.random.random() < 0.35:
                sc_lap = np.random.randint(5, total_laps - 10)
                sc_laps = set(range(sc_lap, min(sc_lap + 5, total_laps)))

            for team, tdata in TEAMS.items():
                for driver in tdata["drivers"]:
                    team_pace    = tdata["pace"] + np.random.normal(0, 0.1)
                    driver_skill = DRIVER_SKILL[driver]
                    strategy     = generate_pit_strategy(total_laps, is_raining)

                    # Determine stint for each lap
                    pit_laps = [s[0] for s in strategy]
                    compounds_by_stint = {s[0]: s[1] for s in strategy}

                    current_compound  = strategy[0][1]
                    tyre_life         = 0
                    sc_laps_ago       = 99

                    for lap in range(1, total_laps + 1):
                        # Check pit
                        if lap in compounds_by_stint and lap != 1:
                            current_compound = compounds_by_stint[lap]
                            tyre_life = 0

                        tyre_life += 1

                        # Safety car
                        if lap in sc_laps:
                            sc_laps_ago = 0
                        else:
                            sc_laps_ago += 1

                        # Traffic (backmarkers create small penalties)
                        traffic = np.random.exponential(0.05) if np.random.random() < 0.15 else 0.0

                        lap_time, s1, s2, s3 = compute_lap_time(
                            base_time, team_pace, driver_skill,
                            current_compound, tyre_life,
                            lap, total_laps,
                            track_temp, is_raining,
                            sc_laps_ago, deg_factor,
                            traffic, lap
                        )

                        # Pit stop lap — replace with pit lane time
                        is_pit_lap = (lap + 1) in pit_laps and lap != 1
                        if is_pit_lap:
                            pit_time_loss = np.random.uniform(19, 25)
                            lap_time += pit_time_loss

                        rows.append({
                            "season":              season,
                            "circuit":             circuit,
                            "team":                team,
                            "driver":              driver,
                            "lap_number":          lap,
                            "total_laps":          total_laps,
                            "compound":            current_compound,
                            "tyre_life":           tyre_life,
                            "track_temp":          track_temp,
                            "air_temp":            air_temp,
                            "humidity":            humidity,
                            "is_raining":          int(is_raining),
                            "safety_car_laps_ago": min(sc_laps_ago, 50),
                            "fuel_laps_completed": lap,
                            "fuel_remaining_laps": total_laps - lap,
                            "is_pit_lap":          int(is_pit_lap),
                            "lap_time_s":          lap_time,
                            "sector1_s":           s1,
                            "sector2_s":           s2,
                            "sector3_s":           s3,
                            "team_pace_offset":    round(team_pace, 4),
                            "driver_skill":        driver_skill,
                            "circuit_deg_factor":  deg_factor,
                        })

    df = pd.DataFrame(rows)

    # Remove obvious outliers (pit stop laps, first lap chaos)
    df["is_outlier"] = (
        (df["lap_time_s"] > df.groupby(["circuit", "compound"])["lap_time_s"].transform("mean") * 1.15) |
        (df["lap_number"] == 1)
    ).astype(int)

    # Previous lap time (autoregressive feature)
    df = df.sort_values(["season", "circuit", "driver", "lap_number"])
    df["prev_lap_time"] = df.groupby(["season", "circuit", "driver"])["lap_time_s"].shift(1)
    df["lap_delta"] = df["lap_time_s"] - df["prev_lap_time"]

    out = os.path.join(os.path.dirname(__file__), "raw", "f1_laps.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    print(f"✅ Generated {len(df):,} laps → {out}")
    print(f"   Seasons:  {df['season'].unique()}")
    print(f"   Circuits: {df['circuit'].nunique()}")
    print(f"   Drivers:  {df['driver'].nunique()}")
    print(f"   Avg lap time: {df[df['is_outlier']==0]['lap_time_s'].mean():.2f}s")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(5).to_string())
