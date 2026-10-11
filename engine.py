"""
F1 2026 Race Winner Prediction Engine

Two models in one file:
  Monte Carlo: 100K simulations using 18 weighted features. Self-calibrates
               feature weights after each race using gradient descent.
  XGBoost:     trained on past race features and finishing positions. Re-trains
               from scratch each race using all completed races as data.

The two run side by side. predict() saves both outputs to prediction.json.
The dashboard reads both and shows them as separate cards.

Usage:
    python engine.py 04_miami
    python engine.py              # runs latest race
"""

import json
import os
import sys
import importlib
import numpy as np
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RACES_DIR = os.path.join(BASE_DIR, "races")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


# ---------------------------------------------------------------
# 2026 FIA regulation constants
# ---------------------------------------------------------------

# Active aero replaced DRS. All drivers get low-drag mode on straights,
# not just cars within 1s. Pole historically won ~60% of races, but
# active aero makes overtaking much easier. Leclerc jumped P4 to P1
# at the start in Melbourne. 45% from pole feels right for now.
POLE_WIN_RATE = 0.45

# Multiplies the track_history feature. Tested at R17 against 2026 rounds 1-16.
TRACK_HISTORY_SCALE = 0.0   # off from R17: 2026 log loss 1.198 -> 1.083

# FIA target: cars retain 90% downforce when following 20m behind.
# Was ~70% by end of 2025. Simpler front wings + flatter floors +
# no beam wing = way less dirty air. Biggest single change for racing.
DIRTY_AIR_RETENTION = 0.90

# Overtake Mode: within 1s of car ahead, driver gets a burst of
# extra electrical power from the 350kW MGU-K. Replaces DRS.
# Strategic tool: dump it all at once or spread across a lap.
OVERTAKE_BOOST = 1.4

# MGU-H removed. MGU-K tripled: 120kW to 350kW. Power split is
# now 50/50 between ICE and electric. Verstappen called it
# "Formula E on steroids." Battery mismanagement = sitting duck.
ENERGY_NOISE = 0.06

# Cars are 76kg lighter (800kg to 724kg) and smaller (wheelbase
# -200mm, width -100mm). FIA "Nimble Car Concept." Lighter = more
# sensitive to setup, fuel load, and tyre condition.
WEIGHT_VARIANCE = 0.015

# Sustainable fuel is mandatory for the first time. Different
# suppliers have different performance levels. Mercedes/Petronas
# nailed it in testing. Cadillac is struggling.
FUEL_SUPPLIERS = {
    "Mercedes": 0.92,
    "Ferrari": 0.90,
    "McLaren": 0.85,
    "Red Bull": 0.80,
    "Racing Bulls": 0.75,
    "Audi": 0.65,
    "Haas": 0.78,
    "Alpine": 0.70,
    "Williams": 0.72,
    "Aston Martin": 0.60,
    "Cadillac": 0.55,
}

# Mechanical DNF rates per team. New engine partnerships fail more.
# 27% of the grid DNS/DNF'd in Australia. These rates held up.
DNF_RATES = {
    "Mercedes": 0.05,
    "Ferrari": 0.06,
    "McLaren": 0.06,
    "Red Bull": 0.08,
    "Racing Bulls": 0.07,
    "Audi": 0.11,
    "Haas": 0.07,
    "Alpine": 0.08,
    "Williams": 0.06,
    "Aston Martin": 0.10,
    "Cadillac": 0.11,
}

# Pit crew average stop times (seconds). Top teams are under 2.5s.
# These are from 2025 averages, adjusted for 2026 smaller tyres.
PIT_CREW_SPEED = {
    "McLaren": 2.2,
    "Red Bull": 2.3,
    "Ferrari": 2.4,
    "Mercedes": 2.4,
    "Racing Bulls": 2.5,
    "Williams": 2.6,
    "Haas": 2.7,
    "Alpine": 2.7,
    "Audi": 2.9,
    "Aston Martin": 2.8,
    "Cadillac": 3.1,
}

# Tyre degradation sensitivity by team. How well does the car
# preserve tyre life? Lower = better at managing deg.
# Based on long run data from practice and testing.
TYRE_MANAGEMENT = {
    "Mercedes": 0.85,
    "Ferrari": 0.80,
    "McLaren": 0.82,
    "Red Bull": 0.78,
    "Racing Bulls": 0.70,
    "Audi": 0.55,
    "Haas": 0.65,
    "Alpine": 0.60,
    "Williams": 0.62,
    "Aston Martin": 0.50,
    "Cadillac": 0.45,
}


# ---------------------------------------------------------------
# config
# ---------------------------------------------------------------

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------
# data loading
# ---------------------------------------------------------------

def load_race_data(race_folder):
    """Import data.py from a race folder dynamically."""
    race_path = os.path.join(RACES_DIR, race_folder)
    sys.path.insert(0, race_path)

    if "data" in sys.modules:
        del sys.modules["data"]

    mod = importlib.import_module("data")

    return {
        "GRID": mod.GRID,
        "FP1_TIMES": getattr(mod, "FP1_TIMES", {}),
        "SPRINT_RESULT": getattr(mod, "SPRINT_RESULT", []),
        "DRIVER_EXPERIENCE": getattr(mod, "DRIVER_EXPERIENCE", {}),
        "TEAM_PACE_DEFICIT": getattr(mod, "TEAM_PACE_DEFICIT", {}),
        "START_PROCEDURE": getattr(mod, "START_PROCEDURE", {}),
        "ENERGY_READINESS": getattr(mod, "ENERGY_READINESS", {}),
        "CIRCUIT_HISTORY": getattr(mod, "CIRCUIT_HISTORY", {}),
        "RACE_INFO": getattr(mod, "RACE_INFO", {"name": race_folder}),
        # new for v2
        "CIRCUIT": getattr(mod, "CIRCUIT", {}),
        "TYRE_COMPOUNDS": getattr(mod, "TYRE_COMPOUNDS", {}),
        "WEATHER": getattr(mod, "WEATHER", {}),
    }


# ---------------------------------------------------------------
# feature engineering (18 features)
# ---------------------------------------------------------------

def build_features(driver, race_data):
    """
    18 features per driver, all normalized [0, 1].

    Categories:
      Car speed:    quali_pace, race_pace, practice_pace, grid_win_rate
      Driver skill: teammate_gap, adaptability, sprint_score
      Race factors: start_score, reliability, energy_score
      Tyre/pit:     tyre_management, pit_execution, tyre_compound_fit
      2026-specific: fuel_quality, dirty_air, circuit_fit, track_temp
      History:      track_history
    """
    name = driver["driver"]
    team = driver["team"]
    grid = driver["pos"]
    q_time = driver["q_time"]
    # Fastest qualifying lap, not the time of whoever starts first. With a
    # penalised pole sitter GRID[0] is someone slower, which pushed other
    # drivers' quali_pace above 1.0. Fixed R17.
    _times = [d["q_time"] for d in race_data["GRID"] if d.get("q_time")]
    pole_time = min(_times) if _times else None

    fp1 = race_data["FP1_TIMES"]
    exp = race_data["DRIVER_EXPERIENCE"].get(name, {"f1_seasons": 0})
    deficit = race_data["TEAM_PACE_DEFICIT"].get(team, 1.5)
    start_adv = race_data["START_PROCEDURE"].get(team, 0.0)
    energy = race_data["ENERGY_READINESS"].get(team, 0.4)
    history = race_data["CIRCUIT_HISTORY"].get(name, {"wins": 0, "podiums": 0})
    sprint = race_data["SPRINT_RESULT"]
    circuit = race_data["CIRCUIT"]
    tyre_info = race_data["TYRE_COMPOUNDS"]
    weather = race_data["WEATHER"]

    # ---- CAR SPEED ----

    # 1. QUALIFYING PACE
    if q_time and pole_time:
        quali_pace = max(0.0, 1.0 - ((q_time - pole_time) / 4.0))
    else:
        quali_pace = 0.15

    # 2. GRID POSITION WIN RATE
    if grid == 1:
        grid_rate = POLE_WIN_RATE
    elif grid <= 3:
        grid_rate = POLE_WIN_RATE * (0.35 / grid)
    elif grid <= 6:
        grid_rate = POLE_WIN_RATE * (0.12 / (grid - 1))
    elif grid <= 10:
        grid_rate = POLE_WIN_RATE * (0.03 / (grid - 3))
    else:
        grid_rate = max(0.001, 0.01 * (1 - (grid - 10) / 12))

    # 3. RACE PACE
    race_pace = max(0.0, 1.0 - (deficit / 3.0))

    # 4. PRACTICE PACE
    fp1_time = fp1.get(name)
    if fp1_time is not None:
        valid = [t for t in fp1.values() if t is not None]
        best = min(valid) if valid else fp1_time
        practice_pace = max(0.0, 1.0 - ((fp1_time - best) / 3.0))
    else:
        practice_pace = 0.3

    # ---- DRIVER SKILL ----

    # 5. SPRINT RESULT
    sprint_score = 0.3
    for s in sprint:
        if s["driver"] == name:
            sprint_score = max(0.0, 1.0 - (s["pos"] - 1) / 10)
            break

    # 6. TEAMMATE GAP
    tm_time = None
    for d in race_data["GRID"]:
        if d["team"] == team and d["driver"] != name and d["q_time"]:
            tm_time = d["q_time"]
            break
    if q_time and tm_time:
        teammate_gap = min(1.0, max(0.0, (tm_time - q_time + 1.0) / 2.0))
    elif q_time:
        teammate_gap = 0.7
    else:
        teammate_gap = 0.2

    # 7. ADAPTABILITY
    seasons = exp.get("f1_seasons", 0)
    reg_changes = sum([seasons >= 2, seasons >= 5, seasons >= 9, seasons >= 13])
    adaptability = min(1.0, reg_changes * 0.25)

    # ---- RACE FACTORS ----

    # 8. START PROCEDURE
    start_score = min(1.0, max(0.0, 0.5 + start_adv))

    # 9. RELIABILITY
    r1_finish = exp.get("r1_finish")
    if r1_finish is not None and r1_finish <= 10:
        reliability = 0.95
    elif r1_finish is not None:
        reliability = 0.80
    else:
        reliability = 0.50

    # 10. ENERGY MANAGEMENT
    energy_score = energy

    # ---- TYRE AND PIT STRATEGY ----

    # 11. TYRE MANAGEMENT
    tyre_mgmt = TYRE_MANAGEMENT.get(team, 0.5)

    # 12. PIT EXECUTION
    crew_time = PIT_CREW_SPEED.get(team, 2.8)
    pit_exec = max(0.0, min(1.0, (3.5 - crew_time) / 1.5))

    # 13. TYRE COMPOUND FIT
    compound_hardness = tyre_info.get("hardness", 0.5)
    tyre_fit = tyre_mgmt * compound_hardness + quali_pace * (1 - compound_hardness)
    tyre_fit = min(1.0, tyre_fit)

    # ---- 2026-SPECIFIC ----

    # 14. FUEL QUALITY
    fuel = FUEL_SUPPLIERS.get(team, 0.6)

    # 15. DIRTY AIR HANDLING
    dirty_air = min(1.0, DIRTY_AIR_RETENTION + deficit * 0.02)

    # 16. CIRCUIT FIT
    circuit_type = circuit.get("type", "balanced")
    aero_teams = {"Mercedes": 0.9, "Red Bull": 0.85, "McLaren": 0.8}
    mech_teams = {"Ferrari": 0.9, "Haas": 0.75, "Audi": 0.7}
    if circuit_type == "high_speed":
        circuit_fit = aero_teams.get(team, 0.6)
    elif circuit_type == "street":
        circuit_fit = mech_teams.get(team, 0.6)
    else:
        circuit_fit = 0.7 + race_pace * 0.2

    # 17. TRACK TEMPERATURE SENSITIVITY
    track_temp = weather.get("track_temp_c", 30)
    if track_temp >= 40:
        temp_factor = tyre_mgmt
    elif track_temp >= 30:
        temp_factor = tyre_mgmt * 0.6 + quali_pace * 0.4
    else:
        temp_factor = tyre_mgmt * 0.3 + quali_pace * 0.7

    # 18. TRACK HISTORY
    track = min(1.0, history.get("wins", 0) * 0.3 + history.get("podiums", 0) * 0.1)
    track *= TRACK_HISTORY_SCALE

    return {
        "quali_pace": round(quali_pace, 4),
        "grid_win_rate": round(grid_rate, 4),
        "race_pace": round(race_pace, 4),
        "practice_pace": round(practice_pace, 4),
        "sprint_score": round(sprint_score, 4),
        "teammate_gap": round(teammate_gap, 4),
        "adaptability": round(adaptability, 4),
        "start_score": round(start_score, 4),
        "reliability": round(reliability, 4),
        "energy_score": round(energy_score, 4),
        "tyre_management": round(tyre_mgmt, 4),
        "pit_execution": round(pit_exec, 4),
        "tyre_compound_fit": round(tyre_fit, 4),
        "fuel_quality": round(fuel, 4),
        "dirty_air": round(dirty_air, 4),
        "circuit_fit": round(circuit_fit, 4),
        "track_temp": round(temp_factor, 4),
        "track_history": round(track, 4),
    }


def simulate(race_data, config, n_sims=100_000, temp_scale=1.0, recovery="none"):
    """Softmax over weighted features, then the Monte Carlo race simulation.

    Split out of predict() at R17 so the backtest can run the same code with
    fewer simulations. temp_scale multiplies the softmax temperature.
    recovery selects the back-of-grid recovery term: "v1" is the original
    linear bonus that grows with grid slot, "none" removes it (R17 default,
    chosen by backtest).
    """
    weights = config["weights"]
    # Track-dependent softmax temperature
    circuit_type = race_data.get("CIRCUIT", {}).get("type", "balanced")
    weather_rain = race_data.get("WEATHER", {}).get("rain_probability", 0)
    temps = config["regulation_params"].get("track_type_temperatures", {})
    default_temp = config["regulation_params"]["softmax_temperature"]
    if weather_rain > 0.5:
        temp = temps.get("wet", default_temp)
    else:
        temp = temps.get(circuit_type, default_temp)

    preds = []
    for driver in race_data["GRID"]:
        feats = build_features(driver, race_data)
        score = sum(feats.get(k, 0) * weights.get(k, 0) for k in feats)
        preds.append({
            "driver": driver["driver"],
            "team": driver["team"],
            "grid_pos": driver["pos"],
            "features": feats,
            "raw_score": score,
        })

    # softmax
    scores = np.array([p["raw_score"] for p in preds])
    exp_s = np.exp((scores - scores.max()) / (temp * temp_scale))
    probs = exp_s / exp_s.sum()
    for i, p in enumerate(preds):
        p["win_prob"] = float(probs[i])

    # --- Monte Carlo ---
    np.random.seed(42)
    # n_sims comes from the argument
    drivers = [p["driver"] for p in preds]
    teams = [p["team"] for p in preds]
    n = len(drivers)
    base = np.array([p["win_prob"] for p in preds])

    circuit = race_data["CIRCUIT"]
    tyre_info = race_data["TYRE_COMPOUNDS"]
    weather = race_data["WEATHER"]
    pit_loss = circuit.get("pit_loss_seconds", 22)
    one_stop_pct = tyre_info.get("one_stop_probability", 0.65)

    win_c, pod_c, pts_c, dnf_c = (defaultdict(int) for _ in range(4))

    for _ in range(n_sims):
        perf = np.random.normal(base, base * 0.30 + 0.012)

        for i in range(n):
            perf[i] += np.random.normal(0, ENERGY_NOISE)
            if teams[i] in ("Cadillac", "Audi", "Aston Martin"):
                perf[i] += np.random.normal(0, 0.03)

        for i in range(n):
            fuel = FUEL_SUPPLIERS.get(teams[i], 0.6)
            perf[i] += np.random.normal(0, (1.0 - fuel) * 0.03)

        perf += np.random.normal(0, WEIGHT_VARIANCE, n)

        for i in range(n):
            tyre_skill = TYRE_MANAGEMENT.get(teams[i], 0.5)
            deg_penalty = np.random.normal(0, (1.0 - tyre_skill) * 0.04)
            perf[i] += deg_penalty

        for i in range(n):
            is_one_stop = np.random.random() < one_stop_pct
            if not is_one_stop:
                pit_time_loss = pit_loss / 90.0 * 0.01
                tyre_gain = TYRE_MANAGEMENT.get(teams[i], 0.5) * 0.015
                perf[i] += tyre_gain - pit_time_loss

        for i in range(n):
            crew = PIT_CREW_SPEED.get(teams[i], 2.8)
            if np.random.random() < 0.05:
                perf[i] -= np.random.uniform(0.01, 0.04)
            perf[i] += (2.8 - crew) * 0.005

        track_temp = weather.get("track_temp_c", 30)
        if track_temp > 35:
            perf += np.random.normal(0, 0.025, n)
        elif track_temp < 20:
            for i in range(n):
                seasons = race_data["DRIVER_EXPERIENCE"].get(
                    drivers[i], {}).get("f1_seasons", 0)
                perf[i] += seasons * 0.001

        if np.random.random() < 0.50:
            leader = perf.max()
            perf = perf * 0.7 + leader * 0.3
            perf += np.random.normal(0, 0.02, n)
            for i in range(n):
                if np.random.random() < 0.3:
                    tyre_boost = TYRE_MANAGEMENT.get(teams[i], 0.5) * 0.01
                    perf[i] += tyre_boost

        if np.random.random() < 0.25:
            perf = perf * 0.9 + np.random.uniform(0, 0.08, n)

        rain_chance = weather.get("rain_probability", 0.10)
        if np.random.random() < rain_chance:
            for i in range(n):
                seasons = race_data["DRIVER_EXPERIENCE"].get(
                    drivers[i], {}).get("f1_seasons", 0)
                perf[i] += seasons * 0.003
            perf += np.random.normal(0, 0.05, n)

        if np.random.random() < 0.30:
            victims = np.random.choice([1, 2], p=[0.6, 0.4])
            for _ in range(victims):
                v = np.random.randint(2, min(14, n))
                perf[v] *= np.random.uniform(0.15, 0.55)

        # Recovery term. v1 gave every driver outside the top five a bonus
        # that grew with grid slot, so P22 outranked P20 and could outrank pole
        # (see R13 Monza in the README). Off by default from R17.
        if recovery == "v1":
            for i in range(n):
                gp = preds[i]["grid_pos"]
                if gp > 5:
                    rec = (gp - 5) * 0.001 * OVERTAKE_BOOST
                    rec *= np.random.uniform(0.3, 1.0)
                    rec *= DIRTY_AIR_RETENTION
                    perf[i] += rec

        # range(1, n) skipped index 0, which is the pole sitter, so the one
        # driver on the cleanest air was the only one who could never receive
        # this boost. Off-by-one, fixed R15.
        for i in range(n):
            if np.random.random() < 0.15:
                perf[i] += 0.025

        for i in range(n):
            rate = DNF_RATES.get(teams[i], 0.07)
            if np.random.random() < rate:
                perf[i] = -1

        for i in range(n):
            if np.random.random() < 0.03:
                perf[i] *= np.random.uniform(0.2, 0.6)

        perf += np.random.normal(0, 0.012, n)

        ranking = np.argsort(-perf)
        finishers = [drivers[r] for r in ranking if perf[r] > 0]
        if finishers:
            win_c[finishers[0]] += 1
        for d in finishers[:3]:
            pod_c[d] += 1
        for d in finishers[:10]:
            pts_c[d] += 1
        for d in drivers:
            if d not in finishers:
                dnf_c[d] += 1

    results = []
    for p in preds:
        d = p["driver"]
        results.append({
            "driver": d,
            "team": p["team"],
            "grid_pos": p["grid_pos"],
            "win_pct": round(win_c[d] / n_sims * 100, 2),
            "podium_pct": round(pod_c[d] / n_sims * 100, 2),
            "points_pct": round(pts_c[d] / n_sims * 100, 2),
            "dnf_pct": round(dnf_c[d] / n_sims * 100, 2),
            "model_score": round(p["raw_score"], 4),
            "features": p["features"],
        })
    results.sort(key=lambda x: x["win_pct"], reverse=True)
    return results


def predict(race_folder, config=None):
    """
    1. build 18 features per driver
    2. weighted sum + softmax -> probabilities
    3. 100K Monte Carlo simulations with 2026 race events
    4. train XGBoost on past races, predict this race
    5. write both outputs to prediction.json
    """
    if config is None:
        config = load_config()

    race_data = load_race_data(race_folder)
    weights = config["weights"]
    results = simulate(race_data, config)

    # DNF is already handled inside the simulation: a retired driver has
    # perf set to -1 and is excluded from `finishers`, so they cannot win that
    # run. An earlier version multiplied win_pct by (1 - dnf_pct) here as well,
    # which charged every driver for retirement twice and penalised teams with
    # higher hand-set DNF_RATES far more than intended. Removed R15.
    
    # --- XGBoost v2 (from R17) ---
    # Win classifier on 2022-2026 timing data with monotone constraints.
    # See xgb_model.py. v1, a position regressor on 2026 data alone, ran
    # R4-R16 and was removed at R17. Its code is in git history.
    try:
        import xgb_model
        xgb_result = xgb_model.predict(race_data)
    except ImportError:
        xgb_result = {"available": False, "reason": "xgboost not installed"}

    # Models agree if both pick the same winner
    models_agree = None
    if xgb_result and xgb_result.get("available") and results:
        mc_winner = results[0]["driver"]
        xgb_winner = xgb_result["predictions"][0]["driver"]
        models_agree = (mc_winner == xgb_winner)

    output = {
        "race": race_data["RACE_INFO"],
        "simulations": 100_000,
        "weights_used": weights,
        "predictions": results,
        "xgboost": xgb_result,
        "models_agree": models_agree,
    }
    pred_path = os.path.join(RACES_DIR, race_folder, "prediction.json")
    with open(pred_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


# ---------------------------------------------------------------
# self-calibration (gradient descent on Monte Carlo feature weights)
# ---------------------------------------------------------------

def calibrate(race_round):
    """
    Compare predicted ranking vs actual result.
    Nudge weights: overestimated driver -> decrease strong feature weights.
    Learning rate decays each race so early races cause bigger shifts.

    XGBoost is not calibrated. It re-trains from scratch each race using
    the latest result.json files as data.
    """
    config = load_config()
    weights = config["weights"]

    race_folder = None
    for f in sorted(os.listdir(RACES_DIR)):
        if f.startswith(f"{race_round:02d}_") and os.path.isdir(os.path.join(RACES_DIR, f)):
            race_folder = f
            break
    if not race_folder:
        return config, "Race folder not found"

    pred_path = os.path.join(RACES_DIR, race_folder, "prediction.json")
    result_path = os.path.join(RACES_DIR, race_folder, "result.json")
    if not os.path.exists(pred_path) or not os.path.exists(result_path):
        return config, "Need both prediction.json and result.json"

    with open(pred_path) as f:
        pred_data = json.load(f)
    with open(result_path) as f:
        result_data = json.load(f)

    actual_pos = {}
    for r in result_data["result"]:
        if r.get("pos") is not None:
            actual_pos[r["driver"]] = r["pos"]

    completed = config.get("last_calibrated_after_round", 0)
    lr = 0.05 / (1 + completed * 0.3)

    adjustments = {k: 0.0 for k in weights}
    for pred in pred_data["predictions"][:10]:
        driver_name = pred["driver"]
        pred_rank = pred_data["predictions"].index(pred) + 1
        actual_rank = actual_pos.get(driver_name)
        if actual_rank is None:
            continue

        error = pred_rank - actual_rank
        feats = pred.get("features", {})
        for feat_name, feat_val in feats.items():
            if feat_name not in adjustments or feat_val <= 0.5:
                continue
            if error > 0:
                adjustments[feat_name] += lr * feat_val * 0.1
            elif error < 0:
                adjustments[feat_name] -= lr * feat_val * 0.1

    for feat, adj in adjustments.items():
        weights[feat] = max(0.01, weights[feat] + adj)
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    config["weights"] = weights
    config["last_calibrated_after_round"] = race_round

    pred_winner = pred_data["predictions"][0]["driver"]
    actual_winner = next(
        (r["driver"] for r in result_data["result"] if r.get("pos") == 1),
        "Unknown"
    )
    pred_podium = [p["driver"] for p in pred_data["predictions"][:3]]
    actual_podium = [
        r["driver"] for r in result_data["result"]
        if r.get("pos") and r["pos"] <= 3
    ]

    position_errors = []
    for i, pred in enumerate(pred_data["predictions"]):
        ap = actual_pos.get(pred["driver"])
        if ap:
            position_errors.append(abs(i + 1 - ap))

    # Track XGBoost performance too
    xgb_winner_correct = None
    xgb_podium_overlap = None
    if pred_data.get("xgboost") and pred_data["xgboost"].get("available"):
        xgb_winner = pred_data["xgboost"]["predictions"][0]["driver"]
        xgb_winner_correct = (xgb_winner == actual_winner)
        xgb_podium = [p["driver"] for p in pred_data["xgboost"]["predictions"][:3]]
        xgb_podium_overlap = len(set(xgb_podium) & set(actual_podium))

    entry = {
        "round": race_round,
        "race": race_folder.split("_", 1)[1].replace("_", " ").title(),
        "predicted_winner": pred_winner,
        "predicted_win_pct": pred_data["predictions"][0]["win_pct"],
        "actual_winner": actual_winner,
        "correct": pred_winner == actual_winner,
        "podium_predicted": pred_podium,
        "podium_actual": actual_podium,
        "podium_overlap": len(set(pred_podium) & set(actual_podium)),
        "mean_position_error": round(np.mean(position_errors), 2) if position_errors else None,
        "xgb_winner_correct": xgb_winner_correct,
        "xgb_podium_overlap": xgb_podium_overlap,
    }

    history = config.get("accuracy_history", [])
    existing = [i for i, h in enumerate(history) if h["round"] == race_round]
    if existing:
        history[existing[0]] = entry
    else:
        history.append(entry)
    config["accuracy_history"] = history

    save_config(config)
    return config, None


# ---------------------------------------------------------------
# helpers
# ---------------------------------------------------------------

def get_race_folders():
    return sorted([
        f for f in os.listdir(RACES_DIR)
        if os.path.isdir(os.path.join(RACES_DIR, f)) and f[0].isdigit()
    ])

def load_prediction(race_folder):
    path = os.path.join(RACES_DIR, race_folder, "prediction.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_result(race_folder):
    path = os.path.join(RACES_DIR, race_folder, "result.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def has_data(race_folder):
    return os.path.exists(os.path.join(RACES_DIR, race_folder, "data.py"))


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else get_race_folders()[-1]
    print(f"Running prediction for: {folder}")
    out = predict(folder)
    top = out["predictions"][:3]
    print(f"\nMonte Carlo")
    print(f"  P1: {top[0]['driver']} ({top[0]['win_pct']}%)")
    print(f"  P2: {top[1]['driver']} ({top[1]['win_pct']}%)")
    print(f"  P3: {top[2]['driver']} ({top[2]['win_pct']}%)")

    if out.get("xgboost") and out["xgboost"].get("available"):
        xgb_top = out["xgboost"]["predictions"][:3]
        xg = out["xgboost"]
        print(f"\nXGBoost v{xg.get('version', 1)} (trained on {xg['trained_rows']} rows"
              + (f", MAE {xg['mae']})" if "mae" in xg else f", {xg['n_races_trained_on']} races)"))
        for i, r in enumerate(xgb_top, 1):
            print(f"  P{i}: {r['driver']} ({r['win_prob']*100:.1f}%)")
        print(f"\nModels agree on winner: {out['models_agree']}")
    elif out.get("xgboost"):
        print(f"\nXGBoost: {out['xgboost'].get('reason', 'not available')}")
    else:
        print("\nXGBoost not installed. Run: pip install xgboost")

    print(f"\nSimulations: {out['simulations']:,}")