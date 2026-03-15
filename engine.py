"""
F1 2026 Race Prediction Engine
===============================
One model. Gets smarter every race.

How it works:
- Takes qualifying + practice data for any race
- Engineers 11 features per driver
- Scores using weighted ensemble + softmax
- Runs 100K Monte Carlo simulations
- After each race: compares prediction vs result, adjusts weights

The weights start hand-tuned for Race 1.
By Race 5, they're learned from real 2026 data.
By Race 10, the model is fully data-driven.
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

# 2026 regulation constants
POLE_WIN_RATE = 0.45       # Reduced from 60% due to active aero
OVERTAKE_FACTOR = 1.4      # 40% more overtaking than old DRS era
ENERGY_UNCERTAINTY = 0.04  # New power units = unpredictable
NEW_TEAM_DNF = {"Cadillac": 0.06, "Audi": 0.04}


# ============================================================
# CONFIG (weights + accuracy history)
# ============================================================

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


# ============================================================
# DATA LOADER
# ============================================================

def load_race_data(race_folder):
    """Load data.py from a race folder."""
    race_path = os.path.join(RACES_DIR, race_folder)
    sys.path.insert(0, race_path)

    if "data" in sys.modules:
        del sys.modules["data"]

    data_module = importlib.import_module("data")

    return {
        "GRID": data_module.GRID,
        "FP1_TIMES": getattr(data_module, "FP1_TIMES", {}),
        "SPRINT_RESULT": getattr(data_module, "SPRINT_RESULT", []),
        "DRIVER_EXPERIENCE": getattr(data_module, "DRIVER_EXPERIENCE", {}),
        "TEAM_PACE_DEFICIT": getattr(data_module, "TEAM_PACE_DEFICIT", {}),
        "START_PROCEDURE": getattr(data_module, "START_PROCEDURE", {}),
        "ENERGY_READINESS": getattr(data_module, "ENERGY_READINESS", {}),
        "CIRCUIT_HISTORY": getattr(data_module, "CIRCUIT_HISTORY", {}),
        "RACE_INFO": getattr(data_module, "RACE_INFO", {"name": race_folder}),
    }


# ============================================================
# FEATURE ENGINEERING (11 features, all from F1 data)
# ============================================================

def compute_features(driver, race_data):
    """
    Build 11 features for one driver.
    Every feature is 0.0 to 1.0.
    Every feature comes from F1 data. Zero betting market input.
    """
    name = driver["driver"]
    team = driver["team"]
    grid = driver["pos"]
    q_time = driver["q_time"]
    pole_time = race_data["GRID"][0]["q_time"]

    fp1 = race_data["FP1_TIMES"]
    exp = race_data["DRIVER_EXPERIENCE"].get(name, {"f1_seasons": 0})
    deficit = race_data["TEAM_PACE_DEFICIT"].get(team, 1.5)
    start = race_data["START_PROCEDURE"].get(team, 0.0)
    energy = race_data["ENERGY_READINESS"].get(team, 0.4)
    history = race_data["CIRCUIT_HISTORY"].get(name, {"wins": 0, "podiums": 0})
    sprint = race_data["SPRINT_RESULT"]

    # 1. QUALIFYING PACE: time gap to pole in seconds
    #    0.0s gap = 1.0, 4.0s gap = 0.0
    if q_time and pole_time:
        quali_pace = max(0, 1.0 - ((q_time - pole_time) / 4.0))
    else:
        quali_pace = 0.15

    # 2. GRID WIN RATE: historical chance of winning from this position
    #    Adjusted for 2026 active aero (45% from pole, not 60%)
    if grid == 1: grid_rate = POLE_WIN_RATE
    elif grid <= 3: grid_rate = POLE_WIN_RATE * (0.35 / grid)
    elif grid <= 6: grid_rate = POLE_WIN_RATE * (0.12 / (grid - 1))
    elif grid <= 10: grid_rate = POLE_WIN_RATE * (0.03 / (grid - 3))
    else: grid_rate = max(0.001, 0.01 * (1 - (grid - 10) / 12))

    # 3. TEAM RACE PACE: gap to fastest team from practice long runs
    race_pace = max(0, 1.0 - (deficit / 3.0))

    # 4. PRACTICE PACE: FP1 time vs best
    fp1_time = fp1.get(name)
    if fp1_time is not None:
        valid = [t for t in fp1.values() if t is not None]
        best = min(valid) if valid else fp1_time
        practice_pace = max(0, 1.0 - ((fp1_time - best) / 3.0))
    else:
        practice_pace = 0.3

    # 5. SPRINT RESULT: finishing position in sprint race (if sprint weekend)
    sprint_score = 0.3
    for s in sprint:
        if s["driver"] == name:
            sprint_score = max(0, 1.0 - (s["pos"] - 1) / 10)
            break

    # 6. TEAMMATE GAP: qualifying delta to teammate (isolates driver skill)
    tm_time = None
    for d in race_data["GRID"]:
        if d["team"] == team and d["driver"] != name and d["q_time"]:
            tm_time = d["q_time"]
            break
    if q_time and tm_time:
        teammate_gap = min(1.0, max(0, (tm_time - q_time + 1.0) / 2.0))
    elif q_time:
        teammate_gap = 0.7
    else:
        teammate_gap = 0.2

    # 7. ADAPTABILITY: how many regulation changes this driver has survived
    seasons = exp.get("f1_seasons", 0)
    changes = sum([seasons >= 2, seasons >= 5, seasons >= 9, seasons >= 13])
    adaptability = min(1.0, changes * 0.25)

    # 8. START PROCEDURE: team-specific advantage at race start (new for 2026)
    start_score = min(1.0, max(0, 0.5 + start))

    # 9. RELIABILITY: did this driver finish the last race?
    r1 = exp.get("r1_finish")
    if r1 is not None and r1 <= 10: reliability = 0.95
    elif r1 is not None: reliability = 0.80
    else: reliability = 0.50

    # 10. ENERGY MANAGEMENT: team readiness with new 350kW battery
    energy_score = energy

    # 11. TRACK HISTORY: wins and podiums at this specific circuit
    track = min(1.0, history.get("wins", 0) * 0.3 + history.get("podiums", 0) * 0.1)

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
        "track_history": round(track, 4),
    }


# ============================================================
# PREDICTION
# ============================================================

def predict(race_folder, config=None):
    """
    Run full prediction for a race.
    Returns list of drivers sorted by win probability.
    """
    if config is None:
        config = load_config()

    race_data = load_race_data(race_folder)
    weights = config["weights"]
    temp = config["regulation_params"]["softmax_temperature"]

    # Score each driver
    predictions = []
    for driver in race_data["GRID"]:
        features = compute_features(driver, race_data)
        score = sum(features.get(k, 0) * weights.get(k, 0) for k in features)
        predictions.append({
            "driver": driver["driver"],
            "team": driver["team"],
            "grid_pos": driver["pos"],
            "features": features,
            "raw_score": score,
        })

    # Softmax -> probabilities
    scores = np.array([p["raw_score"] for p in predictions])
    exp_s = np.exp((scores - scores.max()) / temp)
    probs = exp_s / exp_s.sum()
    for i, p in enumerate(predictions):
        p["win_prob"] = float(probs[i])

    # Monte Carlo: 100K race simulations
    np.random.seed(42)
    n_sims = 100000
    drivers = [p["driver"] for p in predictions]
    teams = [p["team"] for p in predictions]
    base = np.array([p["win_prob"] for p in predictions])

    win_c, pod_c, pts_c, dnf_c = (defaultdict(int) for _ in range(4))

    for _ in range(n_sims):
        perf = np.random.normal(base, base * 0.35 + 0.015)

        # Energy management noise (2026-specific)
        for i in range(len(drivers)):
            perf[i] += np.random.normal(0, ENERGY_UNCERTAINTY)
            perf[i] += np.random.normal(0, NEW_TEAM_DNF.get(teams[i], 0.0))

        # Safety car (50% at most circuits)
        if np.random.random() < 0.50:
            lv = perf.max()
            perf = perf * 0.7 + lv * 0.3 + np.random.normal(0, 0.02, len(drivers))

        # VSC (25%)
        if np.random.random() < 0.25:
            perf = perf * 0.9 + np.random.uniform(0, 0.08, len(drivers))

        # Rain (10%)
        if np.random.random() < 0.10:
            for i, d in enumerate(drivers):
                s = race_data["DRIVER_EXPERIENCE"].get(d, {}).get("f1_seasons", 0)
                perf[i] += s * 0.002
            perf += np.random.normal(0, 0.04, len(drivers))

        # Lap 1 incidents (30%)
        if np.random.random() < 0.30:
            for _ in range(np.random.choice([1, 2], p=[0.6, 0.4])):
                v = np.random.randint(2, min(14, len(drivers)))
                perf[v] *= np.random.uniform(0.15, 0.55)

        # Active aero overtaking boost (2026-specific)
        for i in range(len(drivers)):
            gp = predictions[i]["grid_pos"]
            if gp > 5:
                perf[i] += (gp - 5) * 0.001 * OVERTAKE_FACTOR * np.random.uniform(0.3, 1.0)

        # Overtake mode (2026-specific)
        for i in range(1, len(drivers)):
            if np.random.random() < 0.15:
                perf[i] += 0.02

        # DNFs
        for i in range(len(drivers)):
            rate = 0.07
            if teams[i] in ("Cadillac", "Audi"): rate = 0.11
            elif teams[i] == "Aston Martin": rate = 0.10
            elif teams[i] == "Red Bull": rate = 0.08
            if np.random.random() < rate:
                perf[i] = -1

        # Driver error (3%)
        for i in range(len(drivers)):
            if np.random.random() < 0.03:
                perf[i] *= np.random.uniform(0.2, 0.6)

        # Strategy variance
        perf += np.random.normal(0, 0.012, len(drivers))

        # Results
        ranking = np.argsort(-perf)
        finishers = [drivers[r] for r in ranking if perf[r] > 0]
        if finishers: win_c[finishers[0]] += 1
        for d in finishers[:3]: pod_c[d] += 1
        for d in finishers[:10]: pts_c[d] += 1
        for d in drivers:
            if d not in finishers: dnf_c[d] += 1

    # Build output
    results = []
    for p in predictions:
        d = p["driver"]
        results.append({
            "driver": d, "team": p["team"], "grid_pos": p["grid_pos"],
            "win_pct": round(win_c[d] / n_sims * 100, 2),
            "podium_pct": round(pod_c[d] / n_sims * 100, 2),
            "points_pct": round(pts_c[d] / n_sims * 100, 2),
            "dnf_pct": round(dnf_c[d] / n_sims * 100, 2),
            "model_score": round(p["raw_score"], 4),
            "features": p["features"],
        })
    results.sort(key=lambda x: x["win_pct"], reverse=True)

    # Save prediction
    output = {
        "race": race_data["RACE_INFO"],
        "simulations": n_sims,
        "weights_used": weights,
        "predictions": results,
    }
    pred_path = os.path.join(RACES_DIR, race_folder, "prediction.json")
    with open(pred_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


# ============================================================
# CALIBRATION (model learns from results)
# ============================================================

def calibrate(race_round):
    """
    After a race: compare prediction vs actual result.
    Adjust weights using gradient descent.
    The model gets smarter every race.
    """
    config = load_config()
    weights = config["weights"]

    # Find race folder
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
        return config, "Missing prediction or result file"

    with open(pred_path) as f: pred_data = json.load(f)
    with open(result_path) as f: result_data = json.load(f)

    # Build actual position map
    actual = {}
    for r in result_data["result"]:
        if r.get("pos") is not None:
            actual[r["driver"]] = r["pos"]

    # Learning rate: decays as we get more data
    completed = config.get("last_calibrated_after_round", 0)
    lr = 0.05 / (1 + completed * 0.3)

    # Gradient: compare predicted rank vs actual rank
    adjustments = {k: 0.0 for k in weights}
    for pred in pred_data["predictions"][:10]:
        driver = pred["driver"]
        pred_rank = pred_data["predictions"].index(pred) + 1
        actual_rank = actual.get(driver)
        if actual_rank is None:
            continue

        rank_error = pred_rank - actual_rank
        features = pred.get("features", {})
        for feat, val in features.items():
            if feat in adjustments and val > 0.5:
                if rank_error > 0:   # overestimated -> decrease weight
                    adjustments[feat] -= lr * val * 0.1
                elif rank_error < 0: # underestimated -> increase weight
                    adjustments[feat] += lr * val * 0.1

    # Apply and normalize
    for feat, adj in adjustments.items():
        weights[feat] = max(0.01, weights[feat] + adj)
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    config["weights"] = weights
    config["last_calibrated_after_round"] = race_round

    # Accuracy tracking
    pred_winner = pred_data["predictions"][0]["driver"]
    actual_winner = next((r["driver"] for r in result_data["result"] if r.get("pos") == 1), "Unknown")
    pred_podium = [p["driver"] for p in pred_data["predictions"][:3]]
    actual_podium = [r["driver"] for r in result_data["result"] if r.get("pos") and r["pos"] <= 3]
    overlap = len(set(pred_podium) & set(actual_podium))

    errors = []
    for i, pred in enumerate(pred_data["predictions"]):
        ap = actual.get(pred["driver"])
        if ap: errors.append(abs(i + 1 - ap))

    entry = {
        "round": race_round,
        "race": race_folder.split("_", 1)[1].replace("_", " ").title(),
        "predicted_winner": pred_winner,
        "predicted_win_pct": pred_data["predictions"][0]["win_pct"],
        "actual_winner": actual_winner,
        "correct": pred_winner == actual_winner,
        "podium_predicted": pred_podium,
        "podium_actual": actual_podium,
        "podium_overlap": overlap,
        "mean_position_error": round(np.mean(errors), 2) if errors else None,
    }

    history = config.get("accuracy_history", [])
    existing = [i for i, h in enumerate(history) if h["round"] == race_round]
    if existing: history[existing[0]] = entry
    else: history.append(entry)
    config["accuracy_history"] = history

    save_config(config)
    return config, None


# ============================================================
# HELPERS
# ============================================================

def get_race_folders():
    return sorted([f for f in os.listdir(RACES_DIR)
                    if os.path.isdir(os.path.join(RACES_DIR, f)) and f[0].isdigit()])

def load_prediction(race_folder):
    p = os.path.join(RACES_DIR, race_folder, "prediction.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

def load_result(race_folder):
    p = os.path.join(RACES_DIR, race_folder, "result.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return None

def has_data(race_folder):
    return os.path.exists(os.path.join(RACES_DIR, race_folder, "data.py"))


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else get_race_folders()[-1]
    print(f"Predicting: {folder}")
    result = predict(folder)
    w = result["predictions"][0]
    print(f"Winner: {w['driver']} ({w['win_pct']}%)")
