"""
F1 2026 Race Predictor - Generic Runner
Usage: python model/run_race.py --race 02_china
"""

import json
import os
import sys
import importlib
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

POLE_WIN_RATE_2026 = 0.45
OVERTAKE_FACTOR = 1.4
ENERGY_UNCERTAINTY = 0.04
NEW_TEAM_UNCERTAINTY = {"Cadillac": 0.06, "Audi": 0.04}


def compute_features(driver_entry, race_data):
    name = driver_entry["driver"]
    team = driver_entry["team"]
    grid_pos = driver_entry["pos"]
    q_time = driver_entry["q_time"]

    pole_time = race_data["GRID"][0]["q_time"]
    fp1_times = race_data.get("FP1_TIMES", {})
    experience = race_data.get("DRIVER_EXPERIENCE", {}).get(name, {"f1_seasons": 0, "career_poles": 0})
    team_deficit = race_data.get("TEAM_PACE_DEFICIT", {}).get(team, 1.5)
    start_adv = race_data.get("START_PROCEDURE", {}).get(team, 0.0)
    energy = race_data.get("ENERGY_READINESS", {}).get(team, 0.4)
    circuit_history = race_data.get("CIRCUIT_HISTORY", {}).get(name, {"wins": 0, "podiums": 0})
    sprint_result = race_data.get("SPRINT_RESULT", [])

    if q_time is not None and pole_time is not None:
        q_gap = q_time - pole_time
        quali_pace = max(0, 1.0 - (q_gap / 4.0))
    else:
        quali_pace = 0.15

    if grid_pos == 1:
        grid_win_rate = POLE_WIN_RATE_2026
    elif grid_pos <= 3:
        grid_win_rate = POLE_WIN_RATE_2026 * (0.35 / grid_pos)
    elif grid_pos <= 6:
        grid_win_rate = POLE_WIN_RATE_2026 * (0.12 / (grid_pos - 1))
    elif grid_pos <= 10:
        grid_win_rate = POLE_WIN_RATE_2026 * (0.03 / (grid_pos - 3))
    else:
        grid_win_rate = max(0.001, 0.01 * (1 - (grid_pos - 10) / 12))

    race_pace = max(0, 1.0 - (team_deficit / 3.0))

    fp1_time = fp1_times.get(name)
    if fp1_time is not None:
        fp1_best = min(t for t in fp1_times.values() if t is not None)
        fp1_gap = fp1_time - fp1_best
        practice_pace = max(0, 1.0 - (fp1_gap / 3.0))
    else:
        practice_pace = 0.3

    sprint_score = 0.3
    for sp in sprint_result:
        if sp["driver"] == name:
            sprint_score = max(0, 1.0 - (sp["pos"] - 1) / 10)
            break

    teammate_time = None
    for d in race_data["GRID"]:
        if d["team"] == team and d["driver"] != name and d["q_time"] is not None:
            teammate_time = d["q_time"]
            break
    if q_time is not None and teammate_time is not None:
        delta = teammate_time - q_time
        teammate_gap = min(1.0, max(0, (delta + 1.0) / 2.0))
    elif q_time is not None:
        teammate_gap = 0.7
    else:
        teammate_gap = 0.2

    seasons = experience.get("f1_seasons", 0)
    reg_changes = sum([seasons >= 2, seasons >= 5, seasons >= 9, seasons >= 13])
    adaptability = min(1.0, reg_changes * 0.25)

    start_score = min(1.0, max(0, 0.5 + start_adv))

    r1_finish = experience.get("r1_finish")
    if r1_finish is not None and r1_finish <= 10:
        reliability = 0.95
    elif r1_finish is not None:
        reliability = 0.80
    else:
        reliability = 0.50

    energy_score = energy

    track_wins = circuit_history.get("wins", 0)
    track_podiums = circuit_history.get("podiums", 0)
    track_score = min(1.0, track_wins * 0.3 + track_podiums * 0.1)

    return {
        "quali_pace":     round(quali_pace, 4),
        "grid_win_rate":  round(grid_win_rate, 4),
        "race_pace":      round(race_pace, 4),
        "practice_pace":  round(practice_pace, 4),
        "sprint_score":   round(sprint_score, 4),
        "teammate_gap":   round(teammate_gap, 4),
        "adaptability":   round(adaptability, 4),
        "start_score":    round(start_score, 4),
        "reliability":    round(reliability, 4),
        "energy_score":   round(energy_score, 4),
        "track_history":  round(track_score, 4),
    }


WEIGHTS = {
    "quali_pace":     0.22,
    "grid_win_rate":  0.07,
    "race_pace":      0.13,
    "practice_pace":  0.08,
    "sprint_score":   0.10,
    "teammate_gap":   0.07,
    "adaptability":   0.05,
    "start_score":    0.05,
    "reliability":    0.07,
    "energy_score":   0.07,
    "track_history":  0.04,
}
_total = sum(WEIGHTS.values())
WEIGHTS = {k: round(v / _total, 4) for k, v in WEIGHTS.items()}


def compute_raw_score(features):
    return sum(features.get(k, 0) * w for k, w in WEIGHTS.items())


def softmax_scores(scores, temperature=0.14):
    s = np.array(scores)
    exp_s = np.exp((s - s.max()) / temperature)
    return exp_s / exp_s.sum()


def simulate_race(predictions, race_data):
    drivers = [p["driver"] for p in predictions]
    teams = [p["team"] for p in predictions]
    base = np.array([p["win_prob"] for p in predictions])

    noise_scale = base * 0.35 + 0.015
    performance = np.random.normal(base, noise_scale)

    for i in range(len(drivers)):
        energy_noise = np.random.normal(0, ENERGY_UNCERTAINTY)
        team_extra = NEW_TEAM_UNCERTAINTY.get(teams[i], 0.0)
        performance[i] += np.random.normal(0, team_extra)
        performance[i] += energy_noise

    if np.random.random() < 0.50:
        leader_val = performance.max()
        performance = performance * 0.7 + leader_val * 0.3
        performance += np.random.normal(0, 0.02, len(drivers))

    if np.random.random() < 0.25:
        performance *= 0.9
        performance += np.random.uniform(0, 0.08, len(drivers))

    if np.random.random() < 0.10:
        for i, d in enumerate(drivers):
            exp = race_data.get("DRIVER_EXPERIENCE", {}).get(d, {})
            seasons = exp.get("f1_seasons", 0)
            performance[i] += seasons * 0.002
        performance += np.random.normal(0, 0.04, len(drivers))

    if np.random.random() < 0.30:
        n_victims = np.random.choice([1, 2], p=[0.6, 0.4])
        for _ in range(n_victims):
            victim = np.random.randint(2, min(14, len(drivers)))
            performance[victim] *= np.random.uniform(0.15, 0.55)

    for i in range(len(drivers)):
        grid_pos = predictions[i]["grid_pos"]
        if grid_pos > 5:
            overtake_boost = (grid_pos - 5) * 0.001 * OVERTAKE_FACTOR
            performance[i] += overtake_boost * np.random.uniform(0.3, 1.0)

    for i in range(1, len(drivers)):
        if np.random.random() < 0.15:
            performance[i] += 0.02

    for i in range(len(drivers)):
        base_dnf = 0.07
        if teams[i] in ("Cadillac", "Audi"):
            base_dnf = 0.11
        elif teams[i] == "Aston Martin":
            base_dnf = 0.10
        elif teams[i] == "Red Bull":
            base_dnf = 0.08
        if np.random.random() < base_dnf:
            performance[i] = -1

    for i in range(len(drivers)):
        if np.random.random() < 0.03:
            performance[i] *= np.random.uniform(0.2, 0.6)

    for i in range(len(drivers)):
        performance[i] += np.random.normal(0, 0.012)

    ranking = np.argsort(-performance)
    return [drivers[r] for r in ranking if performance[r] > 0]


def run_monte_carlo(predictions, race_data, n_sims=100000):
    np.random.seed(42)
    all_drivers = [p["driver"] for p in predictions]
    win_counts = defaultdict(int)
    podium_counts = defaultdict(int)
    points_counts = defaultdict(int)
    dnf_counts = defaultdict(int)

    for _ in range(n_sims):
        result = simulate_race(predictions, race_data)
        if result:
            win_counts[result[0]] += 1
        for d in result[:3]:
            podium_counts[d] += 1
        for d in result[:10]:
            points_counts[d] += 1
        for d in all_drivers:
            if d not in result:
                dnf_counts[d] += 1

    output = []
    for p in predictions:
        d = p["driver"]
        output.append({
            "driver": d,
            "team": p["team"],
            "grid_pos": p["grid_pos"],
            "win_pct": round(win_counts[d] / n_sims * 100, 2),
            "podium_pct": round(podium_counts[d] / n_sims * 100, 2),
            "points_pct": round(points_counts[d] / n_sims * 100, 2),
            "dnf_pct": round(dnf_counts[d] / n_sims * 100, 2),
            "model_score": round(p["raw_score"], 4),
            "features": p["features"],
        })
    output.sort(key=lambda x: x["win_pct"], reverse=True)
    return output


def run_prediction(race_folder):
    race_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "races", race_folder)
    sys.path.insert(0, race_path)

    try:
        data_module = importlib.import_module("data")
    except ModuleNotFoundError:
        print(f"Error: No data.py found in races/{race_folder}/")
        return None

    race_data = {
        "GRID": data_module.GRID,
        "FP1_TIMES": getattr(data_module, "FP1_TIMES", {}),
        "SPRINT_RESULT": getattr(data_module, "SPRINT_RESULT", []),
        "DRIVER_EXPERIENCE": getattr(data_module, "DRIVER_EXPERIENCE", {}),
        "TEAM_PACE_DEFICIT": getattr(data_module, "TEAM_PACE_DEFICIT", {}),
        "START_PROCEDURE": getattr(data_module, "START_PROCEDURE", {}),
        "ENERGY_READINESS": getattr(data_module, "ENERGY_READINESS", {}),
        "CIRCUIT_HISTORY": getattr(data_module, "CIRCUIT_HISTORY", {}),
    }

    race_info = getattr(data_module, "RACE_INFO", {"name": race_folder, "laps": 56, "distance_km": 305})
    N_SIMS = 100000

    print("=" * 70)
    print(f"F1 2026 {race_info.get('name', race_folder)} - Race Prediction")
    print("=" * 70)
    print(f"Circuit: {race_info.get('circuit', 'Unknown')}")
    print(f"Date: {race_info.get('date', 'Unknown')} | Laps: {race_info.get('laps', '?')}")
    print(f"Simulations: {N_SIMS:,} | Betting data: NONE")
    print("=" * 70)

    print(f"\n[1/3] Engineering features for {len(race_data['GRID'])} drivers...")
    predictions = []
    for entry in race_data["GRID"]:
        features = compute_features(entry, race_data)
        raw_score = compute_raw_score(features)
        predictions.append({
            "driver": entry["driver"],
            "team": entry["team"],
            "grid_pos": entry["pos"],
            "features": features,
            "raw_score": raw_score,
        })

    print("[2/3] Softmax scoring...")
    scores = [p["raw_score"] for p in predictions]
    probs = softmax_scores(scores)
    for i, p in enumerate(predictions):
        p["win_prob"] = float(probs[i])

    print(f"[3/3] Running {N_SIMS:,} simulations...\n")
    results = run_monte_carlo(predictions, race_data, n_sims=N_SIMS)

    header = f"{'#':<4} {'Driver':<22} {'Team':<15} {'Grid':<5} {'Win%':<8} {'Podium%':<9} {'DNF%':<7}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results):
        print(f"{i+1:<4} {r['driver']:<22} {r['team']:<15} P{r['grid_pos']:<4} "
              f"{r['win_pct']:<8} {r['podium_pct']:<9} {r['dnf_pct']:<7}")

    output_path = os.path.join(race_path, "prediction.json")
    output = {
        "model_version": "v3_generic",
        "race": race_info,
        "simulations": N_SIMS,
        "feature_weights": WEIGHTS,
        "predictions": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    winner = results[0]
    print(f"\n{'=' * 70}")
    print(f"PREDICTED WINNER: {winner['driver']} ({winner['team']})")
    print(f"Win Probability: {winner['win_pct']}%")
    print(f"DNF Risk: {winner['dnf_pct']}%")
    print(f"Grid: P{winner['grid_pos']}")
    print(f"{'=' * 70}")
    print(f"Saved to: {output_path}")
    return output


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--race":
        race = sys.argv[2]
    else:
        races_dir = os.path.join(os.path.dirname(__file__), "..", "races")
        folders = sorted([f for f in os.listdir(races_dir) if os.path.isdir(os.path.join(races_dir, f))])
        race = folders[-1] if folders else "01_australia"
        print(f"No --race specified. Using latest: {race}")

    run_prediction(race)