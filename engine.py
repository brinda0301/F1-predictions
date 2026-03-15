"""
F1 2026 Race Winner Prediction Engine

Predicts race winners using qualifying data, practice sessions, and
Monte Carlo simulation. Accounts for the 2026 FIA regulation overhaul:
active aero, 350kW MGU-K, sustainable fuels, lighter chassis, etc.

The model self-calibrates after each race. Weights start hand-tuned
based on pre-season estimates, then get adjusted via gradient descent
as real 2026 results come in. By mid-season the model should be
running on learned weights rather than guesses.

Usage:
    python engine.py 02_china
    python engine.py              # runs latest race folder
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


# -- 2026 FIA regulation constants --
#
# Active aero replaced DRS. Every driver gets low-drag mode on straights,
# not just those within 1s. This fundamentally changes how grid position
# translates to win probability. Historically pole won ~60% of races,
# but with everyone getting the wing adjustment, overtaking is way easier.
# After Australia, Leclerc jumped from P4 to P1 at the start and led
# multiple laps from behind. 45% feels right for now.
POLE_WIN_RATE = 0.45

# 90% downforce retained when following 20m behind (was ~70% in 2025).
# Simpler front wings + flatter floors + no beam wing = much less
# dirty air. This is a huge change from the old regs.
DIRTY_AIR_RETENTION = 0.90

# Overtake Mode gives drivers within 1s a burst of extra electrical
# power from the 350kW MGU-K. Replaces DRS. It's a strategic tool -
# you can dump it all at once or spread it across a lap.
OVERTAKE_BOOST = 1.4

# MGU-H is gone. MGU-K went from 120kW to 350kW. The power split
# is now roughly 50/50 between ICE and electric. This means energy
# management is THE defining challenge. Verstappen called it
# "Formula E on steroids." Drivers who mismanage battery become
# sitting ducks on straights.
ENERGY_NOISE = 0.06

# Cars are 30kg lighter (770kg -> 724kg + tyres) and smaller
# (wheelbase -200mm, width -100mm). The "Nimble Car Concept."
# Lighter cars respond differently to fuel load changes and
# are more sensitive to setup. This adds variance.
WEIGHT_VARIANCE = 0.015

# Sustainable fuel is mandatory. Teams are using different blends
# and the performance delta between fuel suppliers is non-trivial.
# Some teams nailed it in testing, others are still figuring it out.
FUEL_SUPPLIERS = {
    "Mercedes": 0.92,     # Petronas, strong pre-season
    "Ferrari": 0.90,      # Shell, good but optimizing
    "McLaren": 0.85,      # BP/Castrol
    "Red Bull": 0.80,     # ExxonMobil, new PU partnership with Ford
    "Racing Bulls": 0.75, # shares Red Bull supply chain
    "Audi": 0.65,         # new operation, still developing
    "Haas": 0.78,         # Ferrari fuel supply
    "Alpine": 0.70,       # own fuel program
    "Williams": 0.72,     # Mercedes fuel supply
    "Aston Martin": 0.60, # new Honda PU, fuel integration issues
    "Cadillac": 0.55,     # brand new team + Ferrari PU, worst integration
}

# New engine partnerships have higher mechanical failure risk.
# Cadillac is a brand new constructor. Audi is basically starting fresh.
# Aston Martin switched to Honda PU. Red Bull runs their own PU for
# the first time (with Ford). These are all unproven combinations.
DNF_RATES = {
    "Mercedes": 0.05,
    "Ferrari": 0.06,
    "McLaren": 0.06,
    "Red Bull": 0.08,     # first season with own PU
    "Racing Bulls": 0.07,
    "Audi": 0.11,         # new everything
    "Haas": 0.07,
    "Alpine": 0.08,
    "Williams": 0.06,
    "Aston Martin": 0.10, # Honda PU integration is rough
    "Cadillac": 0.11,     # brand new team
}

# Smaller tyres (front -25mm, rear -30mm). Different deg profile.
# Less rubber = less grip = more variation in stint lengths.
TYRE_VARIANCE = 0.02


# ---------------------------------------------------------------
# config load/save
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
    """
    Each race has a data.py with qualifying grid, practice times,
    driver experience, team pace, etc. This loader imports it
    dynamically so we can add new races without changing engine code.
    """
    race_path = os.path.join(RACES_DIR, race_folder)
    sys.path.insert(0, race_path)

    # force reimport if we already loaded a different race's data.py
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
    }


# ---------------------------------------------------------------
# feature engineering
# ---------------------------------------------------------------

def build_features(driver, race_data):
    """
    13 features per driver. All normalized to [0, 1].

    The first 11 come from F1 session data (qualifying, practice, sprint).
    The last 2 are 2026-specific: fuel quality and dirty air handling.
    No betting odds. No social media sentiment. Just racing data.
    """
    name = driver["driver"]
    team = driver["team"]
    grid = driver["pos"]
    q_time = driver["q_time"]
    pole_time = race_data["GRID"][0]["q_time"]

    fp1 = race_data["FP1_TIMES"]
    exp = race_data["DRIVER_EXPERIENCE"].get(name, {"f1_seasons": 0})
    deficit = race_data["TEAM_PACE_DEFICIT"].get(team, 1.5)
    start_adv = race_data["START_PROCEDURE"].get(team, 0.0)
    energy = race_data["ENERGY_READINESS"].get(team, 0.4)
    history = race_data["CIRCUIT_HISTORY"].get(name, {"wins": 0, "podiums": 0})
    sprint = race_data["SPRINT_RESULT"]

    # -- QUALIFYING PACE --
    # gap to pole in seconds. 0s gap = 1.0, 4s+ gap = 0.0
    # using seconds not position because the actual time delta
    # tells you way more than "P3 vs P4" which could be 0.001s
    if q_time and pole_time:
        quali_pace = max(0.0, 1.0 - ((q_time - pole_time) / 4.0))
    else:
        quali_pace = 0.15  # no time = back of grid penalty or no lap

    # -- GRID POSITION WIN RATE --
    # historical win rate from each grid slot, adjusted for active aero.
    # the drop-off is steeper than it looks because even though
    # overtaking is easier, the front row still has clean air advantage
    # and gets first call on pit strategy.
    if grid == 1:
        grid_rate = POLE_WIN_RATE
    elif grid <= 3:
        grid_rate = POLE_WIN_RATE * (0.35 / grid)
    elif grid <= 6:
        grid_rate = POLE_WIN_RATE * (0.12 / (grid - 1))
    elif grid <= 10:
        grid_rate = POLE_WIN_RATE * (0.03 / (grid - 3))
    else:
        # P11+ is basically a long shot. but not zero -
        # Verstappen went P20 to P6 in Australia
        grid_rate = max(0.001, 0.01 * (1 - (grid - 10) / 12))

    # -- RACE PACE --
    # team deficit to fastest car from practice long runs.
    # 0.0s deficit = 1.0, 3.0s deficit = 0.0
    race_pace = max(0.0, 1.0 - (deficit / 3.0))

    # -- PRACTICE PACE --
    # FP1 lap time vs session best. rough indicator of single-lap
    # potential on low fuel. not super reliable because teams
    # run different programs, but it's data.
    fp1_time = fp1.get(name)
    if fp1_time is not None:
        valid = [t for t in fp1.values() if t is not None]
        best = min(valid) if valid else fp1_time
        practice_pace = max(0.0, 1.0 - ((fp1_time - best) / 3.0))
    else:
        practice_pace = 0.3

    # -- SPRINT RESULT --
    # sprint race finishing position, if it's a sprint weekend.
    # sprint is basically a mini race with real overtaking and
    # real tyre deg, so it's actually a decent predictor.
    sprint_score = 0.3  # default for non-sprint weekends
    for s in sprint:
        if s["driver"] == name:
            sprint_score = max(0.0, 1.0 - (s["pos"] - 1) / 10)
            break

    # -- TEAMMATE GAP --
    # qualifying delta between teammates. isolates driver skill
    # from car performance. if you're faster than your teammate
    # in the same car, you're extracting more from the package.
    tm_time = None
    for d in race_data["GRID"]:
        if d["team"] == team and d["driver"] != name and d["q_time"]:
            tm_time = d["q_time"]
            break
    if q_time and tm_time:
        teammate_gap = min(1.0, max(0.0, (tm_time - q_time + 1.0) / 2.0))
    elif q_time:
        teammate_gap = 0.7  # no teammate time, assume decent
    else:
        teammate_gap = 0.2

    # -- ADAPTABILITY --
    # how many major reg changes has this driver survived?
    # 2026 is a massive reset. experienced drivers who've been
    # through 2009, 2014, 2017, 2022 reg changes have an edge
    # in adapting their driving style.
    seasons = exp.get("f1_seasons", 0)
    reg_changes_survived = sum([seasons >= 2, seasons >= 5, seasons >= 9, seasons >= 13])
    adaptability = min(1.0, reg_changes_survived * 0.25)

    # -- START PROCEDURE --
    # 2026 start procedure is completely different. the clutch
    # engagement, anti-stall, and energy deployment at launch
    # are all new. some teams nailed it in testing (Ferrari),
    # others are still struggling.
    start_score = min(1.0, max(0.0, 0.5 + start_adv))

    # -- RELIABILITY --
    # did this driver/team finish the previous race?
    # new regs = new failures. if you DNF'd last time,
    # the underlying issue might not be fully fixed.
    r1_finish = exp.get("r1_finish")
    if r1_finish is not None and r1_finish <= 10:
        reliability = 0.95
    elif r1_finish is not None:
        reliability = 0.80
    else:
        reliability = 0.50  # no data yet, coin flip

    # -- ENERGY MANAGEMENT --
    # the big one for 2026. 350kW MGU-K means the battery is now
    # half the car's power. teams that figured out regen strategies
    # in testing have a massive edge. Russell said in Melbourne his
    # battery had "nothing in the tank" at the start. FIA is already
    # reviewing the rules after Australia because it's too dominant
    # a factor.
    energy_score = energy

    # -- TRACK HISTORY --
    # past wins and podiums at this specific circuit.
    # Hamilton has 6 wins at Shanghai - that matters.
    track = min(1.0, history.get("wins", 0) * 0.3 + history.get("podiums", 0) * 0.1)

    # -- FUEL QUALITY (new for 2026) --
    # sustainable fuel is mandatory. different suppliers have
    # different performance levels. it's a genuine differentiator
    # this season because the tech is so new.
    fuel = FUEL_SUPPLIERS.get(team, 0.6)

    # -- DIRTY AIR HANDLING (new for 2026) --
    # with 90% downforce retention at 20m, following is much easier.
    # but some cars handle the remaining 10% loss better than others.
    # teams with simpler front wings and less aero sensitivity
    # benefit more from the new regs. for now, estimating based on
    # how well the team performed in dirty air during testing.
    # front-runners with cleaner aero concepts score higher.
    dirty_air = min(1.0, DIRTY_AIR_RETENTION + deficit * 0.02)

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
        "fuel_quality": round(fuel, 4),
        "dirty_air": round(dirty_air, 4),
    }


# ---------------------------------------------------------------
# prediction
# ---------------------------------------------------------------

def predict(race_folder, config=None):
    """
    Full race prediction pipeline:
    1. load qualifying + practice data
    2. build features for all 22 drivers
    3. score each driver using weighted sum
    4. convert to probabilities via softmax
    5. run 100K Monte Carlo race simulations
    6. output win/podium/DNF percentages
    """
    if config is None:
        config = load_config()

    race_data = load_race_data(race_folder)
    weights = config["weights"]
    temp = config["regulation_params"]["softmax_temperature"]

    # score each driver
    preds = []
    for driver in race_data["GRID"]:
        feats = build_features(driver, race_data)
        # weighted sum across all features
        score = sum(feats.get(k, 0) * weights.get(k, 0) for k in feats)
        preds.append({
            "driver": driver["driver"],
            "team": driver["team"],
            "grid_pos": driver["pos"],
            "features": feats,
            "raw_score": score,
        })

    # softmax to get win probabilities
    # temperature controls how spread out the distribution is.
    # lower temp = winner takes more, higher = more even field.
    # 0.14 gives roughly 30-40% to the favorite which feels
    # realistic for a new reg era where anything can happen.
    scores = np.array([p["raw_score"] for p in preds])
    exp_s = np.exp((scores - scores.max()) / temp)
    probs = exp_s / exp_s.sum()
    for i, p in enumerate(preds):
        p["win_prob"] = float(probs[i])

    # --- Monte Carlo simulation ---
    # 100K races. each simulation applies random events that
    # happen in real F1: safety cars, rain, mechanical failures,
    # driver errors, strategy calls, energy management mistakes.
    # the beauty of Monte Carlo is it captures the fat tails -
    # the 2% chance of a backmarker winning under chaos.

    np.random.seed(42)
    n_sims = 100_000
    drivers = [p["driver"] for p in preds]
    teams = [p["team"] for p in preds]
    n_drivers = len(drivers)
    base_probs = np.array([p["win_prob"] for p in preds])

    win_count = defaultdict(int)
    pod_count = defaultdict(int)
    pts_count = defaultdict(int)
    dnf_count = defaultdict(int)

    for _ in range(n_sims):
        # start with base probabilities + gaussian noise
        perf = np.random.normal(base_probs, base_probs * 0.35 + 0.015)

        # -- energy management chaos (2026-specific) --
        # the 50/50 power split means drivers who mismanage
        # their battery are completely exposed on straights.
        # this is THE story of early 2026.
        for i in range(n_drivers):
            perf[i] += np.random.normal(0, ENERGY_NOISE)
            # new teams have even more energy variance
            if teams[i] in ("Cadillac", "Audi", "Aston Martin"):
                perf[i] += np.random.normal(0, 0.03)

        # -- fuel quality variance (2026-specific) --
        # sustainable fuel performance is inconsistent.
        # some laps the fuel blend works perfectly, others
        # you lose a few tenths. this is a real thing teams
        # are dealing with.
        for i in range(n_drivers):
            fuel_factor = FUEL_SUPPLIERS.get(teams[i], 0.6)
            perf[i] += np.random.normal(0, (1.0 - fuel_factor) * 0.03)

        # -- lighter car sensitivity (2026-specific) --
        # 724kg cars are more sensitive to setup, fuel load,
        # and tyre condition. this adds general variance that
        # wasn't there with the heavier 2025 cars.
        perf += np.random.normal(0, WEIGHT_VARIANCE, n_drivers)

        # -- tyre degradation variance (2026-specific) --
        # smaller tyres = different deg profile. stint lengths
        # vary more, which shuffles the order.
        perf += np.random.normal(0, TYRE_VARIANCE, n_drivers)

        # -- safety car (50% chance) --
        # bunches up the field, resets gaps.
        # with 22 cars and new regs, SC is very common.
        if np.random.random() < 0.50:
            leader_perf = perf.max()
            perf = perf * 0.7 + leader_perf * 0.3
            perf += np.random.normal(0, 0.02, n_drivers)

        # -- VSC (25% chance) --
        # less impactful than full SC but still shuffles things
        # if teams make different pit strategy calls (like Ferrari
        # not pitting under VSC in Australia)
        if np.random.random() < 0.25:
            perf = perf * 0.9 + np.random.uniform(0, 0.08, n_drivers)

        # -- rain (10% chance, depends on circuit) --
        # rain in F1 is a great equalizer. experienced drivers
        # gain an edge. Hamilton's 6 Shanghai wins in mixed
        # conditions aren't a coincidence.
        if np.random.random() < 0.10:
            for i in range(n_drivers):
                seasons = race_data["DRIVER_EXPERIENCE"].get(
                    drivers[i], {}).get("f1_seasons", 0)
                perf[i] += seasons * 0.002
            perf += np.random.normal(0, 0.04, n_drivers)

        # -- lap 1 incidents (30% chance) --
        # 22 cars into turn 1 on new tyres with new start
        # procedures and energy deployment. it's chaos.
        # Piastri crashed on his sighting lap in Melbourne.
        if np.random.random() < 0.30:
            n_victims = np.random.choice([1, 2], p=[0.6, 0.4])
            for _ in range(n_victims):
                victim = np.random.randint(2, min(14, n_drivers))
                perf[victim] *= np.random.uniform(0.15, 0.55)

        # -- active aero overtaking (2026-specific) --
        # everyone gets low-drag mode, not just cars within 1s.
        # this means cars starting further back have a real
        # chance of making up positions on every lap, not just
        # when they're right behind someone. the 90% downforce
        # retention makes following through corners viable too.
        for i in range(n_drivers):
            gp = preds[i]["grid_pos"]
            if gp > 5:
                recovery = (gp - 5) * 0.001 * OVERTAKE_BOOST
                recovery *= np.random.uniform(0.3, 1.0)
                # dirty air is less of a problem now
                recovery *= DIRTY_AIR_RETENTION
                perf[i] += recovery

        # -- overtake mode (2026-specific) --
        # within 1s of car ahead = extra electrical power burst.
        # replaced DRS. it's more powerful but also drains battery
        # so there's a tradeoff. roughly 15% of laps a car behind
        # gets a meaningful boost.
        for i in range(1, n_drivers):
            if np.random.random() < 0.15:
                perf[i] += 0.025

        # -- mechanical DNFs --
        # each team has a base failure rate. new PUs and new
        # teams have higher rates. 27% of the grid DNS/DNF'd
        # in Australia, which validated these estimates.
        for i in range(n_drivers):
            team_rate = DNF_RATES.get(teams[i], 0.07)
            if np.random.random() < team_rate:
                perf[i] = -1  # out of the race

        # -- driver error (3% per driver per race) --
        # everyone makes mistakes. new cars, new braking points,
        # new energy deployment timing. even the best lock up
        # or run wide occasionally.
        for i in range(n_drivers):
            if np.random.random() < 0.03:
                perf[i] *= np.random.uniform(0.2, 0.6)

        # -- pit strategy variance --
        # undercut, overcut, tyre choice, pit stop time.
        # small random noise captures the cumulative effect
        # of 50+ laps of micro-decisions.
        perf += np.random.normal(0, 0.012, n_drivers)

        # tally results
        ranking = np.argsort(-perf)
        finishers = [drivers[r] for r in ranking if perf[r] > 0]

        if finishers:
            win_count[finishers[0]] += 1
        for d in finishers[:3]:
            pod_count[d] += 1
        for d in finishers[:10]:
            pts_count[d] += 1
        for d in drivers:
            if d not in finishers:
                dnf_count[d] += 1

    # build output
    results = []
    for p in preds:
        d = p["driver"]
        results.append({
            "driver": d,
            "team": p["team"],
            "grid_pos": p["grid_pos"],
            "win_pct": round(win_count[d] / n_sims * 100, 2),
            "podium_pct": round(pod_count[d] / n_sims * 100, 2),
            "points_pct": round(pts_count[d] / n_sims * 100, 2),
            "dnf_pct": round(dnf_count[d] / n_sims * 100, 2),
            "model_score": round(p["raw_score"], 4),
            "features": p["features"],
        })
    results.sort(key=lambda x: x["win_pct"], reverse=True)

    # save
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


# ---------------------------------------------------------------
# self-calibration
# ---------------------------------------------------------------

def calibrate(race_round):
    """
    Runs after each race. Compares what the model predicted
    vs what actually happened, then nudges the feature weights
    using simple gradient descent.

    The learning rate decays over the season so early races
    (where we have the least data) cause bigger changes, and
    by mid-season the weights stabilize.

    This is the "train" step. Each race is one training example.
    By race 5 we have 5 data points. Not a lot, but enough to
    correct the worst mis-calibrations from the initial hand-tuning.
    """
    config = load_config()
    weights = config["weights"]

    # find the race folder for this round number
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

    # map driver -> actual finishing position
    actual_pos = {}
    for r in result_data["result"]:
        if r.get("pos") is not None:
            actual_pos[r["driver"]] = r["pos"]

    # learning rate with decay
    # race 1: lr = 0.05
    # race 2: lr = 0.038
    # race 5: lr = 0.02
    # race 10: lr = 0.013
    completed = config.get("last_calibrated_after_round", 0)
    lr = 0.05 / (1 + completed * 0.3)

    # compute weight adjustments
    # basic idea: if the model ranked a driver too high (predicted P2,
    # finished P8), then features that scored high for that driver
    # were overvalued. decrease their weights. and vice versa.
    adjustments = {k: 0.0 for k in weights}
    for pred in pred_data["predictions"][:10]:
        driver = pred["driver"]
        pred_rank = pred_data["predictions"].index(pred) + 1
        actual_rank = actual_pos.get(driver)
        if actual_rank is None:
            continue  # DNF, can't learn from this

        error = pred_rank - actual_rank  # positive = overestimated
        feats = pred.get("features", {})
        for feat_name, feat_val in feats.items():
            if feat_name not in adjustments:
                continue
            if feat_val <= 0.5:
                continue  # only adjust based on strong features
            if error > 0:
                adjustments[feat_name] -= lr * feat_val * 0.1
            elif error < 0:
                adjustments[feat_name] += lr * feat_val * 0.1

    # apply adjustments and renormalize to sum to 1.0
    for feat, adj in adjustments.items():
        weights[feat] = max(0.01, weights[feat] + adj)
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    config["weights"] = weights
    config["last_calibrated_after_round"] = race_round

    # record accuracy for this race
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
    podium_overlap = len(set(pred_podium) & set(actual_podium))

    position_errors = []
    for i, pred in enumerate(pred_data["predictions"]):
        ap = actual_pos.get(pred["driver"])
        if ap:
            position_errors.append(abs(i + 1 - ap))

    entry = {
        "round": race_round,
        "race": race_folder.split("_", 1)[1].replace("_", " ").title(),
        "predicted_winner": pred_winner,
        "predicted_win_pct": pred_data["predictions"][0]["win_pct"],
        "actual_winner": actual_winner,
        "correct": pred_winner == actual_winner,
        "podium_predicted": pred_podium,
        "podium_actual": actual_podium,
        "podium_overlap": podium_overlap,
        "mean_position_error": round(np.mean(position_errors), 2) if position_errors else None,
    }

    # update or append to accuracy history
    history = config.get("accuracy_history", [])
    existing_idx = [i for i, h in enumerate(history) if h["round"] == race_round]
    if existing_idx:
        history[existing_idx[0]] = entry
    else:
        history.append(entry)
    config["accuracy_history"] = history

    save_config(config)
    return config, None


# ---------------------------------------------------------------
# utility functions (used by app.py)
# ---------------------------------------------------------------

def get_race_folders():
    """List all race folders sorted by round number."""
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


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else get_race_folders()[-1]
    print(f"Running prediction for: {folder}")
    out = predict(folder)
    top3 = out["predictions"][:3]
    print(f"\nPredicted winner: {top3[0]['driver']} ({top3[0]['win_pct']}%)")
    print(f"P2: {top3[1]['driver']} ({top3[1]['win_pct']}%)")
    print(f"P3: {top3[2]['driver']} ({top3[2]['win_pct']}%)")
    print(f"\nSimulations: {out['simulations']:,}")