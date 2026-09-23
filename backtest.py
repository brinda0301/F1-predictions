"""
backtest.py

Replays past seasons through the feature model so changes can be measured
against 60-plus races instead of 14.

Why this exists: with 14 races, a two-race swing is noise, so no tuning change
can be shown to help. This pulls 2024 and 2025 from the same timing API the
live pipeline uses, builds the same feature rows, and scores them.

What it measures, all leave-one-race-out so nothing is scored on data it
trained on:

  - XGBoost held-out MAE against the finish-equals-grid baseline
  - Winner accuracy for the weighted-score model against the always-pole baseline
  - Whether each feature group earns its place, by dropping it and re-measuring

Practice times are not exposed by any public API, so FP1_TIMES is empty for
every backtest race and every driver takes the same neutral value. Circuit,
tyre and weather blocks are likewise held at defaults. That is deliberate: it
isolates the features that come from timing data, which are the ones available
for any season.

Usage:
    python backtest.py                # 2024 + 2025
    python backtest.py --years 2025   # one season
    python backtest.py --quick        # first 8 rounds of each, for a fast check
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path(".backtest_cache")
API = "https://api.jolpi.ca/ergast/f1"

TEAMS = {
    "mclaren": "McLaren", "mercedes": "Mercedes", "ferrari": "Ferrari",
    "red_bull": "Red Bull", "rb": "Racing Bulls", "sauber": "Audi",
    "audi": "Audi", "alpine": "Alpine", "haas": "Haas",
    "williams": "Williams", "aston_martin": "Aston Martin",
    "cadillac": "Cadillac",
}

# Feature groups, used by the ablation. "measured" comes from timing data;
# "hand_set" comes from constants typed into engine.py by hand.
HAND_SET = [
    "energy_score", "tyre_management", "fuel_quality", "track_temp",
    "start_score", "pit_execution", "circuit_fit", "dirty_air",
    "tyre_compound_fit", "reliability",
]
MEASURED = [
    "quali_pace", "race_pace", "grid_win_rate", "practice_pace",
    "sprint_score", "teammate_gap", "adaptability", "track_history",
]


_last_call = [0.0]
MIN_GAP = 0.35   # seconds between requests; the API allows a few per second


def get(path, attempts=6):
    """Fetch one endpoint, cached on disk, throttled, with backoff on 429.

    The API rate-limits unauthenticated callers. A backtest fires hundreds of
    requests, so without a gap between them the run dies partway through with
    HTTP 429. Responses are cached, so a second run costs no requests at all.
    """
    CACHE.mkdir(exist_ok=True)
    key = CACHE / (path.replace("/", "_").replace("?", "_") + ".json")
    if key.exists():
        return json.loads(key.read_text())

    url = f"{API}/{path}?format=json&limit=100"
    delay = 2.0
    for attempt in range(attempts):
        gap = MIN_GAP - (time.time() - _last_call[0])
        if gap > 0:
            time.sleep(gap)
        _last_call[0] = time.time()
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)["MRData"]
            key.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            print(f"    rate limited, waiting {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"gave up on {url}")


def secs(clock):
    if not clock:
        return None
    parts = clock.strip().split(":")
    try:
        if len(parts) == 2:
            return round(int(parts[0]) * 60 + float(parts[1]), 3)
        return round(float(parts[0]), 3)
    except ValueError:
        return None


def name_of(d):
    return f"{d['givenName']} {d['familyName']}"


def qualifying(year, rnd):
    races = get(f"{year}/{rnd}/qualifying/")["RaceTable"]["Races"]
    if not races:
        return []
    out = []
    for q in races[0]["QualifyingResults"]:
        laps = [secs(q.get(k)) for k in ("Q1", "Q2", "Q3")]
        laps = [l for l in laps if l is not None]
        out.append({
            "driver": name_of(q["Driver"]),
            "team": TEAMS.get(q["Constructor"]["constructorId"], q["Constructor"]["name"]),
            "pos": int(q["position"]),
            "q_time": min(laps) if laps else None,
        })
    out.sort(key=lambda x: x["pos"])
    return out


def results(year, rnd):
    races = get(f"{year}/{rnd}/results/")["RaceTable"]["Races"]
    if not races:
        return []
    return [{"pos": int(x["position"]), "driver": name_of(x["Driver"])}
            for x in races[0]["Results"]]


def sprint(year, rnd):
    races = get(f"{year}/{rnd}/sprint/")["RaceTable"]["Races"]
    if not races:
        return []
    return [{"pos": int(x["position"]), "driver": name_of(x["Driver"]),
             "team": TEAMS.get(x["Constructor"]["constructorId"], x["Constructor"]["name"])}
            for x in races[0]["SprintResults"]]


def seasons_before(driver, year, debut_cache={}):
    """Rough experience count: seasons the driver appears in before this one."""
    if not debut_cache:
        for y in range(2018, 2027):
            try:
                tbl = get(f"{y}/drivers/")["DriverTable"]["Drivers"]
            except Exception:
                continue
            for d in tbl:
                n = name_of(d)
                debut_cache.setdefault(n, y)
    return max(0, year - debut_cache.get(driver, year))


def pace_deficit(grid):
    best = {}
    for d in grid:
        if d["q_time"] is None:
            continue
        best[d["team"]] = min(best.get(d["team"], 9e9), d["q_time"])
    if not best:
        return {}
    fastest = min(best.values())
    return {t: round(v - fastest, 3) for t, v in best.items()}


def race_data(year, rnd, engine):
    grid = qualifying(year, rnd)
    if len(grid) < 10:
        return None
    res = results(year, rnd)
    if not res:
        return None
    prev = {r["driver"]: r["pos"] for r in results(year, rnd - 1)} if rnd > 1 else {}
    fallback = (len(grid) + 1) // 2
    exp = {
        d["driver"]: {
            "f1_seasons": seasons_before(d["driver"], year),
            "r1_finish": prev.get(d["driver"], fallback),
        }
        for d in grid
    }
    return {
        "RACE_INFO": {"round": rnd, "name": f"{year} R{rnd}"},
        "GRID": grid,
        "FP1_TIMES": {},
        "SPRINT_RESULT": sprint(year, rnd),
        "DRIVER_EXPERIENCE": exp,
        "TEAM_PACE_DEFICIT": pace_deficit(grid),
        "START_PROCEDURE": engine.START_PROCEDURE_DEFAULT if hasattr(engine, "START_PROCEDURE_DEFAULT") else {},
        "ENERGY_READINESS": {},
        "CIRCUIT": {"type": "balanced", "pit_loss_seconds": 21},
        "TYRE_COMPOUNDS": {"hardness": 0.5, "one_stop_probability": 0.65},
        "WEATHER": {"track_temp_c": 30, "rain_probability": 0.10},
        "CIRCUIT_HISTORY": {},
        "_RESULT": {r["driver"]: r["pos"] for r in res},
    }


def build(years, max_round, engine):
    races = []
    for year in years:
        try:
            n = len(get(f"{year}/")["RaceTable"]["Races"])
        except Exception as exc:
            print(f"  {year}: could not list rounds ({type(exc).__name__}), skipped")
            continue
        for rnd in range(1, min(n, max_round) + 1):
            try:
                rd = race_data(year, rnd, engine)
            except Exception as exc:
                print(f"  {year} R{rnd}: skipped ({type(exc).__name__})")
                continue
            if rd is None:
                continue
            feats, ys, grids, names = [], [], [], []
            for d in rd["GRID"]:
                if d["driver"] not in rd["_RESULT"]:
                    continue
                f = engine.build_features(d, rd)
                feats.append(f)
                ys.append(rd["_RESULT"][d["driver"]])
                grids.append(d["pos"])
                names.append(d["driver"])
            if len(feats) < 10:
                continue
            races.append({"label": f"{year} R{rnd}", "feats": feats, "y": ys,
                          "grid": grids, "names": names})
            print(f"  {year} R{rnd}: {len(feats)} drivers")
    return races


def evaluate(races, config, drop=()):
    """Leave-one-race-out. Returns held-out MAE, model and baseline accuracy."""
    from xgboost import XGBRegressor
    keys = [k for k in sorted(races[0]["feats"][0].keys()) if k not in drop]
    weights = config["weights"]

    maes, base_maes = [], []
    model_hits = pole_hits = 0

    for i, hold in enumerate(races):
        train = [r for j, r in enumerate(races) if j != i]
        X = np.array([[f[k] for k in keys] for r in train for f in r["feats"]])
        y = np.array([p for r in train for p in r["y"]])
        m = XGBRegressor(n_estimators=min(300, 50 + (len(X) // 20) * 20),
                         max_depth=3, learning_rate=0.1,
                         random_state=42, verbosity=0)
        m.fit(X, y)
        Xh = np.array([[f[k] for k in keys] for f in hold["feats"]])
        maes.append(np.mean(np.abs(m.predict(Xh) - np.array(hold["y"]))))
        base_maes.append(np.mean(np.abs(np.array(hold["grid"]) - np.array(hold["y"]))))

        scores = [sum(f.get(k, 0) * weights.get(k, 0) for k in keys) for f in hold["feats"]]
        winner = hold["names"][int(np.argmax(hold["y"] == np.min(hold["y"])))] if False else \
            hold["names"][int(np.argmin(hold["y"]))]
        if hold["names"][int(np.argmax(scores))] == winner:
            model_hits += 1
        if hold["names"][int(np.argmin(hold["grid"]))] == winner:
            pole_hits += 1

    n = len(races)
    return {
        "races": n,
        "xgb_mae": float(np.mean(maes)),
        "baseline_mae": float(np.mean(base_maes)),
        "model_winners": model_hits,
        "pole_winners": pole_hits,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    sys.argv = ["engine"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("engine", "engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    config = json.loads(Path("config.json").read_text())

    print(f"Building {args.years}")
    races = build(args.years, 8 if args.quick else 99, engine)
    print(f"\n{len(races)} races built\n")
    if len(races) < 6:
        sys.exit("Not enough races.")

    print("=" * 62)
    base = evaluate(races, config)
    print(f"{'ALL FEATURES':<28}{base['races']:>4} races")
    print(f"  XGBoost held-out MAE      {base['xgb_mae']:.3f} positions")
    print(f"  finish = grid baseline    {base['baseline_mae']:.3f} positions")
    print(f"  weighted-score winners    {base['model_winners']}/{base['races']}"
          f" ({100*base['model_winners']/base['races']:.0f}%)")
    print(f"  always-pole winners       {base['pole_winners']}/{base['races']}"
          f" ({100*base['pole_winners']/base['races']:.0f}%)")

    print("\n" + "=" * 62)
    print("ABLATION: drop a group, see if the numbers improve\n")
    for label, group in (("without hand-set constants", HAND_SET),
                         ("without measured features", MEASURED)):
        r = evaluate(races, config, drop=group)
        d_mae = r["xgb_mae"] - base["xgb_mae"]
        d_win = r["model_winners"] - base["model_winners"]
        print(f"{label:<28} MAE {r['xgb_mae']:.3f} ({d_mae:+.3f})   "
              f"winners {r['model_winners']}/{r['races']} ({d_win:+d})")

    print("\n" + "=" * 62)
    print("A negative MAE change means dropping that group made predictions better.")


if __name__ == "__main__":
    main()