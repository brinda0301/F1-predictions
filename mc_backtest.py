"""
mc_backtest.py

Scores the Monte Carlo engine on win probability (log loss) so its settings are
chosen by measurement, not by hand. Added R17.

Two test sets:
  2026      the 16 completed 2026 rounds, from each race's data.py, with their
            real hand-set inputs (circuit, weather, history). Uses the current
            config weights, which were calibrated through R16, so this set
            flatters the engine slightly.
  2024-25   48 races rebuilt from the timing API (race_data below). Hand-set blocks
            sit at defaults, so this isolates the timing features.

Each setting runs with fewer simulations than live (default 3,000) to keep a
full sweep to a few minutes. Same seed for every setting.

    python mc_backtest.py              # both sets
    python mc_backtest.py --only2026
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.argv, _argv = ["engine"], sys.argv
import engine
import probscore
sys.argv = _argv


# Race rebuilder for 2024-2025, moved from the old backtest.py at R17 ----------
# Rebuilds each race in the engine's data.py shape from the timing API.
# Hand-set blocks (circuit, tyres, weather, history) sit at defaults.

import history_data

TEAMS = {
    "mclaren": "McLaren", "mercedes": "Mercedes", "ferrari": "Ferrari",
    "red_bull": "Red Bull", "rb": "Racing Bulls", "sauber": "Audi",
    "audi": "Audi", "alpine": "Alpine", "haas": "Haas",
    "williams": "Williams", "aston_martin": "Aston Martin",
    "cadillac": "Cadillac",
}


def get(path):
    return history_data.fetch(path + "?limit=100")


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



# Relative to the live settings in config.json and engine.py. At R17 the
# current settings scored best across all 64 races combined (log loss 1.355).
SETTINGS = [
    ("current settings", dict(recovery="none", temp_scale=1.0)),
    ("recovery bonus back on", dict(recovery="v1", temp_scale=1.0)),
    ("temperature x1.4", dict(recovery="none", temp_scale=1.4)),
    ("temperature x0.7", dict(recovery="none", temp_scale=0.7)),
]


def races_2026():
    out = []
    for f in engine.get_race_folders():
        res = Path("races") / f / "result.json"
        if not f[:2].isdigit() or not res.exists():
            continue
        r = json.loads(res.read_text())["result"]
        out.append((f, engine.load_race_data(f), r[0]["driver"]))
    return out


def races_hist(years=(2024, 2025)):
    out = []
    for y in years:
        n = len(get(f"{y}/")["RaceTable"]["Races"])
        for rnd in range(1, n + 1):
            try:
                rd = race_data(y, rnd, engine)
            except Exception:
                continue
            if rd:
                out.append((f"{y} R{rnd}", rd, min(rd["_RESULT"], key=rd["_RESULT"].get)))
    return out


def run(races, cfg, n_sims):
    print(f"  {'setting':<30}{'log loss':>10}{'winners':>10}{'avg P(win)':>12}")
    best = None
    for label, kw in SETTINGS:
        sc = []
        for _, rd, win in races:
            res = engine.simulate(rd, cfg, n_sims=n_sims, **kw)
            sc.append(probscore.score({r["driver"]: r["win_pct"] for r in res}, win))
        ll = np.mean([s["log_loss"] for s in sc])
        print(f"  {label:<30}{ll:>10.3f}{sum(s['hit'] for s in sc):>6}/{len(sc):<3}"
              f"{100 * np.mean([s['p_winner'] for s in sc]):>11.1f}%")
        if best is None or ll < best[1]:
            best = (label, ll)
    print(f"  best: {best[0]}")


if __name__ == "__main__":
    cfg = engine.load_config()
    n = 3000
    print(f"2026 rounds 1-16, {n} sims per race")
    run(races_2026(), cfg, n)
    if "--only2026" not in sys.argv:
        print(f"\n2024-2025 rebuilt races, {n} sims per race")
        run(races_hist(), cfg, n)
