"""
logit_model.py

The four-input conditional logit from prob_backtest.py, run live on a race.

    python logit_model.py --train     # fit on 2022-2025, save logit_model.json
    python logit_model.py 17_<race>   # print the logit prediction for a race

engine.predict() calls predict() and writes the result into prediction.json
under "logit", beside Monte Carlo and XGBoost. Training needs the API (or the
.backtest_cache folder); prediction reads only logit_model.json and data.py.

Inputs per driver: log of grid slot, qualifying gap to pole in seconds (capped
at 2.5), and both scaled by the circuit overtaking index. The index for a race
uses only races at that circuit run before it, so re-running an old race gives
the same answer it gave on the day.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).with_name("logit_model.json")


def train(path=MODEL_PATH, seasons=(2022, 2023, 2024, 2025), live_year=2026):
    import prob_backtest as pb

    history = pb.load(range(pb.HISTORY_FROM, max(seasons) + 1))
    try:
        history += pb.load([live_year])
    except Exception as exc:
        print(f"  {live_year} results unavailable ({exc}), index uses {max(seasons)} and earlier")
    for r in history:
        r["_corr"] = pb.race_order_corr(r)
    pre = [r for r in history if r["year"] <= max(seasons)]
    ot_mean = float(np.mean([r["_corr"] for r in pre if r["_corr"] is not None]))

    evals = [r for r in pre if r["year"] in seasons
             and all(v["gap"] is not None for v in r["drivers"].values())]
    for r in evals:
        r["ot"], _ = pb.overtaking_index(history, r["circuit"], r["date"])
        r["names"], r["X"] = pb.design(r, r["ot"], ot_mean)
        r["wi"] = r["names"].index(r["winner"])
    w = pb.fit_logit([(r["X"], r["wi"]) for r in evals])

    circuits = {}
    for r in history:
        if r["_corr"] is not None:
            circuits.setdefault(r["circuit"], []).append([r["date"], round(r["_corr"], 4)])

    schedule = {}
    try:
        for r in pb.fetch(f"{live_year}/")["RaceTable"]["Races"]:
            schedule[r["round"]] = {"circuit": r["Circuit"]["circuitId"], "date": r["date"]}
    except Exception as exc:
        print(f"  {live_year} schedule unavailable ({exc})")

    model = {
        "features": pb.FEATURES,
        "weights": [round(float(x), 5) for x in w],
        "ot_mean": round(ot_mean, 5),
        "gap_cap": pb.GAP_CAP,
        "trained_on": list(seasons),
        "training_races": len(evals),
        "circuits": circuits,
        "schedule": schedule,
    }
    Path(path).write_text(json.dumps(model, indent=1), encoding="utf-8")
    print(f"Wrote {path}: {len(evals)} training races, weights "
          + ", ".join(f"{k} {v:+.3f}" for k, v in zip(model["features"], model["weights"])))
    return model


def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def circuit_index(model, circuit, before, min_editions=2):
    """Same rule as prob_backtest.overtaking_index, from the saved table."""
    here = [c for d, c in model["circuits"].get(circuit, []) if d < before]
    if len(here) >= min_editions:
        return float(np.mean(here)), len(here)
    allv = [c for rows in model["circuits"].values() for d, c in rows if d < before]
    return (float(np.mean(allv)) if allv else model["ot_mean"]), len(here)


def predict(race_data, model=None):
    """Win probabilities for every driver on the grid.

    `race_data` is the dict engine.load_race_data returns (GRID, RACE_INFO).
    Returns {"available": False, "reason": ...} rather than raising, so a
    missing model never blocks the Monte Carlo prediction.
    """
    model = model or load_model()
    if not model:
        return {"available": False, "reason": "logit_model.json missing, run --train"}

    info = race_data.get("RACE_INFO", {})
    sched = model["schedule"].get(str(info.get("round")), {})
    circuit = sched.get("circuit")
    date = info.get("date") or sched.get("date") or "9999-12-31"
    if circuit:
        ot, editions = circuit_index(model, circuit, date)
    else:
        ot, editions = model["ot_mean"], 0
    otc = ot - model["ot_mean"]

    grid = race_data["GRID"]
    times = [d["q_time"] for d in grid if d.get("q_time")]
    pole = min(times) if times else None
    cap = model["gap_cap"]

    rows, X = [], []
    for d in grid:
        gap = min(d["q_time"] - pole, cap) if (pole and d.get("q_time")) else cap
        lg = math.log(max(d["pos"], 1))
        X.append([lg, gap, lg * otc, gap * otc])
        rows.append({"driver": d["driver"], "team": d["team"], "grid_pos": d["pos"],
                     "quali_gap": round(gap, 3)})
    s = np.array(X) @ np.array(model["weights"])
    p = np.exp(s - s.max())
    p /= p.sum()
    for r, v in zip(rows, p):
        r["win_prob"] = round(float(v), 4)
    rows.sort(key=lambda r: -r["win_prob"])

    return {
        "available": True,
        "circuit": circuit,
        "overtaking_index": round(ot, 3),
        "circuit_editions": editions,
        "trained_on": model["trained_on"],
        "weights": dict(zip(model["features"], model["weights"])),
        "predictions": rows,
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--train"]:
        train()
    elif len(sys.argv) == 2:
        import importlib.util
        spec = importlib.util.spec_from_file_location("engine", Path(__file__).with_name("engine.py"))
        engine = importlib.util.module_from_spec(spec)
        folder = sys.argv[1]
        sys.argv = ["engine"]
        spec.loader.exec_module(engine)
        out = predict(engine.load_race_data(folder))
        if not out["available"]:
            sys.exit(out["reason"])
        print(f"{folder}: circuit {out['circuit']}, overtaking index "
              f"{out['overtaking_index']} over {out['circuit_editions']} earlier races")
        for r in out["predictions"][:8]:
            print(f"  P{r['grid_pos']:<3}{r['driver']:<26}{100 * r['win_prob']:6.1f}%"
                  f"   gap {r['quali_gap']:.3f}s")
    else:
        print(__doc__)
