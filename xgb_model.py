"""
xgb_model.py

XGBoost v2, live from R17. Replaces the position regressor in engine.py.

What changed and why, each measured in the backtest before going live:

  Data      2022-2025 plus every completed 2026 round, about 2,200 driver rows,
            instead of 2026 alone (352 rows). The v1 model fit 300 trees to
            352 rows and learned noise: at R17 it ranked the pole sitter tenth.
  Target    Did this driver win the race (binary), instead of finishing
            position. The model is scored on win probability, so it now trains
            on win probability. Position labels also mixed retirements into
            the target.
  Features  Six timing-derived inputs that exist for every season, instead of
            eighteen, ten of them hand-set constants that cannot be rebuilt
            for past years:
              gap           qualifying gap to the fastest lap, seconds, capped 2.5
              log_grid      log of starting slot
              teammate_gap  own gap minus teammate's gap, seconds, clipped +-1.5
              sprint_pos    sprint finish on sprint weekends, missing otherwise
              ot            circuit overtaking index (see history_data.py)
              field_gap     gap from this driver to the next-fastest car behind
  Guardrail Monotone constraints: a smaller gap, a better grid slot or a better
            sprint finish can never lower win probability. v1 had no such
            rule and learned that winning the sprint hurt Verstappen.

Usage:
    python xgb_model.py --backtest     # walk-forward test on 2026 R4-R16
    python xgb_model.py --train        # fit on everything up to now, save model
    python xgb_model.py 17_singapore   # print the v2 prediction for a race
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
MODEL_PATH = HERE / "xgb_model.json"
META_PATH = HERE / "xgb_model_meta.json"
FEATURES = ["gap", "log_grid", "teammate_gap", "sprint_pos", "ot", "field_gap"]
MONOTONE = (-1, -1, 0, -1, 0, 1)
PARAMS = dict(n_estimators=250, max_depth=2, learning_rate=0.05,
              min_child_weight=2, subsample=0.9, colsample_bytree=1.0,
              reg_lambda=2.0, objective="binary:logistic",
              monotone_constraints=MONOTONE, random_state=42, verbosity=0)
GAP_CAP = 2.5


def feature_rows(drivers, ot):
    """drivers: list of dicts with name, team, grid, gap (s or None), sprint (or None)."""
    gaps = {d["name"]: (d["gap"] if d["gap"] is not None else GAP_CAP) for d in drivers}
    by_team = {}
    for d in drivers:
        by_team.setdefault(d["team"], []).append(d["name"])
    ordered = sorted(gaps.values())
    rows = []
    for d in drivers:
        g = gaps[d["name"]]
        mates = [m for m in by_team[d["team"]] if m != d["name"]]
        tg = max(-1.5, min(1.5, g - gaps[mates[0]])) if mates else 0.0
        behind = [x for x in ordered if x > g]
        fg = min(1.0, behind[0] - g) if behind else 0.0
        rows.append([g, math.log(max(d["grid"], 1)), tg,
                     float(d["sprint"]) if d.get("sprint") else np.nan, ot, fg])
    return np.array(rows, dtype=float)


def history_races(through_year=2025, live_year=2026, before_round=None):
    """Feature matrices for 2022-2025 plus live-year rounds before `before_round`."""
    import history_data as pb
    hist = pb.load(range(pb.HISTORY_FROM, through_year + 1))
    live = pb.load([live_year])
    allr = hist + live
    for r in allr:
        r["_corr"] = pb.race_order_corr(r)
    out = []
    for r in allr:
        rnd = int(r["label"].split()[1][1:])
        if r["year"] < 2022:
            continue
        if r["year"] == live_year and before_round is not None and rnd >= before_round:
            continue
        if any(v["gap"] is None for v in r["drivers"].values()):
            continue
        ot, _ = pb.overtaking_index(allr, r["circuit"], r["date"])
        ds = [{"name": v["name"], "team": v["team"], "grid": v["grid"], "gap": v["gap"],
               "sprint": v.get("sprint")} for v in r["drivers"].values()]
        X = feature_rows(ds, ot)
        y = np.array([1 if d == r["winner"] else 0 for d in r["drivers"]])
        out.append({"label": r["label"], "year": r["year"], "round": rnd, "X": X, "y": y,
                    "names": [v["name"] for v in r["drivers"].values()],
                    "winner": r["drivers"][r["winner"]]["name"],
                    "grids": [v["grid"] for v in r["drivers"].values()]})
    return out, allr


def fit(races):
    from xgboost import XGBClassifier
    X = np.vstack([r["X"] for r in races])
    y = np.concatenate([r["y"] for r in races])
    return XGBClassifier(**PARAMS).fit(X, y)


def race_probs(model, X):
    p = model.predict_proba(X)[:, 1]
    return p / p.sum()


# Live ------------------------------------------------------------------------

def circuit_index(meta, circuit, before, min_editions=2):
    """Mean grid/finish rank correlation at a circuit over races before `before`.

    Same rule as history_data.overtaking_index, read from the table saved at
    training time so a live prediction needs no network call.
    """
    here = [c for d, c in meta["circuits"].get(circuit, []) if d < before]
    if len(here) >= min_editions:
        return float(np.mean(here))
    allv = [c for rows in meta["circuits"].values() for d, c in rows if d < before]
    return float(np.mean(allv)) if allv else meta["ot_mean"]


def train(live_year=2026):
    import history_data as pb
    races, allr = history_races()
    model = fit(races)
    model.save_model(MODEL_PATH)
    circuits = {}
    for r in allr:
        if r["_corr"] is not None:
            circuits.setdefault(r["circuit"], []).append([r["date"], round(r["_corr"], 4)])
    schedule = {}
    for r in pb.fetch(f"{live_year}/")["RaceTable"]["Races"]:
        schedule[r["round"]] = {"circuit": r["Circuit"]["circuitId"], "date": r["date"]}
    META_PATH.write_text(json.dumps({
        "circuits": circuits,
        "schedule": schedule,
        "ot_mean": round(float(np.mean([c for v in circuits.values() for _, c in v])), 5),
        "features": FEATURES, "monotone": MONOTONE,
        "params": {k: v for k, v in PARAMS.items() if k != "monotone_constraints"},
        "training_races": len(races), "training_rows": int(sum(len(r["y"]) for r in races)),
        "seasons": sorted({r["year"] for r in races}),
        "last_race": races[-1]["label"],
    }, indent=1))
    print(f"Wrote {MODEL_PATH.name}: {len(races)} races, "
          f"{sum(len(r['y']) for r in races)} rows, last {races[-1]['label']}")


def predict(race_data):
    """Win probabilities for a race from data.py. Same shape engine.py publishes."""
    from xgboost import XGBClassifier
    if not MODEL_PATH.exists():
        return {"available": False, "reason": "xgb_model.json missing, run --train"}
    meta = json.loads(META_PATH.read_text())
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    info = race_data.get("RACE_INFO", {})
    ot = meta["ot_mean"]
    sched = meta["schedule"].get(str(info.get("round")), {})
    if sched.get("circuit"):
        ot = circuit_index(meta, sched["circuit"], info.get("date") or sched.get("date", "9999"))

    grid = race_data["GRID"]
    times = [d["q_time"] for d in grid if d.get("q_time")]
    fastest = min(times) if times else None
    sprint = {s["driver"]: s["pos"] for s in race_data.get("SPRINT_RESULT", [])}
    ds = [{"name": d["driver"], "team": d["team"], "grid": d["pos"],
           "gap": min(d["q_time"] - fastest, GAP_CAP) if (fastest and d.get("q_time")) else None,
           "sprint": sprint.get(d["driver"])} for d in grid]
    X = feature_rows(ds, ot)
    p = race_probs(model, X)

    preds = []
    for d, x, v in zip(grid, X, p):
        preds.append({"driver": d["driver"], "team": d["team"], "grid_pos": d["pos"],
                      "quali_gap": round(float(x[0]), 3), "win_prob": round(float(v), 4)})
    preds.sort(key=lambda r: -r["win_prob"])
    for i, r in enumerate(preds, 1):
        r["predicted_position"] = i      # rank by win probability, kept for the dashboard

    contrib = dict(zip(FEATURES, (float(v) for v in model.feature_importances_)))
    return {
        "available": True,
        "version": 2,
        "objective": "win (binary:logistic), monotone",
        "trained_rows": meta["training_rows"],
        "n_races_trained_on": meta["training_races"],
        "seasons": meta["seasons"],
        "n_estimators": PARAMS["n_estimators"],
        "max_depth": PARAMS["max_depth"],
        "overtaking_index": round(ot, 3),
        "predictions": preds,
        "feature_importance": {k: round(v, 4) for k, v in contrib.items()},
    }


# Backtest --------------------------------------------------------------------

def backtest(first=4, last=16):
    """Walk forward through 2026: for round k, train on 2022-2025 + rounds < k."""
    import probscore
    allraces, _ = history_races()
    live = [r for r in allraces if r["year"] == 2026]
    hist = [r for r in allraces if r["year"] < 2026]
    folders = {int(p.name[:2]): p for p in (HERE / "races").iterdir() if p.name[:2].isdigit()}
    rows = []
    for r in live:
        if not first <= r["round"] <= last:
            continue
        train_set = hist + [x for x in live if x["round"] < r["round"]]
        m = fit(train_set)
        p = race_probs(m, r["X"])
        v2 = probscore.score(dict(zip(r["names"], p)), r["winner"])
        pred = json.loads((folders[r["round"]] / "prediction.json").read_text())
        v1 = probscore.round_scores(pred, r["winner"])
        rows.append((r["round"], r["winner"], v1.get("xgb_log_loss"), v2["log_loss"],
                     v1.get("mc_log_loss"), v2["p_winner"], v2["hit"]))
    print(f"{'Rd':<4}{'winner':<24}{'XGB v1':>8}{'XGB v2':>8}{'MC':>8}{'v2 P(win)':>11}")
    for rd, w, a, b, mc, pw, h in rows:
        a_s = f"{a:.2f}" if a is not None else "-"
        print(f"{rd:<4}{w:<24}{a_s:>8}{b:>8.2f}{mc:>8.2f}{100 * pw:>10.0f}%")
    v1 = [a for _, _, a, _, _, _, _ in rows if a is not None]
    v2 = [b for _, _, a, b, _, _, _ in rows if a is not None]
    mc = [m for _, _, a, _, m, _, _ in rows if a is not None]
    print(f"\nmean log loss over {len(v1)} rounds: XGB v1 {np.mean(v1):.3f}   "
          f"XGB v2 {np.mean(v2):.3f}   Monte Carlo {np.mean(mc):.3f}")
    print(f"v2 winners {sum(r[6] for r in rows)}/{len(rows)}")
    return rows


def cv_history(n_folds=5):
    """Grouped k-fold over 2022-2025 races, for tuning without touching 2026."""
    import probscore
    races = [r for r in history_races()[0] if r["year"] < 2026]
    idx = np.arange(len(races))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    lls, hits = [], 0
    for k in range(n_folds):
        test = set(idx[k::n_folds])
        m = fit([r for i, r in enumerate(races) if i not in test])
        for i in test:
            r = races[i]
            s = probscore.score(dict(zip(r["names"], race_probs(m, r["X"]))), r["winner"])
            lls.append(s["log_loss"])
            hits += s["hit"]
    print(f"2022-2025 grouped {n_folds}-fold: log loss {np.mean(lls):.3f}, "
          f"winners {hits}/{len(races)}")
    return float(np.mean(lls))


if __name__ == "__main__":
    arg = sys.argv[1:] or ["--help"]
    if arg == ["--backtest"]:
        cv_history()
        backtest()
    elif arg == ["--train"]:
        train()
    elif arg[0].startswith("--"):
        print(__doc__)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location("engine", HERE / "engine.py")
        engine = importlib.util.module_from_spec(spec)
        sys.argv = ["engine"]
        spec.loader.exec_module(engine)
        out = predict(engine.load_race_data(arg[0]))
        print(json.dumps({k: v for k, v in out.items() if k != "predictions"}, indent=1))
        for r in out["predictions"][:8]:
            print(f"  P{r['grid_pos']:<3}{r['driver']:<26}{100 * r['win_prob']:6.1f}%")
