"""
prob_backtest.py

Scores win probabilities, not winner picks, over past seasons, and tests
whether information beyond the grid improves them.

Three models, all scored leave-one-race-out on log loss and Brier score
(see probscore.py):

  grid      P(win | starting slot), estimated from the other races. This is
            "always pick pole" written as a probability. The bar to beat.
  engine    The live engine's weighted feature score through its softmax,
            2024-2025 only (the seasons backtest.py builds). The Monte Carlo
            layer is not run: 100k simulations per race in a Python loop is
            too slow for 48 races, so this scores the distribution the
            simulation starts from.
  logit     A conditional logit (softmax across the drivers in each race) on
            four inputs:
              log_grid       log of starting slot
              gap            qualifying gap to pole, seconds, capped at 2.5
              log_grid x ot  grid effect scaled by the circuit's overtaking index
              gap x ot       pace effect scaled by the same

Circuit overtaking index (`ot`): Spearman rank correlation between grid and
finish among classified finishers, averaged over every earlier race at that
circuit from 2014 on. Near 1.0 means the race finishes in grid order (Monaco,
Hungary). Lower means positions change (Monza, Spa). Only races before the
target race are used, so a race never informs its own index.

Usage:
    python prob_backtest.py                       # evaluate 2022-2025
    python prob_backtest.py --eval 2024 2025      # seasons to score
    python prob_backtest.py --write-prior         # save grid_prior.json
    python prob_backtest.py --index               # print the circuit index
"""

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

import probscore

CACHE = Path(".backtest_cache")
API = "https://api.jolpi.ca/ergast/f1"
HISTORY_FROM = 2014
GAP_CAP = 2.5
FEATURES = ["log_grid", "gap", "log_grid_x_ot", "gap_x_ot"]

_last = [0.0]


def fetch(path, attempts=6):
    """One cached, throttled request. Same cache directory as backtest.py."""
    CACHE.mkdir(exist_ok=True)
    key = CACHE / ("pb_" + path.replace("/", "_").replace("?", "_").replace("&", "_") + ".json")
    if key.exists():
        return json.loads(key.read_text())
    sep = "&" if "?" in path else "?"
    url = f"{API}/{path}{sep}format=json"
    delay = 2.0
    for attempt in range(attempts):
        gap = 0.35 - (time.time() - _last[0])
        if gap > 0:
            time.sleep(gap)
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.load(r)["MRData"]
            key.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(url)


def season(year, kind):
    """Every race of a season for `kind` in ("results", "qualifying").

    The API pages at 100 rows and splits a race across pages, so rows are
    merged by round.
    """
    field = "Results" if kind == "results" else "QualifyingResults"
    races, offset = {}, 0
    while True:
        data = fetch(f"{year}/{kind}/?limit=100&offset={offset}")
        for r in data["RaceTable"]["Races"]:
            rnd = int(r["round"])
            entry = races.setdefault(rnd, {
                "year": year, "round": rnd, "name": r["raceName"],
                "circuit": r["Circuit"]["circuitId"], "date": r["date"], "rows": []})
            entry["rows"].extend(r.get(field, []))
        offset += 100
        if offset >= int(data["total"]):
            break
    return races


def secs(clock):
    if not clock:
        return None
    parts = clock.strip().split(":")
    try:
        return int(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[0])
    except ValueError:
        return None


def load(years):
    """Races as dicts: grid slot, finish, quali gap per driverId."""
    out = []
    for year in years:
        res = season(year, "results")
        qual = season(year, "qualifying") if year >= 2018 else {}
        for rnd in sorted(res):
            r = res[rnd]
            rows = r["rows"]
            n = len(rows)
            drivers = {}
            for x in rows:
                g = int(x["grid"])
                drivers[x["Driver"]["driverId"]] = {
                    "name": f'{x["Driver"]["givenName"]} {x["Driver"]["familyName"]}',
                    "grid": g if g > 0 else n,          # pit-lane start goes to the back
                    "finish": int(x["position"]),
                    "classified": x["positionText"].isdigit(),
                    "gap": None,
                }
            q = qual.get(rnd)
            if q:
                best = {}
                for x in q["rows"]:
                    laps = [secs(x.get(k)) for k in ("Q1", "Q2", "Q3")]
                    laps = [l for l in laps if l]
                    if laps:
                        best[x["Driver"]["driverId"]] = min(laps)
                if best:
                    pole = min(best.values())
                    for d, v in drivers.items():
                        t = best.get(d)
                        v["gap"] = min(t - pole, GAP_CAP) if t else GAP_CAP
            winner = min(drivers, key=lambda d: drivers[d]["finish"])
            out.append({"label": f"{year} R{rnd} {r['name']}", "year": year,
                        "circuit": r["circuit"], "date": r["date"],
                        "drivers": drivers, "winner": winner})
    return out


# Overtaking index ------------------------------------------------------------

def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def race_order_corr(race):
    fin = [v for v in race["drivers"].values() if v["classified"]]
    if len(fin) < 8:
        return None
    return spearman([v["grid"] for v in fin], [v["finish"] for v in fin])


def overtaking_index(history, circuit, before, min_editions=2):
    """Mean grid/finish rank correlation at `circuit` over races before `before`.

    Falls back to the mean over all circuits when a track has fewer than
    `min_editions` earlier races (a new venue such as Las Vegas in 2023).
    """
    vals = [c for r in history if r["date"] < before
            for c in [r["_corr"]] if c is not None]
    here = [r["_corr"] for r in history
            if r["circuit"] == circuit and r["date"] < before and r["_corr"] is not None]
    if len(here) >= min_editions:
        return float(np.mean(here)), len(here)
    return (float(np.mean(vals)) if vals else 0.6), len(here)


# Conditional logit -----------------------------------------------------------

def design(race, ot, ot_mean):
    names = list(race["drivers"])
    otc = ot - ot_mean
    rows = []
    for d in names:
        v = race["drivers"][d]
        lg = math.log(max(v["grid"], 1))
        gap = v["gap"] if v["gap"] is not None else GAP_CAP
        rows.append([lg, gap, lg * otc, gap * otc])
    return names, np.array(rows)


def fit_logit(races, l2=0.5, iters=400, lr=0.5):
    """Maximise sum of log P(winner) with softmax across each race's field."""
    w = np.zeros(races[0][0].shape[1])
    for _ in range(iters):
        grad = l2 * w
        for X, wi in races:
            s = X @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            grad -= (X[wi] - p @ X)
        w -= lr * grad / len(races)
    return w


def logit_probs(names, X, w):
    s = X @ w
    p = np.exp(s - s.max())
    p /= p.sum()
    return dict(zip(names, p))


# Engine ----------------------------------------------------------------------

def engine_probs(years):
    """Engine softmax probabilities for 2024-2025, keyed by 'year R<round>'."""
    import importlib.util
    import sys
    import backtest
    sys.argv = ["engine"]
    spec = importlib.util.spec_from_file_location("engine", "engine.py")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    cfg = json.loads(Path("config.json").read_text())
    weights = cfg["weights"]
    temp = cfg["regulation_params"].get("track_type_temperatures", {}).get(
        "balanced", cfg["regulation_params"]["softmax_temperature"])
    out = {}
    for year in years:
        n = len(backtest.get(f"{year}/")["RaceTable"]["Races"])
        for rnd in range(1, n + 1):
            try:
                rd = backtest.race_data(year, rnd, engine)
            except Exception:
                continue
            if rd is None:
                continue
            scores = {d["driver"]: sum(v * weights.get(k, 0) for k, v in
                                       engine.build_features(d, rd).items())
                      for d in rd["GRID"]}
            arr = np.array(list(scores.values()))
            e = np.exp((arr - arr.max()) / temp)
            e /= e.sum()
            winner = min(rd["_RESULT"], key=rd["_RESULT"].get)
            out[(year, rnd)] = (dict(zip(scores, e)), winner)
    return out


# Live 2026 ------------------------------------------------------------------

def match_name(name, drivers):
    """Engine display name -> API driverId. Full name first, then surname."""
    for d, v in drivers.items():
        if v["name"] == name:
            return d
    sur = name.split()[-1].lower()
    hits = [d for d, v in drivers.items() if v["name"].split()[-1].lower() == sur]
    return hits[0] if len(hits) == 1 else None


def live(history, evals, ot_mean, year=2026):
    """Score the published 2026 predictions against grid and logit.

    The logit and grid prior are fit on 2022-2025 only, so every 2026 race is
    out of sample for them. The Monte Carlo and XGBoost numbers are read from
    each round's prediction.json, which was committed before the race.
    """
    prior = probscore.grid_prior((list(v["grid"] for v in r["drivers"].values()),
                                  r["drivers"][r["winner"]]["grid"]) for r in evals)
    w = fit_logit([(r["X"], r["wi"]) for r in evals])
    season_races = load([year])
    for r in season_races:
        r["_corr"] = race_order_corr(r)
    pool = history + season_races
    folders = {int(p.name.split("_")[0]): p for p in Path("races").iterdir()
               if p.is_dir() and p.name[:2].isdigit()}

    rows = []
    for r in season_races:
        rnd = int(r["label"].split()[1][1:])
        pred_path = folders.get(rnd, Path("-")) / "prediction.json"
        if not pred_path.exists() or any(v["gap"] is None for v in r["drivers"].values()):
            continue
        pred = json.loads(pred_path.read_text())
        ot, _ = overtaking_index(pool, r["circuit"], r["date"])
        names, X = design(r, ot, ot_mean)
        mc = {}
        for p in pred["predictions"]:
            d = match_name(p["driver"], r["drivers"])
            if d:
                mc[d] = p["win_pct"]
        xgb = {}
        for p in (pred.get("xgboost") or {}).get("predictions", []):
            d = match_name(p["driver"], r["drivers"])
            if d and p.get("win_prob") is not None:
                xgb[d] = p["win_prob"]
        win = r["winner"]
        rows.append({
            "round": rnd, "winner": r["drivers"][win]["name"],
            "start": r["drivers"][win]["grid"],
            "grid": probscore.score(probscore.grid_probs(
                {d: v["grid"] for d, v in r["drivers"].items()}, prior), win),
            "logit": probscore.score(logit_probs(names, X, w), win),
            "mc": probscore.score(mc, win),
            "xgb": probscore.score(xgb, win) if xgb else None,
        })

    print("\n" + "=" * 78)
    print(f"LIVE {year}, out of sample. P(actual winner) per model")
    print(f"  {'Rd':<4}{'winner':<24}{'grid':>5}{'MC':>8}{'XGB':>8}{'grid':>8}{'logit':>8}")
    for x in rows:
        xg = f"{100 * x['xgb']['p_winner']:.0f}%" if x["xgb"] else "-"
        print(f"  {x['round']:<4}{x['winner']:<24}{'P' + str(x['start']):>5}"
              f"{100 * x['mc']['p_winner']:>7.0f}%{xg:>8}"
              f"{100 * x['grid']['p_winner']:>7.0f}%{100 * x['logit']['p_winner']:>7.0f}%")
    print()
    for k, lab in (("mc", "Monte Carlo"), ("grid", "grid"), ("logit", "logit")):
        summarise(lab, [x[k] for x in rows])
    xr = [x for x in rows if x["xgb"]]
    print(f"  on the {len(xr)} rounds XGBoost ran:")
    for k, lab in (("xgb", "XGBoost"), ("mc", "Monte Carlo"), ("grid", "grid"), ("logit", "logit")):
        summarise(lab, [x[k] for x in xr])
    for k, lab in (("mc", "Monte Carlo"), ("logit", "logit")):
        d, lo, hi = paired_ci([x[k]["log_loss"] for x in rows], [x["grid"]["log_loss"] for x in rows])
        print(f"  {lab} minus grid, log loss {d:+.3f} (90% CI {lo:+.3f} to {hi:+.3f})")
    return rows


# Evaluation ------------------------------------------------------------------

def summarise(label, scores):
    ll = np.mean([s["log_loss"] for s in scores])
    br = np.mean([s["brier"] for s in scores])
    hits = sum(s["hit"] for s in scores)
    pw = np.mean([s["p_winner"] for s in scores])
    print(f"  {label:<14} log loss {ll:.3f}   Brier {br:.3f}   "
          f"winners {hits}/{len(scores)} ({100 * hits / len(scores):.0f}%)   "
          f"avg P(winner) {100 * pw:.1f}%")
    return ll


def paired_ci(a, b, n_boot=4000, seed=0):
    """Bootstrap 90% interval for mean(a - b) over races."""
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(means, 5), np.percentile(means, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument("--write-prior", action="store_true")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--no-engine", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="score the 2026 rounds out of sample: Monte Carlo, XGBoost, grid, logit")
    args = ap.parse_args()

    years = list(range(HISTORY_FROM, max(args.eval) + 1))
    print(f"Loading {years[0]}-{years[-1]}")
    history = load(years)
    for r in history:
        r["_corr"] = race_order_corr(r)
    ot_mean = float(np.mean([r["_corr"] for r in history if r["_corr"] is not None]))

    evals = [r for r in history if r["year"] in args.eval
             and all(v["gap"] is not None for v in r["drivers"].values())]
    for r in evals:
        r["ot"], r["ot_n"] = overtaking_index(history, r["circuit"], r["date"])
        r["names"], r["X"] = design(r, r["ot"], ot_mean)
        r["wi"] = r["names"].index(r["winner"])
    print(f"{len(evals)} races scored, {args.eval[0]}-{args.eval[-1]}\n")

    if args.index:
        latest = {}
        for r in history:
            latest[r["circuit"]] = r["date"]
        rows = sorted(((c,) + overtaking_index(history, c, "9999") for c in latest),
                      key=lambda x: -x[1])
        print("Circuit overtaking index (1.0 = finishes in grid order)")
        for c, v, n in rows:
            print(f"  {c:<16} {v:.2f}  ({n} races)")
        print()

    if args.write_prior:
        pr = probscore.grid_prior((list(v["grid"] for v in r["drivers"].values()),
                                   r["drivers"][r["winner"]]["grid"]) for r in evals)
        Path(probscore.PRIOR_PATH).write_text(json.dumps(
            {"seasons": args.eval, "races": len(evals), "prior": pr}, indent=2))
        print(f"Wrote {probscore.PRIOR_PATH}: pole {pr[1]:.3f}, P2 {pr[2]:.3f}, "
              f"P3 {pr[3]:.3f}\n")

    # Ablation: which inputs earn their place. "log_grid only" is the grid
    # baseline with a smooth curve instead of a per-slot table, so any gain the
    # full logit has over it comes from the new information, not the smoothing.
    variants = {
        "log_grid only": [0],
        "+ pole gap": [0, 1],
        "+ overtaking": [0, 2],
        "all four": [0, 1, 2, 3],
    }
    var_s = {k: [] for k in variants}

    grid_s, logit_s, weights = [], [], []
    for i, hold in enumerate(evals):
        train = [r for j, r in enumerate(evals) if j != i]
        prior = probscore.grid_prior((list(v["grid"] for v in r["drivers"].values()),
                                      r["drivers"][r["winner"]]["grid"]) for r in train)
        gp = probscore.grid_probs({d: v["grid"] for d, v in hold["drivers"].items()}, prior)
        grid_s.append(probscore.score(gp, hold["winner"]))
        w = fit_logit([(r["X"], r["wi"]) for r in train])
        weights.append(w)
        logit_s.append(probscore.score(logit_probs(hold["names"], hold["X"], w), hold["winner"]))
        for k, cols in variants.items():
            wv = w if cols == [0, 1, 2, 3] else fit_logit([(r["X"][:, cols], r["wi"]) for r in train])
            var_s[k].append(probscore.score(
                logit_probs(hold["names"], hold["X"][:, cols], wv), hold["winner"]))

    print("=" * 78)
    print("LEAVE-ONE-RACE-OUT, lower log loss and Brier are better")
    base = summarise("grid", grid_s)
    summarise("logit", logit_s)
    d, lo, hi = paired_ci([s["log_loss"] for s in logit_s], [s["log_loss"] for s in grid_s])
    print(f"\n  logit minus grid, log loss: {d:+.3f}  (90% CI {lo:+.3f} to {hi:+.3f})")
    print(f"  negative means the logit beats the grid baseline")

    print("\n  ABLATION, logit inputs:")
    ref = [s["log_loss"] for s in var_s["log_grid only"]]
    for k in variants:
        summarise(k, var_s[k])
        if k != "log_grid only":
            d, lo, hi = paired_ci([s["log_loss"] for s in var_s[k]], ref)
            print(f"           vs log_grid only {d:+.3f}  (90% CI {lo:+.3f} to {hi:+.3f})")

    w = np.mean(weights, axis=0)
    print("\n  logit weights (mean over folds):")
    for k, v in zip(FEATURES, w):
        print(f"    {k:<16} {v:+.3f}")

    if not args.no_engine:
        eng_years = [y for y in args.eval if y in (2024, 2025)]
        if eng_years:
            print("\n" + "=" * 78)
            print(f"ENGINE vs grid vs logit on the same races, {eng_years}")
            ep = engine_probs(eng_years)
            sub = [(i, r) for i, r in enumerate(evals) if (r["year"], int(r["label"].split()[1][1:])) in ep]
            eng_s = []
            for i, r in sub:
                probs, winner = ep[(r["year"], int(r["label"].split()[1][1:]))]
                eng_s.append(probscore.score(probs, winner))
            summarise("grid", [grid_s[i] for i, _ in sub])
            summarise("engine", eng_s)
            summarise("logit", [logit_s[i] for i, _ in sub])

    if args.live:
        live(history, evals, ot_mean)

    print("\n" + "=" * 78)
    print("Reference: a uniform guess over 20 cars scores log loss 3.00")


if __name__ == "__main__":
    main()
