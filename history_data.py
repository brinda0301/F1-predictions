"""
history_data.py

Race history from the Ergast-compatible timing API (api.jolpi.ca), cached in
.backtest_cache. Feeds XGBoost v2 and the pole baseline.

  load(years)          grid slot, finish, qualifying gap, team and sprint
                       position per driver, per race
  overtaking_index()   mean grid/finish rank correlation at a circuit over
                       races before a date. Near 0.87 the race finishes close
                       to grid order (Monaco). Near 0.5 positions change a lot
                       (Las Vegas). A race never informs its own index.

    python history_data.py --index          # print the circuit index
    python history_data.py --write-prior    # regenerate grid_prior.json
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

_last = [0.0]


def fetch(path, attempts=6):
    """One cached, throttled request to the timing API."""
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
    field = {"results": "Results", "qualifying": "QualifyingResults",
             "sprint": "SprintResults"}[kind]
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
        try:
            sprints = season(year, "sprint") if year >= 2021 else {}
        except Exception:
            sprints = {}
        for rnd in sorted(res):
            r = res[rnd]
            rows = r["rows"]
            n = len(rows)
            drivers = {}
            for x in rows:
                g = int(x["grid"])
                drivers[x["Driver"]["driverId"]] = {
                    "name": f'{x["Driver"]["givenName"]} {x["Driver"]["familyName"]}',
                    "team": x["Constructor"]["constructorId"],
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
            sp = sprints.get(rnd) if sprints else None
            if sp:
                for x in sp["rows"]:
                    d = x["Driver"]["driverId"]
                    if d in drivers:
                        drivers[d]["sprint"] = int(x["position"])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--write-prior", action="store_true")
    ap.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    args = ap.parse_args()
    history = load(range(HISTORY_FROM, max(args.seasons) + 1))
    for r in history:
        r["_corr"] = race_order_corr(r)

    if args.index:
        circuits = sorted({r["circuit"] for r in history})
        rows = sorted(((c,) + overtaking_index(history, c, "9999") for c in circuits),
                      key=lambda x: -x[1])
        print("Circuit overtaking index (1.0 = finishes in grid order)")
        for c, v, n in rows:
            print(f"  {c:<16} {v:.2f}  ({n} races)")

    if args.write_prior:
        races = [r for r in history if r["year"] in args.seasons]
        pr = probscore.grid_prior((list(v["grid"] for v in r["drivers"].values()),
                                   r["drivers"][r["winner"]]["grid"]) for r in races)
        Path(probscore.PRIOR_PATH).write_text(json.dumps(
            {"seasons": args.seasons, "races": len(races), "prior": pr}, indent=2))
        print(f"Wrote {probscore.PRIOR_PATH}: pole {pr[1]:.3f}, P2 {pr[2]:.3f}, P3 {pr[3]:.3f}")

    if not (args.index or args.write_prior):
        print(__doc__)


if __name__ == "__main__":
    main()
