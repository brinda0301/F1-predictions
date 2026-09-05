"""
fetch_race_data.py

Builds race data files for the prediction engine straight from the official
timing API. Replaces the FastF1-only version.

Why the rewrite:
  FastF1 reads fresh sessions from livetiming.formula1.com, which blocks many
  hosts and lags for hours after a session ends. The Ergast-compatible API at
  api.jolpi.ca serves qualifying, sprint and race classifications from anywhere,
  usually within an hour or two of a session. FastF1 is now optional and used
  only for FP1 times, which no public API exposes.

Three modes:

  Build a data.py for an upcoming race
      python fetch_race_data.py 12_netherlands --round 12

  Apply grid penalties without hand editing
      python fetch_race_data.py 11_hungary --round 11 \
          --penalty "Lewis Hamilton:3" --penalty "Andrea Kimi Antonelli:3" \
          --pitlane "Sergio Perez"

  Write result.json for a finished race and score it into config.json
      python fetch_race_data.py 11_hungary --round 11 --result --score

Hand-edit fields (WEATHER, CIRCUIT, TYRE_COMPOUNDS, CIRCUIT_HISTORY,
START_PROCEDURE, ENERGY_READINESS) are carried over from an existing data.py
when one is present, so a re-fetch no longer wipes your tuning.
"""

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

API = "https://api.jolpi.ca/ergast/f1"

# Stable API ids mapped to the names used across the repo. Keyed by id, not by
# display name, because the API's display names drift ("RB F1 Team" vs "Racing
# Bulls") while ids do not.
TEAMS = {
    "mclaren": "McLaren",
    "mercedes": "Mercedes",
    "ferrari": "Ferrari",
    "red_bull": "Red Bull",
    "rb": "Racing Bulls",
    "audi": "Audi",
    "alpine": "Alpine",
    "haas": "Haas",
    "williams": "Williams",
    "aston_martin": "Aston Martin",
    "cadillac": "Cadillac",
}

DRIVERS = {
    "norris": "Lando Norris",
    "piastri": "Oscar Piastri",
    "russell": "George Russell",
    "antonelli": "Andrea Kimi Antonelli",
    "hamilton": "Lewis Hamilton",
    "leclerc": "Charles Leclerc",
    "max_verstappen": "Max Verstappen",
    "verstappen": "Max Verstappen",
    "hadjar": "Isack Hadjar",
    "lawson": "Liam Lawson",
    "arvid_lindblad": "Arvid Lindblad",
    "tsunoda": "Yuki Tsunoda",
    "hulkenberg": "Nico Hulkenberg",
    "bortoleto": "Gabriel Bortoleto",
    "gasly": "Pierre Gasly",
    "colapinto": "Franco Colapinto",
    "ocon": "Esteban Ocon",
    "bearman": "Ollie Bearman",
    "albon": "Alex Albon",
    "sainz": "Carlos Sainz",
    "alonso": "Fernando Alonso",
    "stroll": "Lance Stroll",
    "bottas": "Valtteri Bottas",
    "perez": "Sergio Perez",
}

SEASONS = {
    "Andrea Kimi Antonelli": 2, "George Russell": 7, "Lewis Hamilton": 19,
    "Max Verstappen": 11, "Charles Leclerc": 8, "Lando Norris": 7,
    "Oscar Piastri": 4, "Isack Hadjar": 2, "Pierre Gasly": 9,
    "Franco Colapinto": 2, "Carlos Sainz": 11, "Alex Albon": 6,
    "Liam Lawson": 2, "Arvid Lindblad": 0, "Ollie Bearman": 2,
    "Esteban Ocon": 9, "Nico Hulkenberg": 14, "Gabriel Bortoleto": 2,
    "Fernando Alonso": 23, "Lance Stroll": 9, "Valtteri Bottas": 13,
    "Sergio Perez": 15, "Yuki Tsunoda": 6,
}

DEFAULT_START_PROCEDURE = {
    "Mercedes": 0.00, "Ferrari": 0.05, "McLaren": 0.05, "Red Bull": 0.00,
    "Alpine": 0.00, "Racing Bulls": 0.00, "Audi": -0.02, "Haas": 0.02,
    "Williams": -0.02, "Aston Martin": -0.05, "Cadillac": -0.08,
}

DEFAULT_ENERGY_READINESS = {
    "Mercedes": 0.88, "McLaren": 0.85, "Ferrari": 0.80, "Red Bull": 0.78,
    "Racing Bulls": 0.70, "Alpine": 0.60, "Williams": 0.62, "Haas": 0.58,
    "Audi": 0.50, "Aston Martin": 0.45, "Cadillac": 0.40,
}

CARRY_OVER = [
    "START_PROCEDURE", "ENERGY_READINESS", "CIRCUIT",
    "TYRE_COMPOUNDS", "WEATHER", "CIRCUIT_HISTORY",
]


# ---------------------------------------------------------------- API helpers

def api(path):
    """GET a jolpica endpoint and return the MRData payload."""
    url = f"{API}/{path.lstrip('/')}"
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}format=json&limit=100"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)["MRData"]
    except Exception as exc:
        sys.exit(f"API request failed: {url}\n  {exc}")


def driver_name(d):
    return DRIVERS.get(d["driverId"], f"{d['givenName']} {d['familyName']}")


def team_name(c):
    return TEAMS.get(c["constructorId"], c["name"])


def to_seconds(clock):
    """'1:11.163' -> 71.163. Returns None for blanks."""
    if not clock:
        return None
    m = re.match(r"^(?:(\d+):)?(\d+)\.(\d+)$", clock.strip())
    if not m:
        return None
    minutes = int(m.group(1) or 0)
    return round(minutes * 60 + int(m.group(2)) + int(m.group(3)) / 1000, 3)


def best_lap(entry):
    for key in ("Q3", "Q2", "Q1"):
        secs = to_seconds(entry.get(key))
        if secs is not None:
            return secs
    return None


# ---------------------------------------------------------------- API fetches

def fetch_qualifying(year, rnd):
    races = api(f"{year}/{rnd}/qualifying/")["RaceTable"]["Races"]
    if not races:
        sys.exit(
            f"No qualifying data for {year} round {rnd} yet.\n"
            "The API usually publishes within a couple of hours of the session."
        )
    out = []
    for q in races[0]["QualifyingResults"]:
        out.append({
            "driver": driver_name(q["Driver"]),
            "team": team_name(q["Constructor"]),
            "pos": int(q["position"]),
            "q_time": best_lap(q),
        })
    out.sort(key=lambda x: x["pos"])
    return races[0], out


def fetch_sprint(year, rnd):
    races = api(f"{year}/{rnd}/sprint/")["RaceTable"]["Races"]
    if not races:
        return []
    return [
        {
            "pos": int(s["position"]),
            "driver": driver_name(s["Driver"]),
            "team": team_name(s["Constructor"]),
        }
        for s in races[0]["SprintResults"]
    ]


def fetch_results(year, rnd):
    races = api(f"{year}/{rnd}/results/")["RaceTable"]["Races"]
    if not races:
        return []
    out = []
    for r in races[0]["Results"]:
        status = r["status"]
        if status == "Finished":
            clean = "Finished"
        elif "Lap" in status:
            clean = "Lapped"
        else:
            clean = "Retired"
        out.append({
            "pos": int(r["position"]),
            "driver": driver_name(r["Driver"]),
            "team": team_name(r["Constructor"]),
            "status": clean,
        })
    out.sort(key=lambda x: x["pos"])
    return out


def reconcile_name(raw, grid_names):
    """Map a FastF1 driver name onto the name used in GRID.

    FastF1 and the timing API disagree on several drivers: 'Kimi Antonelli' vs
    'Andrea Kimi Antonelli', 'Oliver Bearman' vs 'Ollie Bearman', 'Alexander
    Albon' vs 'Alex Albon'. The engine looks FP1 up by the GRID name, so an
    unreconciled key silently drops that driver to practice_pace 0.3.
    """
    if raw in grid_names:
        return raw
    surname = raw.split()[-1].lower()
    matches = [g for g in grid_names if g.split()[-1].lower() == surname]
    if len(matches) == 1:
        return matches[0]
    return None


def fetch_fp1(year, rnd, grid):
    """FP1 times via FastF1 if installed and reachable. Empty dict otherwise.

    Deliberately all-or-nothing: the engine gives practice_pace 0.3 to any
    driver missing from this table, so a partial table would hand a false
    edge to whoever happens to be listed.
    """
    grid_names = [d["driver"] for d in grid]
    try:
        import fastf1
    except ImportError:
        print("  FP1: FastF1 not installed, skipping")
        return {}
    try:
        cache_dir = Path("data/f1_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        session = fastf1.get_session(year, rnd, "FP1")
        session.load(telemetry=False, weather=False, messages=False)
        if not len(session.results):
            print("  FP1: no data returned, leaving empty")
            return {}
        times, unmatched = {}, []
        for _, row in session.results.iterrows():
            laps = session.laps.pick_drivers(row["Abbreviation"])
            if laps.empty:
                continue
            best = laps["LapTime"].min()
            if best is None or str(best) == "NaT":
                continue
            name = reconcile_name(str(row["FullName"]), grid_names)
            if name is None:
                unmatched.append(str(row["FullName"]))
                continue
            times[name] = round(best.total_seconds(), 3)
        for name in unmatched:
            print(f"  FP1: '{name}' not on the grid, ignored")
        covered = [g for g in grid_names if g in times]
        missing = [g for g in grid_names if g not in times]
        # A driver who sat out FP1 for a rookie run is not slow, but the engine
        # scores any missing name as practice_pace 0.3. At Monza 2026 that cost
        # the pole sitter three points of win probability. Reject the table
        # unless coverage is near-total, and always name who is absent.
        if missing:
            print(f"  FP1: no time for {', '.join(missing)}")
        if len(covered) < len(grid_names) * 0.95:
            print(f"  FP1: only {len(covered)}/{len(grid_names)} grid drivers covered, leaving empty")
            return {}
        return times
    except Exception as exc:
        print(f"  FP1: unavailable ({type(exc).__name__}), leaving empty")
        return {}


# ---------------------------------------------------------------- derivations

def apply_penalties(grid, penalties, pitlane):
    """Rebuild the grid from the qualifying order.

    penalties: {driver: places_dropped}. pitlane: [driver, ...] sent to the back.
    Non-penalised drivers keep relative order and fill from the front, then each
    penalised driver takes their target slot. This matches how the FIA forms a
    grid when several penalties land at once.
    """
    if not penalties and not pitlane:
        return grid

    by_name = {d["driver"]: d for d in grid}
    for name in list(penalties) + list(pitlane):
        if name not in by_name:
            sys.exit(f"Penalty names a driver not on the grid: {name}")

    order = [d["driver"] for d in sorted(grid, key=lambda x: x["pos"])]
    targets = {}
    for name, drop in penalties.items():
        targets[name] = min(order.index(name) + 1 + drop, len(order))

    moved = set(targets) | set(pitlane)
    clean = [n for n in order if n not in moved]
    placed = {slot: name for name, slot in targets.items()}

    final, slot, i = [], 1, 0
    while len(final) < len(order) - len(pitlane):
        if slot in placed:
            final.append(placed[slot])
        elif i < len(clean):
            final.append(clean[i])
            i += 1
        else:
            break
        slot += 1
    final.extend(pitlane)

    out = []
    for pos, name in enumerate(final, 1):
        entry = dict(by_name[name])
        entry["pos"] = pos
        out.append(entry)
    return out


def team_pace_deficit(grid):
    best = {}
    for d in grid:
        if d["q_time"] is None:
            continue
        best[d["team"]] = min(best.get(d["team"], 9e9), d["q_time"])
    if not best:
        return {}
    fastest = min(best.values())
    return dict(sorted(
        ((t, round(v - fastest, 3)) for t, v in best.items()),
        key=lambda x: x[1],
    ))


def driver_experience(grid, prev_results):
    """r1_finish from the previous round, with a documented fallback.

    A driver absent from the previous classification (mid-season stand-in,
    debut) gets the midpoint of the field rather than None. The old version
    wrote None, which serialised to `null` and crashed the import.
    """
    prev = {r["driver"]: r["pos"] for r in prev_results}
    fallback = (len(grid) + 1) // 2
    out, substitutes = {}, []
    for d in sorted(grid, key=lambda x: x["pos"]):
        name = d["driver"]
        if name in prev:
            finish = prev[name]
        else:
            finish = fallback
            substitutes.append(name)
        out[name] = {"f1_seasons": SEASONS.get(name, 5), "r1_finish": finish}
    return out, substitutes


def carry_over_fields(path):
    """Read hand-tuned blocks out of an existing data.py so a re-fetch keeps them."""
    if not path.exists():
        return {}
    namespace = {}
    try:
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    except Exception as exc:
        print(f"  Could not read existing {path.name} ({type(exc).__name__}), using defaults")
        return {}
    return {k: namespace[k] for k in CARRY_OVER if k in namespace}


# ------------------------------------------------------------------- emitting

def literal(obj):
    """Python literal, not JSON. json.dumps writes true/false/null, which are
    not Python names, so the generated module would fail to import."""
    return repr(obj)


def write_data_py(path, race_info, grid, fp1, sprint, experience,
                  deficits, carried, notes):
    L = ['"""']
    L.append(f"{race_info['name']} ({race_info['circuit']}), round {race_info['round']}")
    L.append("Generated by fetch_race_data.py from the official timing API.")
    for note in notes:
        L.append(note)
    L.append('"""')
    L.append("")
    L.append(f"RACE_INFO = {literal(race_info)}")
    L.append("")

    L.append("GRID = [")
    for d in grid:
        L.append(f"    {literal(d)},")
    L.append("]")
    L.append("")

    L.append("FP1_TIMES = {")
    for name, t in sorted(fp1.items(), key=lambda x: x[1]):
        L.append(f"    {literal(name)}: {t},")
    L.append("}")
    L.append("")

    if sprint:
        L.append("SPRINT_RESULT = [")
        for s in sprint:
            L.append(f"    {literal(s)},")
        L.append("]")
    else:
        L.append("SPRINT_RESULT = []")
    L.append("")

    L.append("DRIVER_EXPERIENCE = {")
    for name, info in experience.items():
        L.append(f"    {literal(name)}: {literal(info)},")
    L.append("}")
    L.append("")

    L.append("TEAM_PACE_DEFICIT = {")
    for team, gap in deficits.items():
        L.append(f"    {literal(team)}: {gap},")
    L.append("}")
    L.append("")

    kept = set(carried)
    marker = "  # carried over from previous data.py"

    def block(name, default, comment):
        L.append(comment + (marker if name in kept else ""))
        L.append(f"{name} = {literal(carried.get(name, default))}")
        L.append("")

    block("START_PROCEDURE", DEFAULT_START_PROCEDURE,
          "# Per-team launch quality.")
    block("ENERGY_READINESS", DEFAULT_ENERGY_READINESS,
          "# Per-team 2026 hybrid deployment estimates.")
    block("CIRCUIT", {"type": "balanced", "pit_loss_seconds": 21},
          "# type is 'high_speed' / 'street' / 'balanced'.")
    block("TYRE_COMPOUNDS", {"hardness": 0.5, "one_stop_probability": 0.65},
          "# hardness: 0=softest, 1=hardest.")
    block("WEATHER", {"track_temp_c": 30, "rain_probability": 0.10},
          "# REPLACE with the race day forecast before predicting.")
    block("CIRCUIT_HISTORY", {},
          "# Past results at this circuit, per driver.")

    path.write_text("\n".join(L), encoding="utf-8")
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def score_round(config_path, race_dir, rnd, race_name, results):
    """Append this round's accuracy entry to config.json."""
    pred_path = race_dir / "prediction.json"
    if not pred_path.exists():
        print("  No prediction.json, skipping scoring")
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    history = config.setdefault("accuracy_history", [])
    if any(e.get("round") == rnd for e in history):
        print(f"  Round {rnd} already scored, leaving config alone")
        return

    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    finish = {r["driver"]: r["pos"] for r in results}
    actual = [r["driver"] for r in results[:3]]
    mc = [p["driver"] for p in pred["predictions"][:3]]
    errors = [abs(finish.get(d, len(results)) - (i + 1)) for i, d in enumerate(mc)]

    entry = {
        "round": rnd,
        "race": race_name,
        "predicted_winner": mc[0],
        "predicted_win_pct": pred["predictions"][0]["win_pct"],
        "actual_winner": actual[0],
        "correct": mc[0] == actual[0],
        "podium_predicted": mc,
        "podium_actual": actual,
        "podium_overlap": len(set(mc) & set(actual)),
        "mean_position_error": round(sum(errors) / len(errors), 2),
    }

    xgb = pred.get("xgboost") or {}
    if xgb.get("predictions"):
        xg = [p["driver"] for p in xgb["predictions"][:3]]
        entry["xgb_winner_correct"] = xg[0] == actual[0]
        entry["xgb_podium_overlap"] = len(set(xg) & set(actual))

    history.append(entry)
    history.sort(key=lambda e: e["round"])
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    hits = sum(e["correct"] for e in history)
    xgb_rounds = [e for e in history if "xgb_winner_correct" in e]
    xgb_hits = sum(e["xgb_winner_correct"] for e in xgb_rounds)
    print(f"  Scored R{rnd}: predicted {mc[0]}, won {actual[0]}")
    print(f"  Monte Carlo {hits}/{len(history)}  XGBoost {xgb_hits}/{len(xgb_rounds)}")


# ----------------------------------------------------------------------- main

def parse_penalty(raw):
    if ":" not in raw:
        sys.exit(f'Penalty must look like "Driver Name:3", got: {raw}')
    name, places = raw.rsplit(":", 1)
    return name.strip(), int(places)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="race folder, e.g. 12_netherlands")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--races-dir", default="races")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--penalty", action="append", default=[],
                    metavar='"Driver:places"', help="grid drop, repeatable")
    ap.add_argument("--pitlane", action="append", default=[],
                    metavar='"Driver"', help="pit lane start, repeatable")
    ap.add_argument("--no-fp1", action="store_true", help="skip the FastF1 lookup")
    ap.add_argument("--result", action="store_true",
                    help="write result.json for a finished race")
    ap.add_argument("--score", action="store_true",
                    help="append this round to config accuracy_history")
    args = ap.parse_args()

    races_dir = Path(args.races_dir)
    race_dir = races_dir / args.folder
    race_dir.mkdir(parents=True, exist_ok=True)

    if args.result or args.score:
        print(f"Fetching results for {args.year} round {args.round}")
        results = fetch_results(args.year, args.round)
        if not results:
            sys.exit("No race results published yet.")
        if args.result:
            payload = {"result": [
                {"pos": r["pos"], "driver": r["driver"],
                 "team": "", "status": r["status"]}
                for r in results
            ]}
            out = race_dir / "result.json"
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  Wrote {out} ({len(results)} classified)")
        if args.score:
            score_round(Path(args.config), race_dir,
                        args.round, args.folder.split("_", 1)[-1].title(), results)
        return

    print(f"Fetching {args.year} round {args.round}")
    race, grid = fetch_qualifying(args.year, args.round)
    print(f"  Qualifying: {len(grid)} drivers")

    penalties = dict(parse_penalty(p) for p in args.penalty)
    if penalties or args.pitlane:
        grid = apply_penalties(grid, penalties, args.pitlane)
        for name, drop in penalties.items():
            print(f"  Penalty: {name} drops {drop}")
        for name in args.pitlane:
            print(f"  Pit lane start: {name}")

    sprint = fetch_sprint(args.year, args.round)
    if sprint:
        print(f"  Sprint: {len(sprint)} classified")

    prev = fetch_results(args.year, args.round - 1) if args.round > 1 else []
    experience, substitutes = driver_experience(grid, prev)
    if prev:
        print(f"  Previous round: {len(prev)} finishers read for r1_finish")
    for name in substitutes:
        print(f"  {name} absent from R{args.round - 1}, r1_finish set to field midpoint")

    fp1 = {} if args.no_fp1 else fetch_fp1(args.year, args.round, grid)
    if fp1:
        print(f"  FP1: {len(fp1)} times")

    data_path = race_dir / "data.py"
    carried = carry_over_fields(data_path)
    if carried:
        print(f"  Carried over: {', '.join(sorted(carried))}")

    notes = []
    if penalties or args.pitlane:
        notes.append("Grid is post-penalty.")
    if substitutes:
        notes.append("r1_finish is a field-midpoint placeholder for: "
                     + ", ".join(substitutes))
    if not fp1:
        notes.append("FP1_TIMES left empty on purpose, see fetch_race_data.py.")

    race_info = {
        "name": race["raceName"],
        "circuit": args.folder.split("_", 1)[-1].title(),
        "date": race.get("date", ""),
        "round": args.round,
        "is_sprint_weekend": bool(sprint),
    }

    write_data_py(data_path, race_info, grid, fp1, sprint, experience,
                  team_pace_deficit(grid), carried, notes)
    print(f"  Wrote {data_path}")
    print(f"\nNext: python engine.py {args.folder}")


if __name__ == "__main__":
    main()