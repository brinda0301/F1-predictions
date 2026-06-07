"""
fetch_race_data.py

Pulls qualifying, practice, and sprint data from FastF1 and generates
a data.py file in your engine's schema. Run this after qualifying ends.

Usage from PowerShell (in your repo root):
    python fetch_race_data.py 06_monaco --year 2026 --round 8 --circuit Monaco

The --round and --circuit args use F1's official numbering. FastF1 needs them
to fetch the right session. The 06_monaco arg is your local folder name.

What the script auto-fetches:
    GRID                 from qualifying classification
    FP1_TIMES           from FP1 session
    SPRINT_RESULT       from sprint race (if sprint weekend)
    TEAM_PACE_DEFICIT   computed from quali times
    DRIVER_EXPERIENCE   r1_finish read from previous race result.json
    CIRCUIT             length and laps from session info

What you still hand-edit after the script runs:
    WEATHER             race day forecast (FastF1 has session weather, not forecasts)
    CIRCUIT_HISTORY     past results at this circuit (FastF1 has it, but adding logic later)
    TYRE_COMPOUNDS      compound choice and strategy estimates
    START_PROCEDURE     per-team launch quality (your domain knowledge)
    ENERGY_READINESS    per-team 2026 hybrid management estimates

The script leaves these fields with sensible defaults you can review.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import fastf1
    from fastf1.core import DataNotLoadedError
except ImportError:
    print("FastF1 not installed. Run: pip install fastf1")
    sys.exit(1)


# Map FastF1 team names to your engine's team names.
# Update this dict when F1 team names change between seasons.
TEAM_NAME_MAP = {
    "Mercedes":             "Mercedes",
    "Red Bull Racing":      "Red Bull",
    "Red Bull":             "Red Bull",
    "Ferrari":              "Ferrari",
    "McLaren":              "McLaren",
    "Alpine":               "Alpine",
    "Racing Bulls":         "Racing Bulls",
    "RB":                   "Racing Bulls",     # 2025 naming
    "Visa Cash App RB":     "Racing Bulls",
    "Aston Martin":         "Aston Martin",
    "Williams":             "Williams",
    "Audi":                 "Audi",
    "Sauber":               "Audi",              # 2025 -> Audi rebrand
    "Kick Sauber":          "Audi",
    "Stake F1 Team Kick Sauber": "Audi",
    "Haas F1 Team":         "Haas",
    "Haas":                 "Haas",
    "Cadillac":             "Cadillac",
}

# Map FastF1 driver names to the exact names your past data.py files use.
# Critical because XGBoost training links drivers by name across races.
# Add entries here whenever you spot a mismatch.
DRIVER_NAME_MAP = {
    "Kimi Antonelli":      "Andrea Kimi Antonelli",
    "Alexander Albon":     "Alex Albon",
    "Oliver Bearman":      "Ollie Bearman",
    "Nico Hülkenberg":     "Nico Hulkenberg",
    "Sergio Pérez":        "Sergio Perez",
    # The rest pass through unchanged
}


def normalise_driver(name: str) -> str:
    """Convert FastF1's driver name to your engine's naming."""
    return DRIVER_NAME_MAP.get(name, name)


def setup_cache():
    """Create a local cache folder for FastF1. Speeds up repeat queries."""
    cache_dir = Path(".fastf1_cache")
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def normalise_team(team_name: str) -> str:
    """Convert FastF1's team name to your engine's team naming."""
    return TEAM_NAME_MAP.get(team_name, team_name)


def fetch_grid(year: int, gp: str) -> list[dict]:
    """Fetch qualifying results and return GRID in engine schema."""
    session = fastf1.get_session(year, gp, "Q")
    session.load()
    results = session.results.sort_values("Position")

    grid = []
    for _, row in results.iterrows():
        # Use the best Q3 time, fall back to Q2, then Q1.
        q_time = None
        for col in ["Q3", "Q2", "Q1"]:
            if col in row and not _is_null_time(row[col]):
                q_time = _time_to_seconds(row[col])
                break
        # Status DSQ means quali time is invalid (the Hadjar Miami bug).
        if row.get("Status") and "Disqualified" in str(row["Status"]):
            q_time = None

        grid.append({
            "driver": normalise_driver(str(row["FullName"])),
            "team": normalise_team(str(row["TeamName"])),
            "pos": int(row["Position"]),
            "q_time": q_time,
        })
    return grid


def fetch_fp1_times(year: int, gp: str) -> dict:
    """Fetch FP1 fastest lap per driver."""
    try:
        session = fastf1.get_session(year, gp, "FP1")
        session.load()
    except DataNotLoadedError:
        print("FP1 data not available. Skipping FP1_TIMES.")
        return {}

    fp1_times = {}
    for driver_code in session.drivers:
        try:
            driver_laps = session.laps.pick_drivers(driver_code)
            fastest = driver_laps.pick_fastest()
            if fastest is not None and fastest["LapTime"] is not None:
                # Match the same FullName key used in GRID for consistency.
                driver_info = session.get_driver(driver_code)
                full_name = normalise_driver(f"{driver_info['FirstName']} {driver_info['LastName']}")
                fp1_times[full_name] = _time_to_seconds(fastest["LapTime"])
        except Exception:
            continue
    return fp1_times


def fetch_sprint_result(year: int, gp: str) -> list[dict]:
    """Fetch sprint race result if this is a sprint weekend. Empty list otherwise."""
    try:
        session = fastf1.get_session(year, gp, "S")
        session.load()
    except (DataNotLoadedError, ValueError, KeyError):
        return []  # Not a sprint weekend

    results = session.results.sort_values("Position")
    sprint = []
    for _, row in results.iterrows():
        sprint.append({
            "driver": normalise_driver(str(row["FullName"])),
            "pos": int(row["Position"]),
        })
    return sprint


def compute_team_pace_deficit(grid: list[dict]) -> dict:
    """Each team's fastest quali time, normalised to fastest team = 0.0."""
    team_best = {}
    for entry in grid:
        if entry["q_time"] is None:
            continue
        team = entry["team"]
        if team not in team_best or entry["q_time"] < team_best[team]:
            team_best[team] = entry["q_time"]

    if not team_best:
        return {}
    fastest = min(team_best.values())
    return {team: round(t - fastest, 3) for team, t in team_best.items()}


def fetch_previous_race_finishes(races_dir: Path, current_round: int) -> dict:
    """Read the previous round's result.json to populate r1_finish."""
    prev_round = current_round - 1
    if prev_round < 1:
        return {}
    # Find the folder matching prev_round
    for folder in races_dir.iterdir():
        if folder.is_dir() and folder.name.startswith(f"{prev_round:02d}_"):
            result_path = folder / "result.json"
            if result_path.exists():
                with open(result_path) as f:
                    data = json.load(f)
                return {r["driver"]: r.get("pos") for r in data["result"]}
    return {}


def _time_to_seconds(td) -> float:
    """Convert pandas Timedelta or datetime.timedelta to seconds."""
    if hasattr(td, "total_seconds"):
        return round(td.total_seconds(), 3)
    return float(td)


def _is_null_time(val) -> bool:
    """Check if a quali time value is null/NaT."""
    if val is None:
        return True
    if hasattr(val, "isna") and val.isna():
        return True
    s = str(val).lower()
    return s in ("nat", "nan", "", "none")


def write_data_py(
    output_path: Path,
    race_name: str,
    round_num: int,
    grid: list[dict],
    fp1_times: dict,
    sprint_result: list[dict],
    team_pace_deficit: dict,
    prev_finishes: dict,
    is_sprint_weekend: bool,
):
    """Write the full data.py file using the schema engine.py expects."""

    # Build DRIVER_EXPERIENCE from previous race + a season-count lookup.
    # f1_seasons is not in FastF1, so we default. User can edit.
    SEASONS = {
        "Andrea Kimi Antonelli": 2, "George Russell": 7, "Lewis Hamilton": 19,
        "Max Verstappen": 11, "Charles Leclerc": 8, "Lando Norris": 7,
        "Oscar Piastri": 4, "Isack Hadjar": 2, "Pierre Gasly": 9,
        "Franco Colapinto": 2, "Carlos Sainz": 11, "Alex Albon": 6,
        "Liam Lawson": 2, "Arvid Lindblad": 0, "Ollie Bearman": 2,
        "Esteban Ocon": 9, "Nico Hulkenberg": 14, "Gabriel Bortoleto": 2,
        "Fernando Alonso": 23, "Lance Stroll": 9, "Valtteri Bottas": 13,
        "Sergio Perez": 15,
    }
    driver_experience = {}
    for entry in grid:
        d = entry["driver"]
        driver_experience[d] = {
            "f1_seasons": SEASONS.get(d, 5),
            "r1_finish": prev_finishes.get(d),
        }

    # Defaults for fields FastF1 can't provide. User edits after generation.
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

    lines = []
    lines.append('"""')
    lines.append(f"R{round_num}: {race_name} 2026")
    lines.append("Auto-generated by fetch_race_data.py. Hand-edit WEATHER and CIRCUIT below.")
    lines.append('"""')
    lines.append("")
    lines.append(f"RACE_INFO = {json_compact({'name': race_name + ' Grand Prix', 'circuit': race_name, 'date': '', 'round': round_num, 'is_sprint_weekend': is_sprint_weekend})}")
    lines.append("")

    lines.append("GRID = [")
    for entry in grid:
        lines.append(f"    {json_compact(entry)},")
    lines.append("]")
    lines.append("")

    lines.append("FP1_TIMES = {")
    for d, t in fp1_times.items():
        lines.append(f"    {json.dumps(d)}: {t},")
    lines.append("}")
    lines.append("")

    if sprint_result:
        lines.append("SPRINT_RESULT = [")
        for entry in sprint_result:
            lines.append(f"    {json_compact(entry)},")
        lines.append("]")
    else:
        lines.append("SPRINT_RESULT = []")
    lines.append("")

    lines.append("DRIVER_EXPERIENCE = {")
    for d, info in driver_experience.items():
        lines.append(f"    {json.dumps(d)}: {json_compact(info)},")
    lines.append("}")
    lines.append("")

    lines.append("TEAM_PACE_DEFICIT = {")
    for team, deficit in sorted(team_pace_deficit.items(), key=lambda x: x[1]):
        lines.append(f"    {json.dumps(team)}: {deficit},")
    lines.append("}")
    lines.append("")

    lines.append("# Edit per race based on team form. Defaults shown.")
    lines.append(f"START_PROCEDURE = {json_compact(DEFAULT_START_PROCEDURE)}")
    lines.append("")

    lines.append(f"ENERGY_READINESS = {json_compact(DEFAULT_ENERGY_READINESS)}")
    lines.append("")

    lines.append("# Edit per race with track-specific veteran advantages.")
    lines.append("CIRCUIT_HISTORY = {}")
    lines.append("")

    lines.append("# Edit per race. type is 'high_speed' / 'street' / 'balanced'.")
    lines.append('CIRCUIT = {"type": "balanced", "pit_loss_seconds": 22}')
    lines.append("")

    lines.append("# hardness: 0=softest, 1=hardest. one_stop_probability: 0-1.")
    lines.append('TYRE_COMPOUNDS = {"hardness": 0.5, "one_stop_probability": 0.65}')
    lines.append("")

    lines.append("# IMPORTANT: Replace with the race day forecast before running prediction.")
    lines.append('WEATHER = {"track_temp_c": 30, "rain_probability": 0.10}')
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def json_compact(obj) -> str:
    """JSON dump with no trailing newline, suitable for inline embedding."""
    return json.dumps(obj, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Fetch race data from FastF1 and generate data.py")
    parser.add_argument("folder", help="Local race folder name, e.g. 06_monaco")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--round", type=int, required=True, help="F1 official round number")
    parser.add_argument("--circuit", required=True, help="Circuit/GP name FastF1 recognizes, e.g. Monaco")
    args = parser.parse_args()

    setup_cache()

    races_dir = Path("races")
    race_folder = races_dir / args.folder
    race_folder.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {args.year} {args.circuit} GP (round {args.round}) from FastF1...")

    print("  Qualifying...")
    grid = fetch_grid(args.year, args.circuit)
    print(f"    {len(grid)} drivers loaded.")

    print("  FP1...")
    fp1 = fetch_fp1_times(args.year, args.circuit)
    print(f"    {len(fp1)} drivers loaded.")

    print("  Sprint (if applicable)...")
    sprint = fetch_sprint_result(args.year, args.circuit)
    is_sprint = bool(sprint)
    print(f"    {'Yes' if is_sprint else 'No'} sprint this weekend.")

    print("  Computing team pace deficit from quali times...")
    team_pace = compute_team_pace_deficit(grid)

    print("  Reading previous race result.json for driver finishes...")
    prev_finishes = fetch_previous_race_finishes(races_dir, args.round)
    print(f"    {len(prev_finishes)} previous-race finishes loaded.")

    output_path = race_folder / "data.py"
    write_data_py(
        output_path=output_path,
        race_name=args.circuit,
        round_num=args.round,
        grid=grid,
        fp1_times=fp1,
        sprint_result=sprint,
        team_pace_deficit=team_pace,
        prev_finishes=prev_finishes,
        is_sprint_weekend=is_sprint,
    )

    print(f"\nWrote {output_path}")
    print("\nNEXT STEPS:")
    print(f"  1. Open {output_path} in VS Code")
    print("  2. Edit WEATHER with race day forecast (rain probability and track temp)")
    print("  3. Edit CIRCUIT type (high_speed / street / balanced) and pit_loss_seconds")
    print("  4. Edit TYRE_COMPOUNDS based on Pirelli's allocation")
    print("  5. Add CIRCUIT_HISTORY entries for veterans with strong records here")
    print(f"  6. Run: python engine.py {args.folder}")


if __name__ == "__main__":
    main()