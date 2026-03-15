"""
FastF1-powered data loader for F1 race prediction.
Pulls real timing, telemetry, and session data from F1's live timing API.

Falls back to static data (data_loader.py) if FastF1 data is unavailable.

Usage:
    from fastf1_loader import load_race_weekend
    data = load_race_weekend(2026, "Australia")
"""

import os
import warnings
import numpy as np

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False
    warnings.warn(
        "FastF1 not installed. Run: pip install fastf1. "
        "Falling back to static data."
    )

# Local cache directory for FastF1 data
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "f1_cache")


def setup_cache():
    """Enable FastF1 caching to avoid re-downloading data."""
    if not FASTF1_AVAILABLE:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)


def load_session(year: int, gp: str, session_type: str):
    """
    Load a single F1 session.

    Args:
        year: Season year (e.g. 2026)
        gp: Grand Prix name (e.g. "Australia")
        session_type: "FP1", "FP2", "FP3", "Q", or "R"

    Returns:
        fastf1.core.Session object with data loaded
    """
    if not FASTF1_AVAILABLE:
        raise ImportError("FastF1 is not installed")

    setup_cache()
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=True, messages=True)
    return session


def extract_qualifying_data(year: int, gp: str) -> list[dict]:
    """
    Pull qualifying results from FastF1.

    Returns:
        list of dicts with keys: pos, driver, team, q_time (seconds)
    """
    session = load_session(year, gp, "Q")
    results = session.results

    grid = []
    for _, row in results.iterrows():
        # Get best qualifying time
        q_time = None
        for q_col in ["Q3", "Q2", "Q1"]:
            if q_col in row and not (hasattr(row[q_col], "total_seconds") and row[q_col].total_seconds() == 0):
                try:
                    q_time = row[q_col].total_seconds()
                    break
                except (AttributeError, TypeError):
                    continue

        grid.append({
            "pos": int(row["Position"]) if not np.isnan(row["Position"]) else 22,
            "driver": row["FullName"],
            "team": row["TeamName"],
            "q_time": q_time,
            "abbreviation": row["Abbreviation"],
            "driver_number": row["DriverNumber"],
        })

    grid.sort(key=lambda x: x["pos"])
    return grid


def extract_practice_data(year: int, gp: str) -> dict:
    """
    Pull FP1, FP2, FP3 rankings from FastF1.

    Returns:
        dict of driver_name -> {"fp1": rank, "fp2": rank, "fp3": rank}
    """
    rankings = {}

    for fp_num, fp_name in [(1, "FP1"), (2, "FP2"), (3, "FP3")]:
        try:
            session = load_session(year, gp, fp_name)
            laps = session.laps

            # Get fastest lap per driver
            fastest = laps.groupby("Driver")["LapTime"].min().sort_values()

            for rank, (driver_abbr, _) in enumerate(fastest.items(), 1):
                # Map abbreviation to full name
                driver_info = session.get_driver(driver_abbr)
                full_name = f"{driver_info['FirstName']} {driver_info['LastName']}"

                if full_name not in rankings:
                    rankings[full_name] = {"fp1": 22, "fp2": 22, "fp3": 22}
                rankings[full_name][f"fp{fp_num}"] = rank

        except Exception as e:
            warnings.warn(f"Could not load {fp_name}: {e}")

    return rankings


def extract_lap_times(year: int, gp: str, session_type: str = "Q") -> dict:
    """
    Pull detailed lap time data including sector times.

    Returns:
        dict of driver_name -> {
            "best_lap": float (seconds),
            "s1": float, "s2": float, "s3": float,
            "speed_trap": float (km/h),
            "laps_completed": int
        }
    """
    session = load_session(year, gp, session_type)
    laps = session.laps
    lap_data = {}

    for driver_abbr in laps["Driver"].unique():
        driver_laps = laps[laps["Driver"] == driver_abbr]
        driver_info = session.get_driver(driver_abbr)
        full_name = f"{driver_info['FirstName']} {driver_info['LastName']}"

        best_lap = driver_laps.loc[driver_laps["LapTime"].idxmin()] if not driver_laps.empty else None

        if best_lap is not None:
            lap_data[full_name] = {
                "best_lap": best_lap["LapTime"].total_seconds() if hasattr(best_lap["LapTime"], "total_seconds") else None,
                "s1": best_lap.get("Sector1Time", None),
                "s2": best_lap.get("Sector2Time", None),
                "s3": best_lap.get("Sector3Time", None),
                "speed_trap": best_lap.get("SpeedST", None),
                "laps_completed": len(driver_laps),
            }

            # Convert timedelta sectors to seconds
            for sector in ["s1", "s2", "s3"]:
                if lap_data[full_name][sector] is not None:
                    try:
                        lap_data[full_name][sector] = lap_data[full_name][sector].total_seconds()
                    except AttributeError:
                        pass

    return lap_data


def extract_weather(year: int, gp: str, session_type: str = "R") -> dict:
    """
    Pull weather conditions for a session.

    Returns:
        dict with keys: air_temp, track_temp, humidity, wind_speed, rain
    """
    session = load_session(year, gp, session_type)

    if session.weather_data is None or session.weather_data.empty:
        return {"air_temp": None, "track_temp": None, "humidity": None, "wind_speed": None, "rain": False}

    weather = session.weather_data
    return {
        "air_temp": round(weather["AirTemp"].mean(), 1),
        "track_temp": round(weather["TrackTemp"].mean(), 1),
        "humidity": round(weather["Humidity"].mean(), 1),
        "wind_speed": round(weather["WindSpeed"].mean(), 1),
        "rain": bool(weather["Rainfall"].any()),
    }


def load_race_weekend(year: int, gp: str) -> dict:
    """
    Load all available data for a race weekend.

    This is the main entry point. It tries FastF1 first,
    then falls back to static data if FastF1 fails.

    Args:
        year: Season year
        gp: Grand Prix name

    Returns:
        dict with keys:
            grid: qualifying results
            practice: FP1/FP2/FP3 rankings
            lap_times: detailed sector and speed data
            weather: race day weather
            source: "fastf1" or "static"
    """
    if not FASTF1_AVAILABLE:
        return _load_static_fallback()

    try:
        setup_cache()
        print(f"Loading {year} {gp} GP data from FastF1...")

        grid = extract_qualifying_data(year, gp)
        print(f"  Qualifying: {len(grid)} drivers loaded")

        practice = extract_practice_data(year, gp)
        print(f"  Practice: {len(practice)} drivers across FP1/FP2/FP3")

        lap_times = extract_lap_times(year, gp, "Q")
        print(f"  Lap times: {len(lap_times)} drivers with sector data")

        weather = extract_weather(year, gp, "Q")
        print(f"  Weather: {weather['air_temp']}C air, rain={weather['rain']}")

        return {
            "grid": grid,
            "practice": practice,
            "lap_times": lap_times,
            "weather": weather,
            "source": "fastf1",
        }

    except Exception as e:
        warnings.warn(f"FastF1 failed: {e}. Falling back to static data.")
        return _load_static_fallback()


def _load_static_fallback() -> dict:
    """Load from hardcoded data_loader.py as fallback."""
    from data_loader import GRID_2026, FP_RANKINGS

    return {
        "grid": GRID_2026,
        "practice": FP_RANKINGS,
        "lap_times": {},
        "weather": {"air_temp": 22, "track_temp": 35, "humidity": 55, "wind_speed": 12, "rain": False},
        "source": "static",
    }


# ============================================================
# ENHANCED FEATURES (only available with FastF1)
# ============================================================

def compute_sector_features(lap_times: dict, driver_name: str) -> dict:
    """
    Compute sector-level features for a driver.
    These features capture WHERE on the track a driver is fast/slow.

    Args:
        lap_times: output from extract_lap_times()
        driver_name: driver full name

    Returns:
        dict of additional features
    """
    if not lap_times or driver_name not in lap_times:
        return {"sector_balance": 0.5, "speed_advantage": 0.0, "lap_completion_rate": 0.5}

    driver = lap_times[driver_name]
    all_s1 = [d["s1"] for d in lap_times.values() if d.get("s1")]
    all_s2 = [d["s2"] for d in lap_times.values() if d.get("s2")]
    all_s3 = [d["s3"] for d in lap_times.values() if d.get("s3")]
    all_speed = [d["speed_trap"] for d in lap_times.values() if d.get("speed_trap")]
    all_laps = [d["laps_completed"] for d in lap_times.values() if d.get("laps_completed")]

    # Sector balance: how evenly fast across all three sectors
    sector_balance = 0.5
    if driver.get("s1") and driver.get("s2") and driver.get("s3"):
        sectors = [driver["s1"], driver["s2"], driver["s3"]]
        # Rank driver in each sector
        ranks = []
        for s, all_s in zip(sectors, [all_s1, all_s2, all_s3]):
            if all_s:
                rank = sum(1 for x in all_s if x < s) + 1
                ranks.append(rank / len(all_s))
        if ranks:
            sector_balance = 1.0 - np.std(ranks)  # low std = balanced

    # Speed trap advantage
    speed_advantage = 0.0
    if driver.get("speed_trap") and all_speed:
        max_speed = max(all_speed)
        min_speed = min(all_speed)
        if max_speed > min_speed:
            speed_advantage = (driver["speed_trap"] - min_speed) / (max_speed - min_speed)

    # Lap completion rate (reliability indicator)
    lap_completion_rate = 0.5
    if driver.get("laps_completed") and all_laps:
        max_laps = max(all_laps)
        if max_laps > 0:
            lap_completion_rate = min(1.0, driver["laps_completed"] / max_laps)

    return {
        "sector_balance": round(sector_balance, 3),
        "speed_advantage": round(speed_advantage, 3),
        "lap_completion_rate": round(lap_completion_rate, 3),
    }


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    if FASTF1_AVAILABLE:
        print("FastF1 is installed. Attempting to load 2026 Australian GP...\n")
        data = load_race_weekend(2026, "Australia")
        print(f"\nData source: {data['source']}")
        print(f"Drivers on grid: {len(data['grid'])}")

        if data["source"] == "fastf1":
            print("\nQualifying Results:")
            for d in data["grid"][:10]:
                q = f"{d['q_time']:.3f}s" if d['q_time'] else "NO TIME"
                print(f"  P{d['pos']:<3} {d['driver']:<22} {d['team']:<15} {q}")
    else:
        print("FastF1 not installed.")
        print("Install with: pip install fastf1")
        print("\nFalling back to static data...")
        data = load_race_weekend(2026, "Australia")
        print(f"Loaded {len(data['grid'])} drivers from static data")
