"""
Enhanced feature engineering with FastF1 telemetry data.
Adds 3 new features when FastF1 data is available:
- Sector balance (consistency across track sections)
- Speed trap advantage (straight-line speed vs field)
- Lap completion rate (reliability from actual laps run)

Falls back to base features.py when FastF1 is unavailable.
"""

import numpy as np
from features import compute_features as compute_base_features
from data_loader import GRID_2026, GRID_SIZE

try:
    from fastf1_loader import load_race_weekend, compute_sector_features
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False


# Updated weights when FastF1 features are available (sum = 1.0)
ENHANCED_WEIGHTS = {
    "grid_score":          0.22,
    "practice_score":      0.08,
    "consistency_score":   0.04,
    "improvement_score":   0.03,
    "experience_score":    0.06,
    "aus_track_score":     0.04,
    "team_score":          0.13,
    "market_score":        0.18,
    "teammate_diff":       0.03,
    "reliability_score":   0.06,
    # New FastF1 features
    "sector_balance":      0.05,
    "speed_advantage":     0.04,
    "lap_completion_rate": 0.04,
}


def compute_enhanced_features(driver_entry: dict, lap_times: dict = None) -> dict:
    """
    Compute base features + FastF1 sector/speed features.

    Args:
        driver_entry: dict from GRID_2026
        lap_times: output from fastf1_loader.extract_lap_times()

    Returns:
        dict of all features (13 if FastF1 available, 10 if not)
    """
    # Start with base features
    features = compute_base_features(driver_entry)

    # Add FastF1 features if data is available
    if lap_times:
        sector_features = compute_sector_features(lap_times, driver_entry["driver"])
        features.update(sector_features)

    return features


def compute_all_enhanced_features(year: int = 2026, gp: str = "Australia") -> tuple:
    """
    Compute features for all drivers using FastF1 when possible.

    Returns:
        (list of driver feature dicts, feature_weights dict)
    """
    lap_times = {}
    source = "static"

    if FASTF1_AVAILABLE:
        try:
            weekend_data = load_race_weekend(year, gp)
            lap_times = weekend_data.get("lap_times", {})
            source = weekend_data["source"]
        except Exception:
            pass

    results = []
    for entry in GRID_2026:
        features = compute_enhanced_features(entry, lap_times if lap_times else None)
        results.append({
            "driver": entry["driver"],
            "team": entry["team"],
            "grid_pos": entry["pos"],
            "q_time": entry["q_time"],
            "features": features,
        })

    # Use enhanced weights if FastF1 features are present
    weights = ENHANCED_WEIGHTS if lap_times else None
    print(f"  Feature source: {source}")
    print(f"  Features per driver: {len(results[0]['features'])}")

    return results, weights
