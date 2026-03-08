"""
Feature engineering pipeline for F1 race prediction.
Transforms raw qualifying, practice, and historical data into model-ready features.
"""

import numpy as np
from data_loader import (
    GRID_2026, FP_RANKINGS, DRIVER_STATS, TEAM_STRENGTH,
    BETTING_ODDS, GRID_SIZE
)


def compute_features(driver_entry: dict) -> dict:
    """
    Build a feature vector for a single driver.

    Args:
        driver_entry: dict with keys 'driver', 'team', 'pos', 'q_time'

    Returns:
        dict of feature_name -> float (0.0 to 1.0 scale)
    """
    name = driver_entry["driver"]
    team = driver_entry["team"]
    grid_pos = driver_entry["pos"]

    stats = DRIVER_STATS.get(name, {
        "wins": 0, "podiums": 0, "poles": 0,
        "seasons": 0, "aus_wins": 0, "aus_podiums": 0
    })
    fp = FP_RANKINGS.get(name, {"fp1": GRID_SIZE, "fp2": GRID_SIZE, "fp3": GRID_SIZE})
    team_str = TEAM_STRENGTH.get(team, 0.3)
    betting_prob = BETTING_ODDS.get(name, 0.001)

    # 1. Grid position score (pole = 1.0, last = 0.0)
    grid_score = max(0, (GRID_SIZE - grid_pos) / (GRID_SIZE - 1))

    # 2. Average practice position score
    avg_fp = (fp["fp1"] + fp["fp2"] + fp["fp3"]) / 3
    practice_score = max(0, (GRID_SIZE - avg_fp) / (GRID_SIZE - 1))

    # 3. Practice consistency (lower variance = higher score)
    fp_vals = [fp["fp1"], fp["fp2"], fp["fp3"]]
    fp_std = np.std(fp_vals)
    consistency_score = max(0, 1 - (fp_std / 10))

    # 4. Practice-to-qualifying improvement
    quali_vs_practice = avg_fp - grid_pos
    improvement_score = min(1, max(0, (quali_vs_practice + 5) / 10))

    # 5. Career experience factor
    exp_score = min(1.0, (
        stats["wins"] * 0.02 +
        stats["podiums"] * 0.005 +
        stats["poles"] * 0.01 +
        stats["seasons"] * 0.03
    ))

    # 6. Australian GP track knowledge
    aus_score = min(1.0, stats["aus_wins"] * 0.4 + stats["aus_podiums"] * 0.15)

    # 7. Team strength
    team_score = team_str

    # 8. Betting market signal
    market_score = betting_prob

    # 9. Teammate differential
    teammate_grid = None
    for d in GRID_2026:
        if d["team"] == team and d["driver"] != name:
            teammate_grid = d["pos"]
            break
    teammate_diff = 0.5
    if teammate_grid:
        diff = teammate_grid - grid_pos
        teammate_diff = min(1, max(0, (diff + 10) / 20))

    # 10. Weekend reliability score
    reliability_score = 1.0
    if driver_entry["q_time"] is None:
        reliability_score = 0.3
    if name == "Kimi Antonelli":
        reliability_score = 0.7  # FP3 crash, recovered for quali

    return {
        "grid_score": grid_score,
        "practice_score": practice_score,
        "consistency_score": consistency_score,
        "improvement_score": improvement_score,
        "experience_score": exp_score,
        "aus_track_score": aus_score,
        "team_score": team_score,
        "market_score": market_score,
        "teammate_diff": teammate_diff,
        "reliability_score": reliability_score,
    }


def compute_all_features() -> list[dict]:
    """Compute features for every driver on the grid."""
    results = []
    for entry in GRID_2026:
        features = compute_features(entry)
        results.append({
            "driver": entry["driver"],
            "team": entry["team"],
            "grid_pos": entry["pos"],
            "q_time": entry["q_time"],
            "features": features,
        })
    return results
