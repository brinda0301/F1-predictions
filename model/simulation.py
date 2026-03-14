"""
Monte Carlo race simulation engine.
Models random events that affect F1 race outcomes:
- Safety cars (compresses the field)
- Rain (amplifies driver skill gap)
- Mechanical DNFs
- First-lap incidents
"""

import numpy as np
from collections import defaultdict
from data_loader import DRIVER_STATS


# Albert Park race event probabilities
SAFETY_CAR_PROB = 0.60    # ~60% chance of at least one safety car
RAIN_PROB = 0.15          # ~15% based on Melbourne autumn forecast
DNF_RATE = 0.08           # ~8% mechanical DNF rate per driver
FIRST_LAP_CHAOS = 0.25   # ~25% chance of first-lap incident


def simulate_race(predictions: list[dict], seed: int = None) -> list[str]:
    """
    Simulate a single race and return finishing order (driver names).

    Args:
        predictions: list of dicts with 'driver', 'win_prob'
        seed: optional random seed for reproducibility

    Returns:
        list of driver names in finishing order (DNFs excluded)
    """
    if seed is not None:
        np.random.seed(seed)

    drivers = [p["driver"] for p in predictions]
    base_probs = np.array([p["win_prob"] for p in predictions])

    # Base performance with noise
    performance = np.random.normal(base_probs, base_probs * 0.4)

    # Safety car
    if np.random.random() < SAFETY_CAR_PROB:
        performance = performance * 0.7 + np.random.uniform(0, 0.3, len(drivers))

    # Rain
    if np.random.random() < RAIN_PROB:
        for i, d in enumerate(drivers):
            exp = DRIVER_STATS.get(d, {}).get("wins", 0)
            performance[i] += exp * 0.002

    # First-lap chaos
    if np.random.random() < FIRST_LAP_CHAOS:
        victim = np.random.randint(0, min(6, len(drivers)))
        performance[victim] *= 0.5

    # DNFs
    for i in range(len(drivers)):
        if np.random.random() < DNF_RATE:
            performance[i] = -1

    # Rank by performance
    ranking = np.argsort(-performance)
    finishers = [drivers[r] for r in ranking if performance[r] > 0]
    return finishers


def run_monte_carlo(predictions: list[dict], n_sims: int = 50000) -> list[dict]:
    """
    Run n_sims race simulations and aggregate results.

    Args:
        predictions: scored driver predictions
        n_sims: number of simulations to run

    Returns:
        list of dicts sorted by win percentage, with:
        - win_pct, podium_pct, points_pct
        - All original prediction fields
    """
    np.random.seed(42)

    drivers = [p["driver"] for p in predictions]
    win_counts = defaultdict(int)
    podium_counts = defaultdict(int)
    points_counts = defaultdict(int)

    for sim in range(n_sims):
        finishers = simulate_race(predictions)

        if len(finishers) > 0:
            win_counts[finishers[0]] += 1

        for j, driver in enumerate(finishers[:3]):
            podium_counts[driver] += 1

        for j, driver in enumerate(finishers[:10]):
            points_counts[driver] += 1

    # Build results
    results = []
    driver_lookup = {p["driver"]: p for p in predictions}

    for driver in drivers:
        p = driver_lookup[driver]
        results.append({
            "driver": driver,
            "team": p["team"],
            "grid_pos": p["grid_pos"],
            "win_pct": round(win_counts[driver] / n_sims * 100, 1),
            "podium_pct": round(podium_counts[driver] / n_sims * 100, 1),
            "points_pct": round(points_counts[driver] / n_sims * 100, 1),
            "model_score": round(p["raw_score"], 4),
            "features": p["features"],
        })

    results.sort(key=lambda x: x["win_pct"], reverse=True)
    return results
