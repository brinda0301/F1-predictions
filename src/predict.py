"""
F1 Australian GP 2026 Winner Prediction
========================================
Main entry point. Runs the full pipeline:
1. Load data
2. Engineer features
3. Score drivers
4. Run Monte Carlo simulation
5. Output predictions
"""

import json
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import compute_all_features
from model import score_all_drivers, FEATURE_WEIGHTS
from monte_carlo import run_monte_carlo


def main(n_sims: int = 50000, output_path: str = None):
    """Run the full prediction pipeline."""

    print("=" * 60)
    print("F1 2026 Australian GP Winner Prediction Model")
    print("=" * 60)
    print("Race: Sunday March 8, 2026 | Albert Park, Melbourne")
    print(f"Laps: 58 | Distance: 306km | Simulations: {n_sims:,}")
    print("=" * 60)

    # Step 1: Feature engineering
    print("\n[1/4] Computing features for 22 drivers...")
    driver_features = compute_all_features()

    # Step 2: Model scoring
    print("[2/4] Scoring drivers with weighted ensemble...")
    predictions = score_all_drivers(driver_features)

    # Step 3: Monte Carlo simulation
    print(f"[3/4] Running {n_sims:,} Monte Carlo race simulations...")
    results = run_monte_carlo(predictions, n_sims=n_sims)

    # Step 4: Output
    print("[4/4] Generating output...\n")

    header = f"{'Rank':<5} {'Driver':<22} {'Team':<15} {'Grid':<5} {'Win%':<8} {'Podium%':<9} {'Points%':<9}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results):
        print(
            f"{i+1:<5} {r['driver']:<22} {r['team']:<15} "
            f"P{r['grid_pos']:<4} {r['win_pct']:<8} "
            f"{r['podium_pct']:<9} {r['points_pct']:<9}"
        )

    # Save JSON output
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "predictions.json"
        )

    output = {
        "race": "2026 Australian Grand Prix",
        "circuit": "Albert Park, Melbourne",
        "date": "2026-03-08",
        "laps": 58,
        "distance_km": 306,
        "simulations": n_sims,
        "feature_weights": FEATURE_WEIGHTS,
        "predictions": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    winner = results[0]
    print(f"\n{'=' * 60}")
    print(f"PREDICTED WINNER: {winner['driver']} ({winner['team']})")
    print(f"Win Probability: {winner['win_pct']}%")
    print(f"Starting: P{winner['grid_pos']}")
    print(f"{'=' * 60}")
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    main(n_sims=n)