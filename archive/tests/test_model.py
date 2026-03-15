"""
Unit tests for F1 prediction model.
Run with: python -m pytest tests/test_model.py -v
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import compute_features, compute_all_features
from model import compute_raw_score, softmax, score_all_drivers, FEATURE_WEIGHTS
from monte_carlo import run_monte_carlo
from data_loader import GRID_2026


def test_feature_weights_sum_to_one():
    total = sum(FEATURE_WEIGHTS.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"


def test_features_are_bounded():
    """All features should be between 0 and 1."""
    all_features = compute_all_features()
    for df in all_features:
        for key, val in df["features"].items():
            assert 0.0 <= val <= 1.0, (
                f"{df['driver']} feature {key} = {val}, out of [0, 1] range"
            )


def test_softmax_sums_to_one():
    scores = [0.8, 0.6, 0.5, 0.3, 0.1]
    probs = softmax(scores)
    assert abs(probs.sum() - 1.0) < 1e-6


def test_softmax_preserves_order():
    scores = [0.9, 0.7, 0.5, 0.3]
    probs = softmax(scores)
    for i in range(len(probs) - 1):
        assert probs[i] > probs[i + 1]


def test_pole_sitter_has_highest_score():
    all_features = compute_all_features()
    predictions = score_all_drivers(all_features)
    top = max(predictions, key=lambda p: p["raw_score"])
    assert top["driver"] == "George Russell", (
        f"Expected Russell as top scorer, got {top['driver']}"
    )


def test_all_drivers_scored():
    all_features = compute_all_features()
    predictions = score_all_drivers(all_features)
    assert len(predictions) == 22


def test_monte_carlo_produces_results():
    all_features = compute_all_features()
    predictions = score_all_drivers(all_features)
    results = run_monte_carlo(predictions, n_sims=1000)
    assert len(results) == 22
    total_win = sum(r["win_pct"] for r in results)
    assert 95 < total_win < 105, f"Total win% = {total_win}, expected ~100"


def test_grid_position_affects_score():
    """Driver on pole should score higher than driver in P20 (all else equal)."""
    pole = GRID_2026[0]
    back = GRID_2026[19]
    pole_features = compute_features(pole)
    back_features = compute_features(back)
    pole_score = compute_raw_score(pole_features)
    back_score = compute_raw_score(back_features)
    assert pole_score > back_score


if __name__ == "__main__":
    tests = [
        test_feature_weights_sum_to_one,
        test_features_are_bounded,
        test_softmax_sums_to_one,
        test_softmax_preserves_order,
        test_pole_sitter_has_highest_score,
        test_all_drivers_scored,
        test_monte_carlo_produces_results,
        test_grid_position_affects_score,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
