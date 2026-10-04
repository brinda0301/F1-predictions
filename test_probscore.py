"""
test_probscore.py

Pins the probability metrics and the overtaking index to known values, so a
refactor that changes a score shows up as a failure rather than a quiet shift
in the published numbers.

Run with:
    python -m pytest test_probscore.py -v
or:
    python test_probscore.py
"""

import math

import numpy as np

import probscore
import prob_backtest as pb


def test_log_loss_known_values():
    assert abs(probscore.log_loss({"a": 0.5, "b": 0.5}, "a") - math.log(2)) < 1e-9
    # Percentages normalise the same as fractions.
    assert abs(probscore.log_loss({"a": 25, "b": 75}, "a") - math.log(4)) < 1e-9


def test_zero_probability_winner_is_floored_not_infinite():
    assert abs(probscore.log_loss({"a": 1.0, "b": 0.0}, "b") + math.log(probscore.EPS)) < 1e-9
    # A winner missing from the prediction entirely is scored the same way.
    assert abs(probscore.log_loss({"a": 1.0}, "z") + math.log(probscore.EPS)) < 1e-9


def test_brier_known_values():
    assert probscore.brier({"a": 1.0, "b": 0.0}, "a") == 0.0
    assert probscore.brier({"a": 1.0, "b": 0.0}, "b") == 2.0
    assert abs(probscore.brier({"a": 0.5, "b": 0.5}, "a") - 0.5) < 1e-9


def test_grid_prior_favours_pole_and_never_hits_zero():
    races = [([1, 2, 3, 4], 1)] * 6 + [([1, 2, 3, 4], 2)] * 3 + [([1, 2, 3, 4], 4)]
    pr = probscore.grid_prior(races, max_slot=4)
    assert pr[1] > pr[2] > pr[3] > 0
    p = probscore.grid_probs({"x": 1, "y": 2, "z": 3}, pr)
    assert abs(sum(p.values()) - 1) < 1e-9


def test_round_scores_resolves_short_names():
    """Early 2026 predictions wrote 'Kimi Antonelli'; results use the full name."""
    pred = {"predictions": [
        {"driver": "Kimi Antonelli", "win_pct": 60, "grid_pos": 1},
        {"driver": "George Russell", "win_pct": 40, "grid_pos": 2}]}
    s = probscore.round_scores(pred, "Andrea Kimi Antonelli", prior=[0, 0.5, 0.2])
    assert abs(s["mc_log_loss"] + math.log(0.6)) < 1e-3


def test_spearman_is_one_for_processional_race():
    assert abs(pb.spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1) < 1e-9
    assert abs(pb.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1) < 1e-9


def test_overtaking_index_uses_only_earlier_races():
    hist = [
        {"circuit": "monaco", "date": "2023-05-28", "_corr": 0.9},
        {"circuit": "monaco", "date": "2024-05-26", "_corr": 0.8},
        {"circuit": "monaco", "date": "2025-05-25", "_corr": 0.1},   # target race
        {"circuit": "monza", "date": "2024-09-01", "_corr": 0.5},
    ]
    v, n = pb.overtaking_index(hist, "monaco", "2025-05-25")
    assert n == 2 and abs(v - 0.85) < 1e-9


def test_logit_learns_that_pole_wins():
    """Synthetic field where the front of the grid always wins."""
    rng = np.random.default_rng(1)
    races = []
    for _ in range(40):
        X = np.column_stack([np.log(np.arange(1, 11)), rng.uniform(0, 1, 10)])
        races.append((X, 0))
    w = pb.fit_logit(races)
    assert w[0] < -1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
