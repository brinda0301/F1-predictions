"""
test_probscore.py

Pins the probability metrics, the overtaking index, XGBoost v2 and the simulation to known values, so a
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
import history_data as pb


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


def test_xgb_v2_is_monotone_in_gap_and_grid():
    """v2 must never score a faster, further-forward car lower. v1 did at R17."""
    import xgb_model
    from xgboost import XGBClassifier
    if not xgb_model.MODEL_PATH.exists():
        return
    m = XGBClassifier()
    m.load_model(xgb_model.MODEL_PATH)
    base = [{"name": n, "team": t, "grid": g, "gap": gap, "sprint": None}
            for n, t, g, gap in [("A", "x", 1, 0.0), ("B", "y", 2, 0.3), ("C", "z", 3, 0.6),
                                  ("D", "x", 4, 0.7), ("E", "y", 5, 0.9), ("F", "z", 6, 1.0)]]
    p = xgb_model.race_probs(m, xgb_model.feature_rows(base, 0.75))
    assert all(p[i] >= p[i + 1] - 1e-9 for i in range(3)), p


def test_simulate_recovery_off_never_favours_back_of_grid():
    """Two identical drivers: the one starting further back must not win more."""
    import sys
    sys.argv = ["engine"]
    import engine
    grid = [{"driver": f"D{i}", "team": "Same", "pos": i, "q_time": 90.0} for i in range(1, 21)]
    rd = {"GRID": grid, "FP1_TIMES": {}, "SPRINT_RESULT": [], "DRIVER_EXPERIENCE": {},
          "TEAM_PACE_DEFICIT": {}, "START_PROCEDURE": {}, "ENERGY_READINESS": {},
          "CIRCUIT_HISTORY": {}, "RACE_INFO": {}, "CIRCUIT": {}, "TYRE_COMPOUNDS": {},
          "WEATHER": {}}
    res = {r["driver"]: r["win_pct"] for r in engine.simulate(rd, engine.load_config(), n_sims=2000)}
    assert res["D20"] <= res["D10"] + 0.5, (res["D10"], res["D20"])


def test_circuit_index_ignores_the_race_itself():
    import xgb_model
    meta = {"circuits": {"x": [["2024-01-01", 0.9], ["2025-01-01", 0.7], ["2026-01-01", 0.0]]},
            "ot_mean": 0.7}
    assert abs(xgb_model.circuit_index(meta, "x", "2026-01-01") - 0.8) < 1e-9


def test_xgb_v2_places_r17_at_marina_bay():
    """The saved model must find the circuit from the round number, offline."""
    import xgb_model
    if not xgb_model.MODEL_PATH.exists():
        return
    grid = [{"driver": "A", "team": "T", "pos": 1, "q_time": 91.373},
            {"driver": "B", "team": "U", "pos": 2, "q_time": 91.427},
            {"driver": "C", "team": "T", "pos": 3, "q_time": None}]
    out = xgb_model.predict({"GRID": grid, "RACE_INFO": {"round": 17, "date": "2026-10-11"}})
    assert out["available"]
    assert 0.7 < out["overtaking_index"] < 0.85
    p = {r["driver"]: r["win_prob"] for r in out["predictions"]}
    assert abs(sum(p.values()) - 1) < 1e-3 and p["A"] > p["C"]


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
