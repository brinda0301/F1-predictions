"""
mc_backtest.py

Scores the Monte Carlo engine on win probability (log loss) so its settings are
chosen by measurement, not by hand. Added R17.

Two test sets:
  2026      the 16 completed 2026 rounds, from each race's data.py, with their
            real hand-set inputs (circuit, weather, history). Uses the current
            config weights, which were calibrated through R16, so this set
            flatters the engine slightly.
  2024-25   47 races rebuilt by backtest.py from the timing API. Hand-set blocks
            sit at defaults, so this isolates the timing features.

Each setting runs with fewer simulations than live (default 3,000) to keep a
full sweep to a few minutes. Same seed for every setting.

    python mc_backtest.py              # both sets
    python mc_backtest.py --only2026
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.argv, _argv = ["engine"], sys.argv
import engine
import probscore
sys.argv = _argv

SETTINGS = [
    ("v1: recovery on, temp x1.0", dict(recovery="v1", temp_scale=1.0)),
    ("recovery off, temp x1.0", dict(recovery="none", temp_scale=1.0)),
    ("recovery off, temp x0.7", dict(recovery="none", temp_scale=0.7)),
    ("recovery off, temp x0.5", dict(recovery="none", temp_scale=0.5)),
    ("recovery off, temp x0.35", dict(recovery="none", temp_scale=0.35)),
]


def races_2026():
    out = []
    for f in engine.get_race_folders():
        res = Path("races") / f / "result.json"
        if not f[:2].isdigit() or not res.exists():
            continue
        r = json.loads(res.read_text())["result"]
        out.append((f, engine.load_race_data(f), r[0]["driver"]))
    return out


def races_hist(years=(2024, 2025)):
    import backtest
    out = []
    for y in years:
        n = len(backtest.get(f"{y}/")["RaceTable"]["Races"])
        for rnd in range(1, n + 1):
            try:
                rd = backtest.race_data(y, rnd, engine)
            except Exception:
                continue
            if rd:
                out.append((f"{y} R{rnd}", rd, min(rd["_RESULT"], key=rd["_RESULT"].get)))
    return out


def run(races, cfg, n_sims):
    print(f"  {'setting':<30}{'log loss':>10}{'winners':>10}{'avg P(win)':>12}")
    best = None
    for label, kw in SETTINGS:
        sc = []
        for _, rd, win in races:
            res = engine.simulate(rd, cfg, n_sims=n_sims, **kw)
            sc.append(probscore.score({r["driver"]: r["win_pct"] for r in res}, win))
        ll = np.mean([s["log_loss"] for s in sc])
        print(f"  {label:<30}{ll:>10.3f}{sum(s['hit'] for s in sc):>6}/{len(sc):<3}"
              f"{100 * np.mean([s['p_winner'] for s in sc]):>11.1f}%")
        if best is None or ll < best[1]:
            best = (label, ll)
    print(f"  best: {best[0]}")


if __name__ == "__main__":
    cfg = engine.load_config()
    n = 3000
    print(f"2026 rounds 1-16, {n} sims per race")
    run(races_2026(), cfg, n)
    if "--only2026" not in sys.argv:
        print(f"\n2024-2025 rebuilt races, {n} sims per race")
        run(races_hist(), cfg, n)
