"""
probscore.py

Scores win probabilities instead of winner picks.

A hit rate treats a 90% call and a 30% call on the same driver as identical.
Log loss and Brier score do not: a confident miss costs far more than a hedged
one, and a confident hit earns more. Both are lower-is-better.

    log loss  = -ln(probability given to the actual winner)
    Brier     = sum over drivers of (p - outcome)^2, outcome 1 for the winner

Reference points for a 22-car field:
    uniform guess (1/22 each)   log loss 3.09, Brier 0.955
    50% on the winner           log loss 0.69
    pole baseline, 2022-2025    see prob_backtest.py output

Also holds the grid-slot prior: P(win | starting slot) estimated from past
races. That is the "always pick pole" baseline written as a probability, so it
can be scored on the same scale as the model.
"""

import math

EPS = 1e-4  # floor so a zero-probability winner costs ln(1e4) = 9.2, not infinity


def normalise(probs):
    """Scale a {driver: p} map so it sums to 1. Accepts percentages."""
    total = sum(max(p, 0.0) for p in probs.values())
    if total <= 0:
        n = len(probs)
        return {d: 1.0 / n for d in probs}
    return {d: max(p, 0.0) / total for d, p in probs.items()}


def log_loss(probs, winner):
    p = normalise(probs).get(winner, 0.0)
    return -math.log(max(p, EPS))


def brier(probs, winner):
    probs = normalise(probs)
    if winner not in probs:
        probs = dict(probs, **{winner: 0.0})
    return sum((p - (1.0 if d == winner else 0.0)) ** 2 for d, p in probs.items())


def score(probs, winner):
    """All three numbers for one race."""
    probs = normalise(probs)
    top = max(probs, key=probs.get)
    return {
        "log_loss": round(log_loss(probs, winner), 4),
        "brier": round(brier(probs, winner), 4),
        "p_winner": round(probs.get(winner, 0.0), 4),
        "hit": top == winner,
    }


# Grid-slot prior -------------------------------------------------------------

def grid_prior(races, max_slot=22, strength=2.0):
    """P(win | grid slot) from past races, shrunk toward 1/max_slot.

    `races` is an iterable of (grid_slots, winner_slot) pairs. Returns a list
    indexed by slot (index 0 unused). Shrinkage keeps back-of-grid slots above
    zero, since a back-row winner happens (Monza 2026 had a P22 winner).
    """
    wins = [0.0] * (max_slot + 1)
    starts = [0.0] * (max_slot + 1)
    for slots, win_slot in races:
        for s in slots:
            starts[min(max(s, 1), max_slot)] += 1
        wins[min(max(win_slot, 1), max_slot)] += 1
    base = 1.0 / max_slot
    return [0.0] + [(wins[s] + strength * base) / (starts[s] + strength)
                    for s in range(1, max_slot + 1)]


def grid_probs(grid, prior):
    """{driver: grid slot} -> {driver: P(win)} from the prior, renormalised."""
    top = len(prior) - 1
    raw = {d: prior[min(max(s, 1), top)] for d, s in grid.items()}
    return normalise(raw)


PRIOR_PATH = "grid_prior.json"


def load_prior(path=PRIOR_PATH):
    """The 2022-2025 prior written by `prob_backtest.py --write-prior`.

    Used to score the pole baseline on live 2026 rounds without a network call.
    Returns None if the file has not been generated.
    """
    import json
    import os
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["prior"]


# Live scoring ----------------------------------------------------------------

def round_scores(pred, winner, prior=None):
    """Probability scores for one published prediction.json.

    Returns fields to merge into a config.json accuracy_history entry:
    Monte Carlo, XGBoost if it ran, and the pole baseline from the grid prior.
    """
    out = {}
    names = [p["driver"] for p in pred["predictions"]]
    if winner not in names:
        # Early rounds wrote "Kimi Antonelli"; results use "Andrea Kimi Antonelli".
        same = [n for n in names if n.split()[-1] == winner.split()[-1]]
        if len(same) == 1:
            winner = same[0]
    mc = {p["driver"]: p["win_pct"] for p in pred["predictions"]}
    s = score(mc, winner)
    out["mc_log_loss"], out["mc_brier"] = s["log_loss"], s["brier"]
    xgb = [p for p in (pred.get("xgboost") or {}).get("predictions", [])
           if p.get("win_prob") is not None]
    if xgb:
        s = score({p["driver"]: p["win_prob"] for p in xgb}, winner)
        out["xgb_log_loss"], out["xgb_brier"] = s["log_loss"], s["brier"]
    lg = pred.get("logit") or {}
    if lg.get("available"):
        s = score({p["driver"]: p["win_prob"] for p in lg["predictions"]}, winner)
        out["logit_log_loss"], out["logit_brier"] = s["log_loss"], s["brier"]
        out["logit_winner_correct"] = s["hit"]
    prior = prior or load_prior()
    if prior:
        g = grid_probs({p["driver"]: p["grid_pos"] for p in pred["predictions"]}, prior)
        s = score(g, winner)
        out["pole_log_loss"], out["pole_brier"] = s["log_loss"], s["brier"]
    return out


def backfill(config_path="config.json", races_dir="races"):
    """Add probability scores to every scored round already in config.json."""
    import json
    from pathlib import Path
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    folders = {int(p.name.split("_")[0]): p for p in Path(races_dir).iterdir()
               if p.is_dir() and p.name[:2].isdigit()}
    prior = load_prior()
    for e in cfg.get("accuracy_history", []):
        f = folders.get(e["round"])
        if not f or not (f / "prediction.json").exists():
            continue
        pred = json.loads((f / "prediction.json").read_text(encoding="utf-8"))
        e.update(round_scores(pred, e["actual_winner"], prior))
    Path(config_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg["accuracy_history"]


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--backfill"]:
        hist = backfill()
        keys = ("mc_log_loss", "xgb_log_loss", "pole_log_loss")
        print(f"{'Rd':<4}{'MC':>8}{'XGB':>8}{'pole':>8}")
        for e in hist:
            vals = [f"{e[k]:.2f}" if k in e else "-" for k in keys]
            print(f"{e['round']:<4}" + "".join(f"{v:>8}" for v in vals))
        for k in keys:
            v = [e[k] for e in hist if k in e]
            print(f"mean {k:<14} {sum(v) / len(v):.3f} over {len(v)} rounds")
    else:
        print("usage: python probscore.py --backfill")
