"""
Ensemble scoring model with softmax normalization.
Weights derived from historical F1 data analysis:
- Pole sitter wins ~40% of races at Albert Park
- Car performance accounts for ~60% of race outcome variance
- Market odds encode collective expert signals
"""

import numpy as np

# Feature weights (sum = 1.0)
FEATURE_WEIGHTS = {
    "grid_score":        0.25,
    "practice_score":    0.10,
    "consistency_score": 0.05,
    "improvement_score": 0.03,
    "experience_score":  0.07,
    "aus_track_score":   0.05,
    "team_score":        0.15,
    "market_score":      0.20,
    "teammate_diff":     0.03,
    "reliability_score": 0.07,
}


def compute_raw_score(features: dict) -> float:
    """Weighted sum of feature values."""
    score = 0.0
    for feat, weight in FEATURE_WEIGHTS.items():
        score += features.get(feat, 0.0) * weight
    return score


def softmax(scores: list[float], temperature: float = 0.15) -> np.ndarray:
    """
    Convert raw scores to probabilities.
    Lower temperature = more peaked distribution (favors top scorer).
    """
    scores = np.array(scores)
    exp_scores = np.exp((scores - np.max(scores)) / temperature)
    return exp_scores / exp_scores.sum()


def score_all_drivers(driver_features: list[dict]) -> list[dict]:
    """
    Score every driver and convert to win probabilities.

    Args:
        driver_features: list of dicts with 'driver', 'team', 'grid_pos', 'features'

    Returns:
        list of dicts with added 'raw_score' and 'win_prob'
    """
    predictions = []
    for df in driver_features:
        raw_score = compute_raw_score(df["features"])
        predictions.append({
            **df,
            "raw_score": raw_score,
        })

    raw_scores = [p["raw_score"] for p in predictions]
    win_probs = softmax(raw_scores)

    for i, p in enumerate(predictions):
        p["win_prob"] = float(win_probs[i])

    return predictions
