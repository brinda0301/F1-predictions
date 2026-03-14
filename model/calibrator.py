"""
Self-Calibrating Weight System
After each race, compares prediction vs actual result.
Adjusts feature weights using gradient descent.
"""

import json
import os
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def compute_position_error(prediction_path, result_path):
    """
    Compare predicted rankings vs actual finishing order.
    Returns per-driver position error and overall metrics.
    """
    with open(prediction_path) as f:
        pred = json.load(f)
    with open(result_path) as f:
        result = json.load(f)

    pred_ranking = {p["driver"]: i + 1 for i, p in enumerate(pred["predictions"])}
    actual_ranking = {}
    for r in result["result"]:
        if r["pos"] is not None:
            actual_ranking[r["driver"]] = r["pos"]

    errors = []
    driver_errors = {}
    for driver, pred_pos in pred_ranking.items():
        if driver in actual_ranking:
            error = abs(pred_pos - actual_ranking[driver])
            errors.append(error)
            driver_errors[driver] = {
                "predicted": pred_pos,
                "actual": actual_ranking[driver],
                "error": error,
            }

    mean_error = np.mean(errors) if errors else 999
    winner_correct = False

    if pred["predictions"] and result["result"]:
        predicted_winner = pred["predictions"][0]["driver"]
        actual_winner = next((r["driver"] for r in result["result"] if r["pos"] == 1), None)
        winner_correct = predicted_winner == actual_winner

    return {
        "mean_position_error": round(mean_error, 2),
        "winner_correct": winner_correct,
        "driver_errors": driver_errors,
        "n_drivers_compared": len(errors),
    }


def calibrate_weights(race_round):
    """
    Adjust model weights based on prediction accuracy for a given race.
    Uses simple gradient: features that contributed to correct top-5
    predictions get weight increases. Features that misled get decreases.

    Learning rate decays as we gather more race data.
    """
    config = load_config()
    weights = config["weights"]
    races_dir = os.path.join(os.path.dirname(__file__), "..", "races")

    # Find race folder
    race_folder = None
    for folder in os.listdir(races_dir):
        if folder.startswith(f"{race_round:02d}_"):
            race_folder = folder
            break

    if not race_folder:
        print(f"No race folder found for round {race_round}")
        return None

    race_path = os.path.join(races_dir, race_folder)
    pred_path = os.path.join(race_path, "prediction.json")
    result_path = os.path.join(race_path, "result.json")

    if not os.path.exists(pred_path) or not os.path.exists(result_path):
        print(f"Missing prediction.json or result.json in {race_folder}")
        return None

    # Load prediction with features
    with open(pred_path) as f:
        pred_data = json.load(f)
    with open(result_path) as f:
        result_data = json.load(f)

    # Build actual finishing map
    actual_positions = {}
    for r in result_data["result"]:
        if r["pos"] is not None:
            actual_positions[r["driver"]] = r["pos"]

    # Learning rate: higher early, lower as we accumulate data
    completed_races = config.get("last_calibrated_after_round", 0)
    learning_rate = 0.05 / (1 + completed_races * 0.3)

    print(f"\nCalibrating weights after Round {race_round}")
    print(f"Learning rate: {learning_rate:.4f}")

    # For each driver in top 10 prediction, check if their features
    # pointed in the right direction
    adjustments = {k: 0.0 for k in weights}

    for pred in pred_data["predictions"][:10]:
        driver = pred["driver"]
        pred_rank = pred_data["predictions"].index(pred) + 1
        actual_rank = actual_positions.get(driver)

        if actual_rank is None:
            continue

        # Positive = model ranked too high (overestimated)
        # Negative = model ranked too low (underestimated)
        rank_error = pred_rank - actual_rank

        features = pred.get("features", {})
        for feat_name, feat_value in features.items():
            if feat_name in adjustments:
                if rank_error > 0:
                    # We overestimated this driver.
                    # Their high-scoring features misled us. Decrease weight.
                    if feat_value > 0.5:
                        adjustments[feat_name] -= learning_rate * feat_value * 0.1
                elif rank_error < 0:
                    # We underestimated this driver.
                    # Their high-scoring features should count more. Increase weight.
                    if feat_value > 0.5:
                        adjustments[feat_name] += learning_rate * feat_value * 0.1

    # Apply adjustments
    print("\nWeight adjustments:")
    for feat, adj in adjustments.items():
        old = weights[feat]
        weights[feat] = max(0.01, weights[feat] + adj)
        if abs(adj) > 0.001:
            direction = "+" if adj > 0 else ""
            print(f"  {feat:<20} {old:.4f} -> {weights[feat]:.4f} ({direction}{adj:.4f})")

    # Normalize to sum to 1.0
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    # Update config
    config["weights"] = weights
    config["last_calibrated_after_round"] = race_round

    # Compute and store accuracy
    error_data = compute_position_error(pred_path, result_path)

    # Update accuracy history
    pred_winner = pred_data["predictions"][0]["driver"]
    pred_win_pct = pred_data["predictions"][0]["win_pct"]
    actual_winner = next((r["driver"] for r in result_data["result"] if r["pos"] == 1), "Unknown")

    pred_podium = [p["driver"] for p in pred_data["predictions"][:3]]
    actual_podium = [r["driver"] for r in result_data["result"] if r["pos"] and r["pos"] <= 3]
    overlap = len(set(pred_podium) & set(actual_podium))

    accuracy_entry = {
        "round": race_round,
        "race": race_folder.split("_", 1)[1].replace("_", " ").title(),
        "predicted_winner": pred_winner,
        "predicted_win_pct": pred_win_pct,
        "actual_winner": actual_winner,
        "correct": pred_winner == actual_winner,
        "podium_predicted": pred_podium,
        "podium_actual": actual_podium,
        "podium_overlap": overlap,
        "mean_position_error": error_data["mean_position_error"],
    }

    # Replace or append
    history = config.get("accuracy_history", [])
    existing = [i for i, h in enumerate(history) if h["round"] == race_round]
    if existing:
        history[existing[0]] = accuracy_entry
    else:
        history.append(accuracy_entry)
    config["accuracy_history"] = history

    save_config(config)

    print(f"\nAccuracy for Round {race_round}:")
    print(f"  Winner: {'CORRECT' if accuracy_entry['correct'] else 'WRONG'}")
    print(f"  Predicted: {pred_winner} ({pred_win_pct}%)")
    print(f"  Actual: {actual_winner}")
    print(f"  Podium overlap: {overlap}/3")
    print(f"  Mean position error: {error_data['mean_position_error']}")
    print(f"\nWeights saved to config.json")

    return config


def get_accuracy_summary():
    """Get overall model accuracy across all completed races."""
    config = load_config()
    history = config.get("accuracy_history", [])

    if not history:
        return {"races_completed": 0, "accuracy": 0}

    correct = sum(1 for h in history if h.get("correct"))
    total = len(history)
    avg_podium = np.mean([h.get("podium_overlap", 0) for h in history])
    avg_error = np.mean([h["mean_position_error"] for h in history if h.get("mean_position_error")])

    return {
        "races_completed": total,
        "winner_accuracy": f"{correct}/{total} ({correct/total*100:.0f}%)",
        "avg_podium_overlap": round(avg_podium, 1),
        "avg_position_error": round(avg_error, 1) if not np.isnan(avg_error) else None,
        "history": history,
        "current_weights": config["weights"],
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        race_round = int(sys.argv[1])
        calibrate_weights(race_round)
    else:
        summary = get_accuracy_summary()
        print(json.dumps(summary, indent=2))