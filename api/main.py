"""
F1 2026 Prediction API
======================
Endpoints:
  GET  /                         Health check
  GET  /races                    List all races
  GET  /races/{round}            Get race data + prediction
  POST /predict/{round}          Run prediction for a race
  POST /results/{round}          Submit actual result + calibrate
  GET  /accuracy                 Model accuracy tracker
  GET  /weights                  Current model weights
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

app = FastAPI(
    title="F1 2026 Race Predictor",
    description="ML model predicting F1 race winners. Zero betting data. Self-calibrating.",
    version="4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
RACES_DIR = os.path.join(BASE_DIR, "races")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


@app.get("/")
def root():
    config = load_config()
    history = config.get("accuracy_history", [])
    correct = sum(1 for h in history if h.get("correct"))
    return {
        "name": "F1 2026 Race Predictor",
        "model_version": config.get("model_version", "v4"),
        "races_predicted": len(history),
        "winner_accuracy": f"{correct}/{len(history)}" if history else "0/0",
        "endpoints": ["/races", "/predict/{round}", "/results/{round}", "/accuracy", "/weights"],
    }


@app.get("/races")
def list_races():
    config = load_config()
    calendar = config.get("calendar_2026", [])
    races = []
    for race in calendar:
        folder = f"{race['round']:02d}_{race['name'].lower().replace(' grand prix', '').replace(' ', '_')}"
        race_path = os.path.join(RACES_DIR, folder)
        has_prediction = os.path.exists(os.path.join(race_path, "prediction.json"))
        has_result = os.path.exists(os.path.join(race_path, "result.json"))
        races.append({
            **race,
            "folder": folder,
            "has_prediction": has_prediction,
            "has_result": has_result,
        })
    return {"races": races}


@app.get("/races/{race_round}")
def get_race(race_round: int):
    folder = None
    for f in sorted(os.listdir(RACES_DIR)):
        if f.startswith(f"{race_round:02d}_"):
            folder = f
            break

    if not folder:
        raise HTTPException(404, f"No data for round {race_round}")

    race_path = os.path.join(RACES_DIR, folder)
    response = {"round": race_round, "folder": folder}

    pred_path = os.path.join(race_path, "prediction.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            response["prediction"] = json.load(f)

    result_path = os.path.join(race_path, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            response["result"] = json.load(f)

    return response


@app.post("/predict/{race_round}")
def predict_race(race_round: int):
    folder = None
    for f in sorted(os.listdir(RACES_DIR)):
        if f.startswith(f"{race_round:02d}_"):
            folder = f
            break

    if not folder:
        raise HTTPException(404, f"No race folder for round {race_round}")

    data_path = os.path.join(RACES_DIR, folder, "data.py")
    if not os.path.exists(data_path):
        raise HTTPException(400, f"No data.py in races/{folder}/. Add qualifying data first.")

    from run_race import run_prediction
    result = run_prediction(folder)

    if result is None:
        raise HTTPException(500, "Prediction failed")

    return {"status": "success", "race": folder, "winner": result["predictions"][0]}


class RaceResult(BaseModel):
    result: list


@app.post("/results/{race_round}")
def submit_result(race_round: int, body: RaceResult):
    folder = None
    for f in sorted(os.listdir(RACES_DIR)):
        if f.startswith(f"{race_round:02d}_"):
            folder = f
            break

    if not folder:
        raise HTTPException(404, f"No race folder for round {race_round}")

    result_path = os.path.join(RACES_DIR, folder, "result.json")
    with open(result_path, "w") as f:
        json.dump({"result": [r.dict() if hasattr(r, 'dict') else r for r in body.result]}, f, indent=2)

    # Auto-calibrate
    from calibrator import calibrate_weights
    config = calibrate_weights(race_round)

    return {
        "status": "result saved and weights calibrated",
        "round": race_round,
        "new_weights": config["weights"] if config else None,
    }


@app.get("/accuracy")
def get_accuracy():
    from calibrator import get_accuracy_summary
    return get_accuracy_summary()


@app.get("/weights")
def get_weights():
    config = load_config()
    return {
        "weights": config["weights"],
        "last_calibrated": config.get("last_calibrated_after_round", 0),
        "regulation_params": config.get("regulation_params", {}),
    }
