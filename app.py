"""
F1 2026 Race Predictor - Unified Dashboard
One command: streamlit run app.py
One link: http://localhost:8501
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import sys
import numpy as np
from collections import defaultdict
import importlib

st.set_page_config(
    page_title="F1 2026 Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RACES_DIR = os.path.join(BASE_DIR, "races")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

TEAM_COLORS = {
    "Mercedes": "#00D2BE", "Ferrari": "#DC0000", "McLaren": "#FF8700",
    "Red Bull": "#3671C6", "Racing Bulls": "#6692FF", "Audi": "#FF0000",
    "Haas": "#B6BABD", "Alpine": "#0090FF", "Williams": "#005AFF",
    "Aston Martin": "#006F62", "Cadillac": "#1E1E1E",
}

# ============================================================
# HELPERS
# ============================================================

def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def get_race_folders():
    folders = []
    for f in sorted(os.listdir(RACES_DIR)):
        if os.path.isdir(os.path.join(RACES_DIR, f)) and f[0].isdigit():
            folders.append(f)
    return folders

def load_prediction(race_folder):
    path = os.path.join(RACES_DIR, race_folder, "prediction.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_result(race_folder):
    path = os.path.join(RACES_DIR, race_folder, "result.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ============================================================
# PREDICTION ENGINE (embedded, no imports needed)
# ============================================================

POLE_WIN_RATE = 0.45
OVERTAKE_FACTOR = 1.4
ENERGY_UNCERTAINTY = 0.04
NEW_TEAM_UNCERTAINTY = {"Cadillac": 0.06, "Audi": 0.04}

def compute_features(driver_entry, race_data, weights):
    name = driver_entry["driver"]
    team = driver_entry["team"]
    grid_pos = driver_entry["pos"]
    q_time = driver_entry["q_time"]
    pole_time = race_data["GRID"][0]["q_time"]
    fp1_times = race_data.get("FP1_TIMES", {})
    experience = race_data.get("DRIVER_EXPERIENCE", {}).get(name, {"f1_seasons": 0, "career_poles": 0})
    team_deficit = race_data.get("TEAM_PACE_DEFICIT", {}).get(team, 1.5)
    start_adv = race_data.get("START_PROCEDURE", {}).get(team, 0.0)
    energy = race_data.get("ENERGY_READINESS", {}).get(team, 0.4)
    circuit_history = race_data.get("CIRCUIT_HISTORY", {}).get(name, {"wins": 0, "podiums": 0})
    sprint_result = race_data.get("SPRINT_RESULT", [])

    if q_time is not None and pole_time is not None:
        quali_pace = max(0, 1.0 - ((q_time - pole_time) / 4.0))
    else:
        quali_pace = 0.15

    if grid_pos == 1: grid_win_rate = POLE_WIN_RATE
    elif grid_pos <= 3: grid_win_rate = POLE_WIN_RATE * (0.35 / grid_pos)
    elif grid_pos <= 6: grid_win_rate = POLE_WIN_RATE * (0.12 / (grid_pos - 1))
    elif grid_pos <= 10: grid_win_rate = POLE_WIN_RATE * (0.03 / (grid_pos - 3))
    else: grid_win_rate = max(0.001, 0.01 * (1 - (grid_pos - 10) / 12))

    race_pace = max(0, 1.0 - (team_deficit / 3.0))

    fp1_time = fp1_times.get(name)
    if fp1_time is not None:
        valid_times = [t for t in fp1_times.values() if t is not None]
        fp1_best = min(valid_times) if valid_times else fp1_time
        practice_pace = max(0, 1.0 - ((fp1_time - fp1_best) / 3.0))
    else:
        practice_pace = 0.3

    sprint_score = 0.3
    for sp in sprint_result:
        if sp["driver"] == name:
            sprint_score = max(0, 1.0 - (sp["pos"] - 1) / 10)
            break

    teammate_time = None
    for d in race_data["GRID"]:
        if d["team"] == team and d["driver"] != name and d["q_time"] is not None:
            teammate_time = d["q_time"]
            break
    if q_time is not None and teammate_time is not None:
        teammate_gap = min(1.0, max(0, (teammate_time - q_time + 1.0) / 2.0))
    elif q_time is not None:
        teammate_gap = 0.7
    else:
        teammate_gap = 0.2

    seasons = experience.get("f1_seasons", 0)
    reg_changes = sum([seasons >= 2, seasons >= 5, seasons >= 9, seasons >= 13])
    adaptability = min(1.0, reg_changes * 0.25)

    start_score = min(1.0, max(0, 0.5 + start_adv))

    r1_finish = experience.get("r1_finish")
    if r1_finish is not None and r1_finish <= 10: reliability = 0.95
    elif r1_finish is not None: reliability = 0.80
    else: reliability = 0.50

    track_score = min(1.0, circuit_history.get("wins", 0) * 0.3 + circuit_history.get("podiums", 0) * 0.1)

    return {
        "quali_pace": round(quali_pace, 4), "grid_win_rate": round(grid_win_rate, 4),
        "race_pace": round(race_pace, 4), "practice_pace": round(practice_pace, 4),
        "sprint_score": round(sprint_score, 4), "teammate_gap": round(teammate_gap, 4),
        "adaptability": round(adaptability, 4), "start_score": round(start_score, 4),
        "reliability": round(reliability, 4), "energy_score": round(energy, 4),
        "track_history": round(track_score, 4),
    }


def run_prediction(race_folder, config):
    race_path = os.path.join(RACES_DIR, race_folder)
    sys.path.insert(0, race_path)

    # Clear cached module if reloading
    if "data" in sys.modules:
        del sys.modules["data"]

    try:
        data_module = importlib.import_module("data")
    except Exception as e:
        return None, f"Could not load data.py: {e}"

    race_data = {
        "GRID": data_module.GRID,
        "FP1_TIMES": getattr(data_module, "FP1_TIMES", {}),
        "SPRINT_RESULT": getattr(data_module, "SPRINT_RESULT", []),
        "DRIVER_EXPERIENCE": getattr(data_module, "DRIVER_EXPERIENCE", {}),
        "TEAM_PACE_DEFICIT": getattr(data_module, "TEAM_PACE_DEFICIT", {}),
        "START_PROCEDURE": getattr(data_module, "START_PROCEDURE", {}),
        "ENERGY_READINESS": getattr(data_module, "ENERGY_READINESS", {}),
        "CIRCUIT_HISTORY": getattr(data_module, "CIRCUIT_HISTORY", {}),
    }
    race_info = getattr(data_module, "RACE_INFO", {"name": race_folder, "laps": 56})
    weights = config["weights"]

    predictions = []
    for entry in race_data["GRID"]:
        features = compute_features(entry, race_data, weights)
        raw_score = sum(features.get(k, 0) * w for k, w in weights.items())
        predictions.append({
            "driver": entry["driver"], "team": entry["team"],
            "grid_pos": entry["pos"], "features": features, "raw_score": raw_score,
        })

    temp = config["regulation_params"]["softmax_temperature"]
    scores = np.array([p["raw_score"] for p in predictions])
    exp_s = np.exp((scores - scores.max()) / temp)
    probs = exp_s / exp_s.sum()
    for i, p in enumerate(predictions):
        p["win_prob"] = float(probs[i])

    # Monte Carlo
    np.random.seed(42)
    n_sims = 100000
    all_drivers = [p["driver"] for p in predictions]
    all_teams = [p["team"] for p in predictions]
    win_c, pod_c, pts_c, dnf_c = (defaultdict(int) for _ in range(4))

    for _ in range(n_sims):
        base = np.array([p["win_prob"] for p in predictions])
        perf = np.random.normal(base, base * 0.35 + 0.015)
        for i in range(len(all_drivers)):
            perf[i] += np.random.normal(0, ENERGY_UNCERTAINTY)
            perf[i] += np.random.normal(0, NEW_TEAM_UNCERTAINTY.get(all_teams[i], 0.0))
        if np.random.random() < 0.50:
            lv = perf.max()
            perf = perf * 0.7 + lv * 0.3 + np.random.normal(0, 0.02, len(all_drivers))
        if np.random.random() < 0.25:
            perf = perf * 0.9 + np.random.uniform(0, 0.08, len(all_drivers))
        if np.random.random() < 0.10:
            for i, d in enumerate(all_drivers):
                s = race_data.get("DRIVER_EXPERIENCE", {}).get(d, {}).get("f1_seasons", 0)
                perf[i] += s * 0.002
            perf += np.random.normal(0, 0.04, len(all_drivers))
        if np.random.random() < 0.30:
            for _ in range(np.random.choice([1, 2], p=[0.6, 0.4])):
                v = np.random.randint(2, min(14, len(all_drivers)))
                perf[v] *= np.random.uniform(0.15, 0.55)
        for i in range(len(all_drivers)):
            gp = predictions[i]["grid_pos"]
            if gp > 5:
                perf[i] += (gp - 5) * 0.001 * OVERTAKE_FACTOR * np.random.uniform(0.3, 1.0)
        for i in range(1, len(all_drivers)):
            if np.random.random() < 0.15:
                perf[i] += 0.02
        for i in range(len(all_drivers)):
            dnf_rate = 0.07
            if all_teams[i] in ("Cadillac", "Audi"): dnf_rate = 0.11
            elif all_teams[i] == "Aston Martin": dnf_rate = 0.10
            elif all_teams[i] == "Red Bull": dnf_rate = 0.08
            if np.random.random() < dnf_rate: perf[i] = -1
        for i in range(len(all_drivers)):
            if np.random.random() < 0.03:
                perf[i] *= np.random.uniform(0.2, 0.6)
        for i in range(len(all_drivers)):
            perf[i] += np.random.normal(0, 0.012)

        ranking = np.argsort(-perf)
        finishers = [all_drivers[r] for r in ranking if perf[r] > 0]
        if finishers: win_c[finishers[0]] += 1
        for d in finishers[:3]: pod_c[d] += 1
        for d in finishers[:10]: pts_c[d] += 1
        for d in all_drivers:
            if d not in finishers: dnf_c[d] += 1

    results = []
    for p in predictions:
        d = p["driver"]
        results.append({
            "driver": d, "team": p["team"], "grid_pos": p["grid_pos"],
            "win_pct": round(win_c[d] / n_sims * 100, 2),
            "podium_pct": round(pod_c[d] / n_sims * 100, 2),
            "points_pct": round(pts_c[d] / n_sims * 100, 2),
            "dnf_pct": round(dnf_c[d] / n_sims * 100, 2),
            "model_score": round(p["raw_score"], 4),
            "features": p["features"],
        })
    results.sort(key=lambda x: x["win_pct"], reverse=True)

    output = {"race": race_info, "simulations": n_sims, "predictions": results}
    with open(os.path.join(race_path, "prediction.json"), "w") as f:
        json.dump(output, f, indent=2)

    return output, None


def calibrate_after_race(race_round, config):
    race_folder = None
    for f in get_race_folders():
        if f.startswith(f"{race_round:02d}_"):
            race_folder = f
            break
    if not race_folder:
        return config, "Race folder not found"

    pred_path = os.path.join(RACES_DIR, race_folder, "prediction.json")
    result_path = os.path.join(RACES_DIR, race_folder, "result.json")
    if not os.path.exists(pred_path) or not os.path.exists(result_path):
        return config, "Missing prediction or result file"

    with open(pred_path) as f: pred_data = json.load(f)
    with open(result_path) as f: result_data = json.load(f)

    actual_positions = {}
    for r in result_data["result"]:
        if r.get("pos") is not None:
            actual_positions[r["driver"]] = r["pos"]

    weights = config["weights"]
    completed = config.get("last_calibrated_after_round", 0)
    lr = 0.05 / (1 + completed * 0.3)
    adjustments = {k: 0.0 for k in weights}

    for pred in pred_data["predictions"][:10]:
        driver = pred["driver"]
        pred_rank = pred_data["predictions"].index(pred) + 1
        actual_rank = actual_positions.get(driver)
        if actual_rank is None:
            continue
        rank_error = pred_rank - actual_rank
        features = pred.get("features", {})
        for feat_name, feat_value in features.items():
            if feat_name in adjustments:
                if rank_error > 0 and feat_value > 0.5:
                    adjustments[feat_name] -= lr * feat_value * 0.1
                elif rank_error < 0 and feat_value > 0.5:
                    adjustments[feat_name] += lr * feat_value * 0.1

    for feat, adj in adjustments.items():
        weights[feat] = max(0.01, weights[feat] + adj)
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}
    config["weights"] = weights
    config["last_calibrated_after_round"] = race_round

    pred_winner = pred_data["predictions"][0]["driver"]
    actual_winner = next((r["driver"] for r in result_data["result"] if r.get("pos") == 1), "Unknown")
    pred_podium = [p["driver"] for p in pred_data["predictions"][:3]]
    actual_podium = [r["driver"] for r in result_data["result"] if r.get("pos") and r["pos"] <= 3]
    overlap = len(set(pred_podium) & set(actual_podium))

    errors = []
    for pred in pred_data["predictions"]:
        ap = actual_positions.get(pred["driver"])
        if ap:
            pp = pred_data["predictions"].index(pred) + 1
            errors.append(abs(pp - ap))

    entry = {
        "round": race_round,
        "race": race_folder.split("_", 1)[1].replace("_", " ").title(),
        "predicted_winner": pred_winner,
        "predicted_win_pct": pred_data["predictions"][0]["win_pct"],
        "actual_winner": actual_winner,
        "correct": pred_winner == actual_winner,
        "podium_predicted": pred_podium,
        "podium_actual": actual_podium,
        "podium_overlap": overlap,
        "mean_position_error": round(np.mean(errors), 2) if errors else None,
    }

    history = config.get("accuracy_history", [])
    existing = [i for i, h in enumerate(history) if h["round"] == race_round]
    if existing:
        history[existing[0]] = entry
    else:
        history.append(entry)
    config["accuracy_history"] = history
    save_config(config)
    return config, None


# ============================================================
# STREAMLIT UI
# ============================================================

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0a0a1a; }
    .block-container { max-width: 1200px; padding-top: 1rem; }
    h1, h2, h3 { font-family: monospace; }
    .stMetric label { font-size: 11px !important; letter-spacing: 2px; }
    .stMetric [data-testid="stMetricValue"] { font-family: monospace; }
    div[data-testid="stSidebar"] { background-color: #111128; }
    .winner-card {
        background: linear-gradient(135deg, rgba(0,210,190,0.1) 0%, rgba(0,210,190,0.02) 100%);
        border: 1px solid rgba(0,210,190,0.3);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .result-correct { color: #00D2BE; font-weight: 900; }
    .result-wrong { color: #DC0000; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

config = load_config()
race_folders = get_race_folders()

# SIDEBAR
with st.sidebar:
    st.markdown("### 🏎️ F1 2026 PREDICTOR")
    st.markdown(f"**Model:** {config.get('model_version', 'v4')}")

    history = config.get("accuracy_history", [])
    correct = sum(1 for h in history if h.get("correct"))
    st.markdown(f"**Winner accuracy:** {correct}/{len(history)}")
    st.markdown(f"**Calibrated after:** Round {config.get('last_calibrated_after_round', 0)}")
    st.markdown("---")

    page = st.radio("Navigate", ["🏁 Race Prediction", "📊 Accuracy Tracker", "⚖️ Model Weights", "📅 Season Calendar", "🔧 Run Prediction", "📝 Submit Result"])

# ============================================================
# PAGE: RACE PREDICTION
# ============================================================
if page == "🏁 Race Prediction":
    st.title("Race Prediction")

    race_options = {f: f.replace("_", " ").title() for f in race_folders}
    selected = st.selectbox("Select Race", list(race_options.keys()), format_func=lambda x: race_options[x])

    pred = load_prediction(selected)
    result = load_result(selected)

    if pred is None:
        st.warning(f"No prediction found for {selected}. Go to 'Run Prediction' to generate one.")
    else:
        predictions = pred["predictions"]
        winner = predictions[0]
        race_info = pred.get("race", {})

        # Winner card
        actual_winner = None
        if result:
            actual_winner = next((r["driver"] for r in result["result"] if r.get("pos") == 1), None)

        col1, col2 = st.columns([3, 1])
        with col1:
            status = ""
            if actual_winner:
                if actual_winner == winner["driver"]:
                    status = "  ✅ CORRECT"
                else:
                    status = f"  ❌ Actual: {actual_winner}"

            st.markdown(f"""
            <div class="winner-card">
                <div style="font-size:10px;letter-spacing:3px;color:#00D2BE;">PREDICTED WINNER{status}</div>
                <div style="font-size:32px;font-weight:900;color:white;font-family:monospace;">{winner['driver']}</div>
                <div style="font-size:14px;color:{TEAM_COLORS.get(winner['team'], '#888')};">{winner['team']} | P{winner['grid_pos']} on Grid | DNF Risk: {winner['dnf_pct']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="text-align:right;padding-top:20px;">
                <div style="font-size:52px;font-weight:900;color:#00D2BE;font-family:monospace;line-height:1;">{winner['win_pct']}%</div>
                <div style="font-size:10px;color:#888;letter-spacing:2px;">WIN PROBABILITY</div>
                <div style="font-size:9px;color:#555;margin-top:4px;">100K Monte Carlo sims</div>
            </div>
            """, unsafe_allow_html=True)

        # Win probability chart
        top10 = predictions[:10]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[d["driver"].split()[-1] for d in top10],
            y=[d["win_pct"] for d in top10],
            marker_color=[TEAM_COLORS.get(d["team"], "#666") for d in top10],
            marker_opacity=0.85,
            text=[f"{d['win_pct']}%" for d in top10],
            textposition="outside",
            textfont=dict(size=11, color="#e0e0e0"),
        ))
        fig.update_layout(
            title="Win Probability (Top 10)",
            paper_bgcolor="#0a0a1a", plot_bgcolor="#111128",
            font=dict(family="monospace", color="#e0e0e0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Win %"),
            margin=dict(t=40, b=40), height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Podium and DNF charts side by side
        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=[d["driver"].split()[-1] for d in top10],
                y=[d["podium_pct"] for d in top10],
                marker_color=[TEAM_COLORS.get(d["team"], "#666") for d in top10],
                marker_opacity=0.6,
            ))
            fig2.update_layout(
                title="Podium Probability", paper_bgcolor="#0a0a1a", plot_bgcolor="#111128",
                font=dict(family="monospace", color="#e0e0e0", size=10),
                yaxis=dict(title="Podium %", gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(t=40, b=40), height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=[d["driver"].split()[-1] for d in top10],
                y=[d["dnf_pct"] for d in top10],
                marker_color=["#DC0000" if d["dnf_pct"] > 15 else "#FF8700" if d["dnf_pct"] > 10 else "#555" for d in top10],
                marker_opacity=0.7,
            ))
            fig3.update_layout(
                title="DNF Risk", paper_bgcolor="#0a0a1a", plot_bgcolor="#111128",
                font=dict(family="monospace", color="#e0e0e0", size=10),
                yaxis=dict(title="DNF %", gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(t=40, b=40), height=280,
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Full grid table
        st.markdown("#### Full Grid Predictions")
        table_data = []
        for i, p in enumerate(predictions):
            row = {
                "#": i + 1,
                "Driver": p["driver"],
                "Team": p["team"],
                "Grid": f"P{p['grid_pos']}",
                "Win%": p["win_pct"],
                "Podium%": p["podium_pct"],
                "DNF%": p["dnf_pct"],
            }
            if result:
                actual = next((r for r in result["result"] if r["driver"] == p["driver"]), None)
                if actual and actual.get("pos"):
                    row["Actual"] = f"P{actual['pos']}"
                elif actual:
                    row["Actual"] = actual.get("status", "DNF")[:10]
                else:
                    row["Actual"] = "?"
            table_data.append(row)
        st.dataframe(table_data, use_container_width=True, hide_index=True)

# ============================================================
# PAGE: ACCURACY TRACKER
# ============================================================
elif page == "📊 Accuracy Tracker":
    st.title("Accuracy Tracker")

    history = config.get("accuracy_history", [])
    if not history:
        st.info("No race results submitted yet. Submit results to start tracking accuracy.")
    else:
        correct = sum(1 for h in history if h.get("correct"))
        total = len(history)
        avg_podium = np.mean([h.get("podium_overlap", 0) for h in history])
        errors = [h["mean_position_error"] for h in history if h.get("mean_position_error")]
        avg_error = np.mean(errors) if errors else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Races", total)
        c2.metric("Winner Accuracy", f"{correct}/{total} ({correct/total*100:.0f}%)")
        c3.metric("Avg Podium Overlap", f"{avg_podium:.1f}/3")
        c4.metric("Avg Position Error", f"{avg_error:.1f}" if avg_error else "N/A")

        st.markdown("---")
        st.markdown("#### Race-by-Race Results")

        for h in history:
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**R{h['round']} {h['race']}**")
                st.caption(f"Predicted: {h['predicted_winner']} ({h['predicted_win_pct']}%)")
            with col2:
                st.markdown(f"Actual: **{h['actual_winner']}**")
                st.caption(f"Podium overlap: {h.get('podium_overlap', '?')}/3")
            with col3:
                if h.get("correct"):
                    st.markdown('<span class="result-correct">✅ CORRECT</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="result-wrong">❌ WRONG</span>', unsafe_allow_html=True)
            st.markdown("---")

# ============================================================
# PAGE: MODEL WEIGHTS
# ============================================================
elif page == "⚖️ Model Weights":
    st.title("Self-Calibrating Model Weights")
    st.caption(f"Last calibrated after Round {config.get('last_calibrated_after_round', 0)}. Weights auto-adjust after each race result.")

    weights = config["weights"]
    sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[w[0].replace("_", " ") for w in sorted_w],
        x=[round(w[1] * 100, 1) for w in sorted_w],
        orientation="h",
        marker_color="#00D2BE",
        marker_opacity=0.7,
        text=[f"{round(w[1]*100, 1)}%" for w in sorted_w],
        textposition="outside",
        textfont=dict(color="#e0e0e0", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a1a", plot_bgcolor="#111128",
        font=dict(family="monospace", color="#e0e0e0"),
        xaxis=dict(title="Weight %", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=120, t=20, b=40), height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 2026 Regulation Parameters")
    reg = config.get("regulation_params", {})
    for k, v in reg.items():
        st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")

# ============================================================
# PAGE: SEASON CALENDAR
# ============================================================
elif page == "📅 Season Calendar":
    st.title("2026 Season Calendar")

    calendar = config.get("calendar_2026", [])
    for race in calendar:
        folder = f"{race['round']:02d}_{race['name'].lower().replace(' grand prix', '').replace(' ', '_')}"
        has_pred = os.path.exists(os.path.join(RACES_DIR, folder, "prediction.json"))
        has_result = os.path.exists(os.path.join(RACES_DIR, folder, "result.json"))

        col1, col2, col3 = st.columns([4, 2, 2])
        with col1:
            st.markdown(f"**R{race['round']} {race['name']}**")
            st.caption(f"{race['circuit']} | {race['date']} | {race['format']}")
        with col2:
            if has_pred:
                st.success("Predicted", icon="🏁")
            else:
                st.caption("No prediction")
        with col3:
            if has_result:
                st.success("Result in", icon="✅")
            else:
                st.caption("Awaiting result")

# ============================================================
# PAGE: RUN PREDICTION
# ============================================================
elif page == "🔧 Run Prediction":
    st.title("Run Prediction")
    st.caption("Select a race with data.py prepared. The model will generate a prediction.")

    available = [f for f in race_folders if os.path.exists(os.path.join(RACES_DIR, f, "data.py"))]

    if not available:
        st.warning("No races have data.py files. Add qualifying data to a race folder first.")
    else:
        selected = st.selectbox("Race to predict", available, format_func=lambda x: x.replace("_", " ").title())

        if st.button("🏎️ Run Prediction (100K simulations)", type="primary"):
            with st.spinner("Running 100,000 Monte Carlo simulations..."):
                result, error = run_prediction(selected, config)
            if error:
                st.error(error)
            else:
                winner = result["predictions"][0]
                st.success(f"Prediction complete. Winner: **{winner['driver']}** ({winner['win_pct']}%)")
                st.balloons()

# ============================================================
# PAGE: SUBMIT RESULT
# ============================================================
elif page == "📝 Submit Result":
    st.title("Submit Race Result")
    st.caption("After the race, enter the finishing order. The model will auto-calibrate its weights.")

    completed = [f for f in race_folders if os.path.exists(os.path.join(RACES_DIR, f, "prediction.json"))]
    if not completed:
        st.warning("No predictions to validate. Run a prediction first.")
    else:
        selected = st.selectbox("Race", completed, format_func=lambda x: x.replace("_", " ").title())
        race_round = int(selected.split("_")[0])

        result_path = os.path.join(RACES_DIR, selected, "result.json")
        if os.path.exists(result_path):
            st.info("Result already submitted for this race. Submitting again will recalibrate weights.")

        st.markdown("#### Enter Top 10 Finishing Order")
        pred = load_prediction(selected)
        driver_names = [p["driver"] for p in pred["predictions"]] if pred else []

        top10 = []
        for i in range(1, 11):
            d = st.selectbox(f"P{i}", [""] + driver_names, key=f"pos_{i}")
            if d:
                top10.append({"pos": i, "driver": d, "team": next((p["team"] for p in pred["predictions"] if p["driver"] == d), ""), "status": "Finished"})

        if st.button("Submit Result and Calibrate", type="primary"):
            if len(top10) < 3:
                st.error("Enter at least the top 3 finishers.")
            else:
                result_data = {"result": top10}
                with open(result_path, "w") as f:
                    json.dump(result_data, f, indent=2)

                with st.spinner("Calibrating weights..."):
                    new_config, error = calibrate_after_race(race_round, config)

                if error:
                    st.error(error)
                else:
                    st.success("Result saved and weights calibrated.")
                    correct = sum(1 for h in new_config["accuracy_history"] if h.get("correct"))
                    total = len(new_config["accuracy_history"])
                    st.metric("Updated Winner Accuracy", f"{correct}/{total}")
