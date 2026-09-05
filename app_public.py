"""
F1 2026 Race Predictor: Public Dashboard
Read-only version for Streamlit Community Cloud.
Loads saved prediction.json and result.json files. No engine, no FastF1, no local paths.
"""

import streamlit as st
import json
import os
import glob

st.set_page_config(page_title="F1 2026 Predictor", page_icon="🏎️", layout="wide")

TEAM_COLORS = {
    "Mercedes": "#00D2BE", "Ferrari": "#DC0000", "McLaren": "#FF8700",
    "Red Bull": "#3671C6", "Racing Bulls": "#6692FF", "Audi": "#FF0000",
    "Haas": "#B6BABD", "Alpine": "#0090FF", "Williams": "#005AFF",
    "Aston Martin": "#006F62", "Cadillac": "#1E1E1E",
}

st.markdown("""<style>
    .block-container { max-width: 1200px; padding-top: 3rem; }
</style>""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING (relative paths, works in the cloud)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RACES_DIR = os.path.join(BASE_DIR, "races")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_race_folders():
    if not os.path.isdir(RACES_DIR):
        return []
    folders = [os.path.basename(p) for p in glob.glob(os.path.join(RACES_DIR, "*")) if os.path.isdir(p)]
    return sorted(folders)


def load_prediction(folder):
    return load_json(os.path.join(RACES_DIR, folder, "prediction.json"))


def load_result(folder):
    return load_json(os.path.join(RACES_DIR, folder, "result.json"))


def load_config():
    cfg = load_json(os.path.join(BASE_DIR, "config.json"))
    return cfg if cfg else {"accuracy_history": [], "last_calibrated_after_round": 0}


def pretty_name(folder):
    parts = folder.split("_", 1)
    if len(parts) == 2:
        return parts[1].replace("_", " ").title()
    return folder.title()


# Full race names and circuits for the banner, keyed by folder.
RACE_META = {
    "01_australia":  ("Australian Grand Prix", "Albert Park, Melbourne"),
    "02_china":      ("Chinese Grand Prix", "Shanghai International Circuit"),
    "03_japan":      ("Japanese Grand Prix", "Suzuka Circuit"),
    "04_miami":      ("Miami Grand Prix", "Miami International Autodrome"),
    "05_canada":     ("Canadian Grand Prix", "Circuit Gilles Villeneuve"),
    "06_monaco":     ("Monaco Grand Prix", "Circuit de Monaco"),
    "07_barcelona":  ("Barcelona-Catalunya Grand Prix", "Circuit de Barcelona-Catalunya"),
    "08_austria":    ("Austrian Grand Prix", "Red Bull Ring"),
    "09_britain":    ("British Grand Prix", "Silverstone Circuit"),
    "10_belgium":    ("Belgian Grand Prix", "Spa-Francorchamps"),
    "11_hungary":    ("Hungarian Grand Prix", "Hungaroring"),
    "12_netherlands": ("Dutch Grand Prix", "Circuit Zandvoort"),
    "13_italy":      ("Italian Grand Prix", "Monza"),
}


def race_banner_text(folder, race_info):
    """Prefer race_info from prediction.json, then RACE_META, then folder name."""
    name = race_info.get("name") if race_info else None
    circuit = race_info.get("circuit") if race_info else None
    date = race_info.get("date") if race_info else None
    if not name and folder in RACE_META:
        name, circuit = RACE_META[folder]
    if not name:
        name = pretty_name(folder)
    parts = [name]
    if circuit:
        parts.append(circuit)
    if date:
        parts.append(date)
    return " | ".join(parts)


def pole_sitter(prediction):
    """Driver who started P1, read from prediction.json rather than data.py."""
    if not prediction:
        return None
    for entry in prediction.get("predictions", []):
        if entry.get("grid_pos") == 1:
            return entry.get("driver")
    return None


def baseline_record(folders):
    """How often the pole sitter won, over races that have a result.

    This is the naive strategy the model has to beat. Showing it next to the
    model's own record is the honest comparison: without it, a hit rate is a
    number with nothing to measure against.
    """
    hits = raced = 0
    for folder in folders:
        result = load_result(folder)
        if not result or not result.get("result"):
            continue
        pole = pole_sitter(load_prediction(folder))
        if not pole:
            continue
        raced += 1
        if result["result"][0].get("driver") == pole:
            hits += 1
    return hits, raced


config = load_config()
folders = get_race_folders()
predicted_races = [f for f in folders if load_prediction(f)]
history = config.get("accuracy_history", [])
mc_correct = sum(1 for h in history if h.get("correct"))


# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div style="background:linear-gradient(90deg,rgba(220,0,0,0.08),rgba(0,210,190,0.08),rgba(54,113,198,0.08));
            border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:20px 28px;margin-bottom:20px;
            display:flex;justify-content:space-between;align-items:center;">
    <div>
        <div style="font-size:9px;letter-spacing:3px;color:#555;">SELF-CALIBRATING ML MODEL</div>
        <div style="font-size:28px;font-weight:900;color:white;font-family:monospace;">F1 2026 RACE PREDICTOR</div>
        <div style="font-size:11px;color:#666;">100K Monte Carlo + XGBoost | Zero betting data | 2026 regulation-aware</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:9px;letter-spacing:2px;color:#555;">RACES PREDICTED</div>
        <div style="font-size:32px;font-weight:900;color:#00D2BE;font-family:monospace;">{len(predicted_races)}</div>
        <div style="font-size:9px;color:#555;">Calibrated after R{config.get('last_calibrated_after_round', 0)} | accuracy in Season tab</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab_race, tab_season = st.tabs(["Race Prediction", "Season Performance"])


# ============================================================
# TAB 1: RACE PREDICTION
# ============================================================
with tab_race:
    if not predicted_races:
        st.warning("No predictions found. Add prediction.json files to the races folders.")
    else:
        labels = {f: (RACE_META.get(f, (pretty_name(f),))[0]) for f in predicted_races}
        selected = st.selectbox(
            "Select Race",
            predicted_races,
            index=len(predicted_races) - 1,
            format_func=lambda f: labels[f],
        )

        pred = load_prediction(selected)
        result = load_result(selected)
        predictions = pred["predictions"]
        winner = predictions[0]
        xgb = pred.get("xgboost", {})
        xgb_top = xgb["predictions"][0] if (xgb and xgb.get("available")) else None

        actual_winner = None
        if result and result.get("result"):
            actual_winner = result["result"][0]["driver"]

        race_info = pred.get("race_info", {})

        # Race banner
        st.markdown(f"""
        <div style="background:#0a0a1a;border:1px solid rgba(255,255,255,0.08);
                    border-radius:10px;padding:14px 20px;margin-bottom:16px;">
            <div style="font-size:11px;color:#555;letter-spacing:2px;">RACE</div>
            <div style="font-size:18px;font-weight:700;color:white;font-family:monospace;">
                {race_banner_text(selected, race_info)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Status badges
        mc_badge = ""
        xgb_badge = ""
        if actual_winner:
            if actual_winner == winner["driver"]:
                mc_badge = '<div style="font-size:11px;color:#00ff88;font-weight:900;letter-spacing:2px;margin-top:6px;">CORRECT</div>'
            else:
                mc_badge = '<div style="font-size:11px;color:#ff5555;font-weight:900;letter-spacing:2px;margin-top:6px;">MISS</div>'
            if xgb_top:
                if actual_winner == xgb_top["driver"]:
                    xgb_badge = '<div style="font-size:11px;color:#00ff88;font-weight:900;letter-spacing:2px;margin-top:6px;">CORRECT</div>'
                else:
                    xgb_badge = '<div style="font-size:11px;color:#ff5555;font-weight:900;letter-spacing:2px;margin-top:6px;">MISS</div>'

        # Dual winner cards
        col_mc, col_xgb = st.columns(2)
        with col_mc:
            mc_color = TEAM_COLORS.get(winner["team"], "#00D2BE")
            st.markdown(f"""
            <div style="background:#111128;border:2px solid {mc_color}88;border-radius:12px;padding:24px;text-align:center;height:340px;">
                <div style="font-size:10px;letter-spacing:3px;color:#00D2BE;font-weight:900;">MONTE CARLO WINNER</div>
                <div style="font-size:9px;color:#666;margin-top:2px;">100K SIMULATIONS</div>
                <div style="font-size:46px;margin-top:8px;">🥇</div>
                <div style="font-size:22px;font-weight:900;color:white;font-family:monospace;margin-top:6px;">{winner['driver']}</div>
                <div style="font-size:13px;color:{mc_color};margin-top:2px;">{winner['team']}</div>
                <div style="font-size:44px;font-weight:900;color:{mc_color};font-family:monospace;margin-top:10px;line-height:1;">{winner['win_pct']}%</div>
                <div style="font-size:11px;color:#888;">P{winner['grid_pos']} grid | {winner.get('podium_pct','')}% podium | {winner.get('dnf_pct','')}% DNF</div>
                {mc_badge}
            </div>
            """, unsafe_allow_html=True)

        with col_xgb:
            if xgb_top:
                xgb_color = TEAM_COLORS.get(xgb_top["team"], "#FFD700")
                xgb_win_pct = round(xgb_top["win_prob"] * 100, 2)
                mae = xgb.get("mae", "n/a")
                rows = xgb.get("trained_rows", 0)
                st.markdown(f"""
                <div style="background:#111128;border:2px solid {xgb_color}88;border-radius:12px;padding:24px;text-align:center;height:340px;">
                    <div style="font-size:10px;letter-spacing:3px;color:#FFD700;font-weight:900;">XGBOOST WINNER</div>
                    <div style="font-size:9px;color:#666;margin-top:2px;">{rows} TRAINING ROWS | MAE {mae}</div>
                    <div style="font-size:46px;margin-top:8px;">🥇</div>
                    <div style="font-size:22px;font-weight:900;color:white;font-family:monospace;margin-top:6px;">{xgb_top['driver']}</div>
                    <div style="font-size:13px;color:{xgb_color};margin-top:2px;">{xgb_top['team']}</div>
                    <div style="font-size:44px;font-weight:900;color:#FFD700;font-family:monospace;margin-top:10px;line-height:1;">{xgb_win_pct}%</div>
                    <div style="font-size:11px;color:#888;">P{xgb_top['grid_pos']} grid | predicted finish P{xgb_top['predicted_position']}</div>
                    {xgb_badge}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#111128;border:1px dashed rgba(255,215,0,0.3);border-radius:12px;padding:24px;text-align:center;height:340px;display:flex;flex-direction:column;justify-content:center;">
                    <div style="font-size:10px;letter-spacing:3px;color:#FFD700;font-weight:900;">XGBOOST</div>
                    <div style="font-size:46px;margin-top:14px;opacity:0.3;">🥇</div>
                    <div style="font-size:14px;color:#888;margin-top:14px;">Not enough training data yet</div>
                </div>
                """, unsafe_allow_html=True)

        # Actual winner banner
        if actual_winner:
            actual_team = next((r.get("team") for r in result["result"] if r["driver"] == actual_winner), "")
            actual_color = TEAM_COLORS.get(actual_team, "#FFD700")
            st.markdown(f"""
            <div style="background:linear-gradient(90deg,rgba(255,215,0,0.18),rgba(255,215,0,0.04));
                        border:2px solid rgba(255,215,0,0.5);border-radius:10px;
                        padding:18px 28px;margin-top:18px;text-align:center;">
                <div style="font-size:11px;letter-spacing:4px;color:#FFD700;font-weight:900;">ACTUAL RACE WINNER</div>
                <div style="font-size:32px;font-weight:900;color:white;font-family:monospace;margin-top:4px;">{actual_winner}</div>
                <div style="font-size:14px;color:{actual_color};">{actual_team}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# TAB 2: SEASON PERFORMANCE
# ============================================================
with tab_season:
    if not history:
        st.info("No race history yet.")
    else:
        # Summary metrics
        total = len(history)
        mc_rate = round(100 * mc_correct / total, 1) if total else 0
        avg_podium = round(sum(h.get("podium_overlap", 0) for h in history) / total, 2) if total else 0

        pole_hits, pole_races = baseline_record(folders)
        pole_rate = round(100 * pole_hits / pole_races, 1) if pole_races else 0
        edge = round(mc_rate - pole_rate, 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Races Predicted", total)
        c2.metric("Monte Carlo Winners", f"{mc_correct}/{total}", f"{mc_rate}%")
        c3.metric("Always-Pole Baseline", f"{pole_hits}/{pole_races}", f"{pole_rate}%")
        c4.metric("Avg Podium Drivers Hit", f"{avg_podium}/3")

        if pole_races:
            verdict = "ahead of" if edge > 0 else "behind" if edge < 0 else "level with"
            st.caption(
                f"The model is {verdict} the naive strategy of always picking the pole sitter, "
                f"by {abs(edge)} points. At {pole_races} races that gap is well inside noise, "
                f"so neither number supports a claim yet. Beating this baseline over a full "
                f"season is the bar the model has to clear."
            )

        st.markdown("")

        # Running accuracy chart
        rounds = [h.get("round") for h in history]
        running = []
        hits = 0
        for i, h in enumerate(history, 1):
            if h.get("correct"):
                hits += 1
            running.append(round(100 * hits / i, 1))

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rounds, y=running, mode="lines+markers",
                name="Running win rate", line=dict(color="#00D2BE", width=3),
                marker=dict(size=8),
            ))
            fig.update_layout(
                title="Monte Carlo Running Winner Accuracy",
                xaxis_title="Round", yaxis_title="Win rate %",
                yaxis=dict(range=[0, 100]),
                template="plotly_dark", height=360,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.line_chart({"Win rate %": running})

        st.markdown("")

        # Race-by-race table
        st.markdown("Race by race history")
        header = "| Round | Race | Predicted | Actual | Result | Podium Hits |\n| --- | --- | --- | --- | --- | --- |\n"
        rows_md = ""
        for h in history:
            mark = "Correct" if h.get("correct") else "Miss"
            rows_md += f"| {h.get('round')} | {h.get('race','')} | {h.get('predicted_winner','')} | {h.get('actual_winner','')} | {mark} | {h.get('podium_overlap','')}/3 |\n"
        st.markdown(header + rows_md)


st.markdown("")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:12px;'>"
    "Built by Brinda Bhanderi | Predictions committed to GitHub before each race | "
    "<a href='https://github.com/brinda0301/F1-predictions' style='color:#00D2BE;'>Repo</a>"
    "</div>",
    unsafe_allow_html=True,
)