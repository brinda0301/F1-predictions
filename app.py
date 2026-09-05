"""
F1 2026 Race Predictor
One command: streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import json
import os
import numpy as np

from engine import (predict, calibrate, load_config,
                     get_race_folders, load_prediction, load_result, has_data)

st.set_page_config(page_title="F1 2026 Predictor", page_icon="🏎️", layout="wide")

TEAM_COLORS = {
    "Mercedes": "#00D2BE", "Ferrari": "#DC0000", "McLaren": "#FF8700",
    "Red Bull": "#3671C6", "Racing Bulls": "#6692FF", "Audi": "#FF0000",
    "Haas": "#B6BABD", "Alpine": "#0090FF", "Williams": "#005AFF",
    "Aston Martin": "#006F62", "Cadillac": "#1E1E1E",
}

st.markdown("""<style>
    .block-container { max-width: 1200px; padding-top: 3rem; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-family: monospace; }
</style>""", unsafe_allow_html=True)

def pole_sitter(prediction):
    """Driver who started P1, read from prediction.json rather than data.py."""
    if not prediction:
        return None
    for entry in prediction.get("predictions", []):
        if entry.get("grid_pos") == 1:
            return entry.get("driver")
    return None


def baseline_record(race_folders):
    """How often the pole sitter won, over races that have a result.

    This is the naive strategy the model has to beat. Showing it next to the
    model's own record is the honest comparison: without it, a hit rate is a
    number with nothing to measure against.
    """
    hits = raced = 0
    for folder in race_folders:
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
correct = sum(1 for h in history if h.get("correct"))

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
        <div style="font-size:9px;color:#555;">Calibrated after R{config.get('last_calibrated_after_round', 0)} | accuracy below</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# RACE SELECTOR (one dropdown at the top)
# ============================================================
if not predicted_races:
    st.warning("No predictions yet. Select a race below and click 'Run Prediction'.")
    available = [f for f in folders if has_data(f)]
    if available:
        selected = st.selectbox("Race with qualifying data", available,
                                 format_func=lambda x: x.replace("_", " ").title())
        if st.button("🏎️ Run 100K Simulations", type="primary"):
            with st.spinner("Running 100,000 Monte Carlo simulations + XGBoost..."):
                predict(selected, config)
            st.balloons()
            st.rerun()
    st.stop()

selected = st.selectbox("Select Race", predicted_races,
                          index=len(predicted_races) - 1,
                          format_func=lambda x: x.replace("_", " ").title())

pred = load_prediction(selected)
result = load_result(selected)
predictions = pred["predictions"]
winner = predictions[0]
race_info = pred.get("race", {})
xgb = pred.get("xgboost")
models_agree = pred.get("models_agree")

actual_winner = None
if result:
    actual_winner = next((r["driver"] for r in result["result"] if r.get("pos") == 1), None)

# ============================================================
# SECTION 1: DUAL WINNER MEDAL CARDS (both models, podium style)
# ============================================================

# Race info banner
st.markdown(f"""
<div style="background:#0a0a1a;border:1px solid rgba(255,255,255,0.08);
            border-radius:10px;padding:14px 20px;margin-bottom:16px;">
    <div style="font-size:11px;color:#555;letter-spacing:2px;">RACE</div>
    <div style="font-size:18px;font-weight:700;color:white;font-family:monospace;">
        {race_info.get('name','')} | {race_info.get('circuit','')} | {race_info.get('date','')}
    </div>
</div>
""", unsafe_allow_html=True)

# Compute status badges for both models
mc_winner_driver = winner["driver"]
xgb_top = xgb["predictions"][0] if (xgb and xgb.get("available")) else None
xgb_winner_driver = xgb_top["driver"] if xgb_top else None

mc_badge = ""
xgb_badge = ""
if actual_winner:
    if actual_winner == mc_winner_driver:
        mc_badge = '<div style="font-size:11px;color:#00ff88;font-weight:900;letter-spacing:2px;margin-top:6px;">✅ CORRECT</div>'
    else:
        mc_badge = '<div style="font-size:11px;color:#ff5555;font-weight:900;letter-spacing:2px;margin-top:6px;">❌ WRONG</div>'
    if xgb_winner_driver:
        if actual_winner == xgb_winner_driver:
            xgb_badge = '<div style="font-size:11px;color:#00ff88;font-weight:900;letter-spacing:2px;margin-top:6px;">✅ CORRECT</div>'
        else:
            xgb_badge = '<div style="font-size:11px;color:#ff5555;font-weight:900;letter-spacing:2px;margin-top:6px;">❌ WRONG</div>'

# Two big medal winner cards, side by side
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
        <div style="font-size:11px;color:#888;">P{winner['grid_pos']} grid | {winner['podium_pct']}% podium | {winner['dnf_pct']}% DNF</div>
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
        st.markdown(f"""
        <div style="background:#111128;border:1px dashed rgba(255,215,0,0.3);border-radius:12px;padding:24px;text-align:center;height:340px;display:flex;flex-direction:column;justify-content:center;">
            <div style="font-size:10px;letter-spacing:3px;color:#FFD700;font-weight:900;">XGBOOST</div>
            <div style="font-size:46px;margin-top:14px;opacity:0.3;">🥇</div>
            <div style="font-size:14px;color:#888;margin-top:14px;">Not enough training data yet</div>
        </div>
        """, unsafe_allow_html=True)

# Actual winner banner (only shown after race is submitted)
if actual_winner:
    actual_team = next((r.get("team") for r in result["result"] if r["driver"] == actual_winner), "")
    actual_color = TEAM_COLORS.get(actual_team, "#FFD700")
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,rgba(255,215,0,0.18),rgba(255,215,0,0.04));
                border:2px solid rgba(255,215,0,0.5);border-radius:10px;
                padding:18px 28px;margin-top:18px;text-align:center;">
        <div style="font-size:11px;letter-spacing:4px;color:#FFD700;font-weight:900;">🏁 ACTUAL RACE WINNER</div>
        <div style="font-size:32px;font-weight:900;color:white;font-family:monospace;margin-top:4px;">
            {actual_winner}
        </div>
        <div style="font-size:14px;color:{actual_color};">{actual_team}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ============================================================
# SECTION 1.5: TOP 3 PODIUM CARDS (Monte Carlo)
# ============================================================
top3 = predictions[:3]
p1, p2, p3 = top3[0], top3[1], top3[2]

def get_actual(driver, result_data):
    if not result_data:
        return ""
    ap = next((r for r in result_data["result"] if r["driver"] == driver["driver"]), None)
    if ap and ap.get("pos"):
        return f"<div style='font-size:11px;color:#FFD700;margin-top:6px;'>Actual: P{ap['pos']}</div>"
    elif ap:
        return f"<div style='font-size:11px;color:#DC0000;margin-top:6px;'>{ap.get('status','DNF')}</div>"
    return ""

def card_html(driver, medal, size, result_data):
    c = TEAM_COLORS.get(driver["team"], "#666")
    actual = get_actual(driver, result_data)
    if size == "big":
        return f"<div style='background:#111128;border:2px solid {c}66;border-radius:12px;padding:24px;text-align:center;'><div style='font-size:36px;'>{medal}</div><div style='font-size:22px;font-weight:900;color:white;font-family:monospace;margin-top:4px;'>{driver['driver']}</div><div style='font-size:13px;color:{c};margin-top:2px;'>{driver['team']}</div><div style='font-size:42px;font-weight:900;color:{c};font-family:monospace;margin-top:10px;'>{driver['win_pct']}%</div><div style='font-size:11px;color:#888;'>P{driver['grid_pos']} grid | {driver['podium_pct']}% podium | {driver['dnf_pct']}% DNF</div>{actual}</div>"
    else:
        return f"<div style='background:#111128;border:1px solid {c}44;border-radius:12px;padding:18px;text-align:center;margin-top:40px;'><div style='font-size:24px;'>{medal}</div><div style='font-size:16px;font-weight:900;color:white;font-family:monospace;margin-top:4px;'>{driver['driver']}</div><div style='font-size:12px;color:{c};margin-top:2px;'>{driver['team']}</div><div style='font-size:28px;font-weight:900;color:{c};font-family:monospace;margin-top:8px;'>{driver['win_pct']}%</div><div style='font-size:10px;color:#888;'>P{driver['grid_pos']} grid | {driver['podium_pct']}% podium</div>{actual}</div>"

# Section divider with Monte Carlo label
st.markdown("""
<div style="margin:8px 0 8px 0;padding:10px 16px;background:rgba(0,210,190,0.05);
            border-left:3px solid #00D2BE;border-radius:4px;">
    <span style="font-size:11px;letter-spacing:3px;color:#00D2BE;font-weight:900;">MONTE CARLO PODIUM</span>
    <span style="font-size:11px;color:#666;margin-left:10px;">100K simulation prediction</span>
</div>
""", unsafe_allow_html=True)

col_l, col_c, col_r = st.columns([2, 3, 2])
with col_l:
    st.markdown(card_html(p2, "🥈", "small", result), unsafe_allow_html=True)
with col_c:
    st.markdown(card_html(p1, "🥇", "big", result), unsafe_allow_html=True)
with col_r:
    st.markdown(card_html(p3, "🥉", "small", result), unsafe_allow_html=True)

st.markdown("")

# ============================================================
# SECTION 1.6: XGBOOST PODIUM (mirror of MC podium, second row)
# ============================================================
if xgb and xgb.get("available"):
    xgb_top3 = xgb["predictions"][:3]
    xp1, xp2, xp3 = xgb_top3[0], xgb_top3[1], xgb_top3[2]

    def xgb_card_html(driver, medal, size, result_data):
        """Same style as MC podium cards but using XGBoost predicted positions."""
        c = TEAM_COLORS.get(driver["team"], "#FFD700")
        actual = ""
        if result_data:
            ap = next((r for r in result_data["result"] if r["driver"] == driver["driver"]), None)
            if ap and ap.get("pos"):
                actual = f"<div style='font-size:11px;color:#FFD700;margin-top:6px;'>Actual: P{ap['pos']}</div>"
            elif ap:
                actual = f"<div style='font-size:11px;color:#DC0000;margin-top:6px;'>{ap.get('status','DNF')}</div>"
        win_pct = round(driver["win_prob"] * 100, 2)
        pos = driver["predicted_position"]
        if size == "big":
            return f"<div style='background:#111128;border:2px solid {c}66;border-radius:12px;padding:24px;text-align:center;'><div style='font-size:36px;'>{medal}</div><div style='font-size:22px;font-weight:900;color:white;font-family:monospace;margin-top:4px;'>{driver['driver']}</div><div style='font-size:13px;color:{c};margin-top:2px;'>{driver['team']}</div><div style='font-size:42px;font-weight:900;color:{c};font-family:monospace;margin-top:10px;'>{win_pct}%</div><div style='font-size:11px;color:#888;'>P{driver['grid_pos']} grid | predicted finish P{pos}</div>{actual}</div>"
        else:
            return f"<div style='background:#111128;border:1px solid {c}44;border-radius:12px;padding:18px;text-align:center;margin-top:40px;'><div style='font-size:24px;'>{medal}</div><div style='font-size:16px;font-weight:900;color:white;font-family:monospace;margin-top:4px;'>{driver['driver']}</div><div style='font-size:12px;color:{c};margin-top:2px;'>{driver['team']}</div><div style='font-size:28px;font-weight:900;color:{c};font-family:monospace;margin-top:8px;'>{win_pct}%</div><div style='font-size:10px;color:#888;'>P{driver['grid_pos']} grid | finish P{pos}</div>{actual}</div>"

    # Section divider with XGBoost label
    st.markdown("""
    <div style="margin:24px 0 8px 0;padding:10px 16px;background:rgba(255,215,0,0.05);
                border-left:3px solid #FFD700;border-radius:4px;">
        <span style="font-size:11px;letter-spacing:3px;color:#FFD700;font-weight:900;">XGBOOST PODIUM</span>
        <span style="font-size:11px;color:#666;margin-left:10px;">data-driven model prediction</span>
    </div>
    """, unsafe_allow_html=True)

    col_l2, col_c2, col_r2 = st.columns([2, 3, 2])
    with col_l2:
        st.markdown(xgb_card_html(xp2, "🥈", "small", result), unsafe_allow_html=True)
    with col_c2:
        st.markdown(xgb_card_html(xp1, "🥇", "big", result), unsafe_allow_html=True)
    with col_r2:
        st.markdown(xgb_card_html(xp3, "🥉", "small", result), unsafe_allow_html=True)

st.markdown("")

# ============================================================
# SECTION 1.7: XGBOOST COMPARISON (added R4 Miami)
# ============================================================
if xgb and xgb.get("available"):
    st.markdown("---")
    st.markdown("### 🤖 XGBoost vs Monte Carlo")

    # Agreement banner
    if models_agree:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(0,255,136,0.12),rgba(0,255,136,0.02));
                    border:1px solid rgba(0,255,136,0.4);border-radius:10px;padding:14px 20px;margin-bottom:14px;">
            <span style="color:#00ff88;font-weight:900;letter-spacing:2px;">✓ MODELS AGREE</span>
            <span style="color:#cfcfcf;margin-left:12px;">Both models picked the same winner</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(255,136,0,0.12),rgba(255,136,0,0.02));
                    border:1px solid rgba(255,136,0,0.4);border-radius:10px;padding:14px 20px;margin-bottom:14px;">
            <span style="color:#ff8800;font-weight:900;letter-spacing:2px;">⚠ MODELS DISAGREE</span>
            <span style="color:#cfcfcf;margin-left:12px;">The two models picked different winners</span>
        </div>""", unsafe_allow_html=True)

    # Side by side cards
    mc_top = predictions[0]
    xgb_top = xgb["predictions"][0]
    mc_color = TEAM_COLORS.get(mc_top["team"], "#00D2BE")
    xgb_color = TEAM_COLORS.get(xgb_top["team"], "#FFD700")

    col_mc, col_xgb = st.columns(2)

    with col_mc:
        st.markdown(f"""
        <div style="background:#111128;border:1px solid rgba(0,210,190,0.4);border-radius:10px;padding:18px;height:170px;">
            <div style="font-size:10px;letter-spacing:2px;color:#00D2BE;">MONTE CARLO</div>
            <div style="font-size:11px;color:#666;">Simulation-based | 100K runs</div>
            <div style="font-size:20px;font-weight:900;color:white;font-family:monospace;margin-top:10px;">{mc_top['driver']}</div>
            <div style="font-size:12px;color:{mc_color};">{mc_top['team']}</div>
            <div style="font-size:28px;font-weight:900;color:#00D2BE;font-family:monospace;margin-top:8px;">{mc_top['win_pct']}%</div>
            <div style="font-size:10px;color:#888;">win probability | P{mc_top['grid_pos']} grid</div>
        </div>""", unsafe_allow_html=True)

    with col_xgb:
        xgb_pct = round(xgb_top["win_prob"] * 100, 2)
        st.markdown(f"""
        <div style="background:#111128;border:1px solid rgba(255,215,0,0.4);border-radius:10px;padding:18px;height:170px;">
            <div style="font-size:10px;letter-spacing:2px;color:#FFD700;">XGBOOST</div>
            <div style="font-size:11px;color:#666;">Data-driven | trained on {xgb['trained_rows']} rows | MAE {xgb['mae']}</div>
            <div style="font-size:20px;font-weight:900;color:white;font-family:monospace;margin-top:10px;">{xgb_top['driver']}</div>
            <div style="font-size:12px;color:{TEAM_COLORS.get(xgb_top['team'],'#888')};">{xgb_top['team']}</div>
            <div style="font-size:28px;font-weight:900;color:#FFD700;font-family:monospace;margin-top:8px;">P{xgb_top['predicted_position']}</div>
            <div style="font-size:10px;color:#888;">predicted finish | {xgb_pct}% win prob</div>
        </div>""", unsafe_allow_html=True)

    # Top 3 from each model side by side
    st.markdown("")
    st.markdown("**Predicted Podium Comparison**")
    col_mc2, col_xgb2 = st.columns(2)

    with col_mc2:
        st.markdown("<div style='font-size:11px;color:#00D2BE;letter-spacing:2px;'>MONTE CARLO TOP 3</div>", unsafe_allow_html=True)
        for i, p in enumerate(predictions[:3], 1):
            tc = TEAM_COLORS.get(p["team"], "#888")
            st.markdown(f"<div style='background:#111128;border-left:3px solid {tc};padding:8px 12px;margin:4px 0;font-family:monospace;'>P{i} <strong>{p['driver']}</strong> <span style='color:#888;'>{p['team']}</span> <span style='float:right;color:#00D2BE;'>{p['win_pct']}%</span></div>", unsafe_allow_html=True)

    with col_xgb2:
        st.markdown("<div style='font-size:11px;color:#FFD700;letter-spacing:2px;'>XGBOOST TOP 3</div>", unsafe_allow_html=True)
        for i, p in enumerate(xgb["predictions"][:3], 1):
            tc = TEAM_COLORS.get(p["team"], "#888")
            st.markdown(f"<div style='background:#111128;border-left:3px solid {tc};padding:8px 12px;margin:4px 0;font-family:monospace;'>P{i} <strong>{p['driver']}</strong> <span style='color:#888;'>{p['team']}</span> <span style='float:right;color:#FFD700;'>pos {p['predicted_position']}</span></div>", unsafe_allow_html=True)

    # Feature importance chart
    st.markdown("")
    st.markdown("**XGBoost Feature Importance** (which features drive the model's predictions)")
    importance = xgb.get("feature_importance", {})
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            y=[k.replace("_", " ") for k, _ in sorted_imp],
            x=[round(v * 100, 2) for _, v in sorted_imp],
            orientation="h", marker_color="#FFD700", marker_opacity=0.7,
            text=[f"{round(v * 100, 1)}%" for _, v in sorted_imp],
            textposition="outside", textfont=dict(color="#e0e0e0", size=10),
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111128",
            font=dict(family="monospace", color="#e0e0e0", size=10),
            xaxis=dict(title="Importance %", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=120, t=20, b=40), height=400,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

elif xgb and not xgb.get("available"):
    st.info(f"🤖 XGBoost: {xgb.get('reason', 'not enough training data yet')}. Will activate once enough race results are submitted.")
elif xgb is None:
    st.info("🤖 XGBoost not installed. Run `pip install xgboost` to enable the second model.")

# ============================================================
# SECTION 2: WIN PROBABILITY CHART
# ============================================================
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
    title="Win Probability (Top 10) - Monte Carlo",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111128",
    font=dict(family="monospace", color="#e0e0e0"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Win %"),
    margin=dict(t=40, b=40), height=350,
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# SECTION 3: PODIUM + DNF SIDE BY SIDE
# ============================================================
c1, c2 = st.columns(2)
with c1:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=[d["driver"].split()[-1] for d in top10],
        y=[d["podium_pct"] for d in top10],
        marker_color=[TEAM_COLORS.get(d["team"], "#666") for d in top10],
        marker_opacity=0.6,
    ))
    fig2.update_layout(title="Podium %", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111128",
        font=dict(family="monospace", color="#e0e0e0", size=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=40, b=40), height=280)
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=[d["driver"].split()[-1] for d in top10],
        y=[d["dnf_pct"] for d in top10],
        marker_color=["#DC0000" if d["dnf_pct"]>15 else "#FF8700" if d["dnf_pct"]>10 else "#555" for d in top10],
    ))
    fig3.update_layout(title="DNF Risk %", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111128",
        font=dict(family="monospace", color="#e0e0e0", size=10),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=40, b=40), height=280)
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# SECTION 4: FULL GRID TABLE
# ============================================================
st.markdown("### Full Grid")
table = []
xgb_pos_lookup = {}
if xgb and xgb.get("available"):
    xgb_pos_lookup = {p["driver"]: p["predicted_position"] for p in xgb["predictions"]}

for i, p in enumerate(predictions):
    row = {"#": i+1, "Driver": p["driver"], "Team": p["team"],
           "Grid": f"P{p['grid_pos']}", "MC Win%": p["win_pct"],
           "MC Podium%": p["podium_pct"], "MC DNF%": p["dnf_pct"]}
    if xgb_pos_lookup:
        row["XGB Pos"] = xgb_pos_lookup.get(p["driver"], "?")
    if result:
        actual = next((r for r in result["result"] if r["driver"] == p["driver"]), None)
        if actual and actual.get("pos"):
            row["Actual"] = f"P{actual['pos']}"
        elif actual:
            row["Actual"] = str(actual.get("status", "DNF"))[:12]
        else:
            row["Actual"] = "?"
    table.append(row)
st.dataframe(table, use_container_width=True, hide_index=True)

# ============================================================
# SECTION 5: SEASON ACCURACY
# ============================================================
if history:
    st.markdown("---")
    st.markdown("### Season Accuracy")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Races", len(history))
    c2.metric("MC Winners Correct", f"{correct}/{len(history)}")
    pole_hits, pole_races = baseline_record(folders)
    if pole_races:
        c3.metric("Always-Pole Baseline", f"{pole_hits}/{pole_races}")
    else:
        c3.metric("Always-Pole Baseline", "—")
    avg_pod = np.mean([h.get("podium_overlap", 0) for h in history])
    c4.metric("MC Avg Podium Overlap", f"{avg_pod:.1f}/3")
    xgb_correct = sum(1 for h in history if h.get("xgb_winner_correct"))
    xgb_total = sum(1 for h in history if h.get("xgb_winner_correct") is not None)
    if xgb_total:
        c5.metric("XGB Winners Correct", f"{xgb_correct}/{xgb_total}")
    else:
        c5.metric("XGB Winners Correct", "—")

    if pole_races:
        mc_rate = 100 * correct / len(history)
        pole_rate = 100 * pole_hits / pole_races
        edge = mc_rate - pole_rate
        verdict = "ahead of" if edge > 0 else "behind" if edge < 0 else "level with"
        st.caption(
            f"The model is {verdict} the naive strategy of always picking the pole "
            f"sitter, by {abs(edge):.1f} points. At {pole_races} races that gap is well "
            f"inside noise, so neither number supports a claim yet. Beating this "
            f"baseline over a full season is the bar the model has to clear."
        )

    for h in history:
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.markdown(f"**R{h['round']} {h['race']}** — Predicted: {h['predicted_winner']} ({h['predicted_win_pct']}%) — Actual: {h['actual_winner']}")
        with col2:
            st.markdown("MC: " + ("✅" if h.get("correct") else "❌"))
        with col3:
            xgb_c = h.get("xgb_winner_correct")
            if xgb_c is True:
                st.markdown("XGB: ✅")
            elif xgb_c is False:
                st.markdown("XGB: ❌")
            else:
                st.markdown("XGB: —")

# ============================================================
# SECTION 6: MODEL WEIGHTS
# ============================================================
st.markdown("---")
st.markdown("### Monte Carlo Feature Weights (self-calibrating)")
st.caption(f"Calibrated after Round {config.get('last_calibrated_after_round', 0)}. Weights adjust after each race result.")

weights = config["weights"]
sorted_w = sorted(weights.items(), key=lambda x: x[1], reverse=True)
fig4 = go.Figure()
fig4.add_trace(go.Bar(
    y=[w[0].replace("_", " ") for w in sorted_w],
    x=[round(w[1]*100, 1) for w in sorted_w],
    orientation="h", marker_color="#00D2BE", marker_opacity=0.7,
    text=[f"{round(w[1]*100,1)}%" for w in sorted_w],
    textposition="outside", textfont=dict(color="#e0e0e0", size=11),
))
fig4.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111128",
    font=dict(family="monospace", color="#e0e0e0"),
    xaxis=dict(title="Weight %", gridcolor="rgba(255,255,255,0.05)"),
    margin=dict(l=120, t=20, b=40), height=350,
)
st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# SECTION 7: ACTIONS (run prediction, submit result)
# ============================================================
st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Run New Prediction")
    available = [f for f in folders if has_data(f)]
    if available:
        run_race = st.selectbox("Race", available, key="run",
                                 format_func=lambda x: x.replace("_", " ").title())
        if st.button("🏎️ Run 100K Simulations + XGBoost", type="primary"):
            with st.spinner("Simulating..."):
                predict(run_race, config)
            st.balloons()
            st.rerun()

with c2:
    st.markdown("### Submit Race Result")
    to_validate = [f for f in folders if load_prediction(f)]
    if to_validate:
        val_race = st.selectbox("Race", to_validate, key="val",
                                 format_func=lambda x: x.replace("_", " ").title())
        race_round = int(val_race.split("_")[0])
        pred_data = load_prediction(val_race)
        drivers = [p["driver"] for p in pred_data["predictions"]]

        winner_input = st.selectbox("Who won?", [""] + drivers, key="winner")
        p2 = st.selectbox("P2?", [""] + drivers, key="p2")
        p3 = st.selectbox("P3?", [""] + drivers, key="p3")

        if st.button("Submit and Calibrate", type="primary"):
            if not winner_input:
                st.error("Select the winner.")
            else:
                top3 = []
                if winner_input: top3.append({"pos": 1, "driver": winner_input, "team": "", "status": "Finished"})
                if p2: top3.append({"pos": 2, "driver": p2, "team": "", "status": "Finished"})
                if p3: top3.append({"pos": 3, "driver": p3, "team": "", "status": "Finished"})

                pos = len(top3) + 1
                for d in drivers:
                    if d not in [t["driver"] for t in top3]:
                        top3.append({"pos": pos, "driver": d, "team": "", "status": "Finished"})
                        pos += 1

                result_path = os.path.join("races", val_race, "result.json")
                with open(result_path, "w") as f:
                    json.dump({"result": top3}, f, indent=2)

                with st.spinner("Calibrating..."):
                    calibrate(race_round)
                st.success("Weights calibrated.")
                st.rerun()

# Footer
st.markdown("---")
st.caption("F1 2026 Race Predictor | Monte Carlo + XGBoost | Self-calibrating | Built with Streamlit + Plotly + NumPy + scikit-learn")