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
    .block-container { max-width: 1200px; padding-top: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-family: monospace; }
</style>""", unsafe_allow_html=True)

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
        <div style="font-size:11px;color:#666;">100K Monte Carlo sims per race | Zero betting data | 2026 regulation-aware</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:9px;letter-spacing:2px;color:#555;">WINNER ACCURACY</div>
        <div style="font-size:32px;font-weight:900;color:#00D2BE;font-family:monospace;">{correct}/{len(history)}</div>
        <div style="font-size:9px;color:#555;">Calibrated after R{config.get('last_calibrated_after_round', 0)}</div>
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
            with st.spinner("Running 100,000 Monte Carlo simulations..."):
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

actual_winner = None
if result:
    actual_winner = next((r["driver"] for r in result["result"] if r.get("pos") == 1), None)

# ============================================================
# SECTION 1: PREDICTED WINNER
# ============================================================
status = ""
if actual_winner:
    status = " ✅ CORRECT" if actual_winner == winner["driver"] else f" ❌ Actual: {actual_winner}"

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,210,190,0.1),rgba(0,210,190,0.02));
                border:1px solid rgba(0,210,190,0.3);border-radius:12px;padding:24px;">
        <div style="font-size:10px;letter-spacing:3px;color:#00D2BE;">PREDICTED WINNER{status}</div>
        <div style="font-size:32px;font-weight:900;color:white;font-family:monospace;">{winner['driver']}</div>
        <div style="font-size:14px;color:{TEAM_COLORS.get(winner['team'],'#888')};">
            {winner['team']} | P{winner['grid_pos']} on Grid | DNF Risk: {winner['dnf_pct']}%</div>
        <div style="font-size:11px;color:#555;margin-top:4px;">
            {race_info.get('name','')} | {race_info.get('circuit','')} | {race_info.get('date','')}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="text-align:right;padding-top:20px;">
        <div style="font-size:52px;font-weight:900;color:#00D2BE;font-family:monospace;line-height:1;">{winner['win_pct']}%</div>
        <div style="font-size:10px;color:#888;letter-spacing:2px;">WIN PROBABILITY</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ============================================================
# SECTION 1.5: TOP 3 PODIUM CARDS
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

col_l, col_c, col_r = st.columns([2, 3, 2])
with col_l:
    st.markdown(card_html(p2, "🥈", "small", result), unsafe_allow_html=True)
with col_c:
    st.markdown(card_html(p1, "🥇", "big", result), unsafe_allow_html=True)
with col_r:
    st.markdown(card_html(p3, "🥉", "small", result), unsafe_allow_html=True)

st.markdown("")

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
    title="Win Probability (Top 10)",
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
for i, p in enumerate(predictions):
    row = {"#": i+1, "Driver": p["driver"], "Team": p["team"],
           "Grid": f"P{p['grid_pos']}", "Win%": p["win_pct"],
           "Podium%": p["podium_pct"], "DNF%": p["dnf_pct"]}
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

    c1, c2, c3 = st.columns(3)
    c1.metric("Races", len(history))
    c2.metric("Winners Correct", f"{correct}/{len(history)}")
    avg_pod = np.mean([h.get("podium_overlap", 0) for h in history])
    c3.metric("Avg Podium Overlap", f"{avg_pod:.1f}/3")

    for h in history:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**R{h['round']} {h['race']}** — Predicted: {h['predicted_winner']} ({h['predicted_win_pct']}%) — Actual: {h['actual_winner']}")
        with col2:
            if h.get("correct"):
                st.success("✅")
            else:
                st.error("❌")

# ============================================================
# SECTION 6: MODEL WEIGHTS
# ============================================================
st.markdown("---")
st.markdown("### Model Weights (self-calibrating)")
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
        if st.button("🏎️ Run 100K Simulations", type="primary"):
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

                # Add remaining drivers with estimated positions
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
st.caption("F1 2026 Race Predictor | Self-calibrating ML model | Zero betting data | Built with Streamlit + Plotly + NumPy")