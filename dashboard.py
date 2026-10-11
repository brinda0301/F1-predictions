"""
dashboard.py

One dashboard, two entry points, so the local and public views can never
drift apart:

    streamlit run app_public.py   read-only, deployed to Streamlit Cloud
    streamlit run app.py          same view plus a Run Prediction button

Reads prediction.json, result.json and config.json only. The engine is
imported only when the local Run Prediction button is pressed, so the cloud
build never needs FastF1.
"""

import glob
import json
import os

import plotly.graph_objects as go
import streamlit as st

import probscore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RACES_DIR = os.path.join(BASE_DIR, "races")

TEAM_COLORS = {
    "Mercedes": "#00D2BE", "Ferrari": "#DC0000", "McLaren": "#FF8700",
    "Red Bull": "#3671C6", "Racing Bulls": "#6692FF", "Audi": "#FF0000",
    "Haas": "#B6BABD", "Alpine": "#0090FF", "Williams": "#005AFF",
    "Aston Martin": "#006F62", "Cadillac": "#1E1E1E",
}
MC_COLOR, XGB_COLOR = "#00D2BE", "#FFD700"

# Hand-written circuit names for the early rounds. From R15 on, the banner
# reads the race's own RACE_INFO, so this is not extended.
RACE_META = {
    "01_australia": ("Australian Grand Prix", "Albert Park, Melbourne"),
    "02_china": ("Chinese Grand Prix", "Shanghai International Circuit"),
    "03_japan": ("Japanese Grand Prix", "Suzuka Circuit"),
    "04_miami": ("Miami Grand Prix", "Miami International Autodrome"),
    "05_canada": ("Canadian Grand Prix", "Circuit Gilles Villeneuve"),
    "06_monaco": ("Monaco Grand Prix", "Circuit de Monaco"),
    "07_barcelona": ("Barcelona-Catalunya Grand Prix", "Circuit de Barcelona-Catalunya"),
    "08_austria": ("Austrian Grand Prix", "Red Bull Ring"),
    "09_britain": ("British Grand Prix", "Silverstone Circuit"),
    "10_belgium": ("Belgian Grand Prix", "Spa-Francorchamps"),
    "11_hungary": ("Hungarian Grand Prix", "Hungaroring"),
    "12_netherlands": ("Dutch Grand Prix", "Circuit Zandvoort"),
    "13_italy": ("Italian Grand Prix", "Monza"),
    "14_spain": ("Spanish Grand Prix", "Madring, Madrid"),
}

CHART = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111128",
             font=dict(family="monospace", color="#e0e0e0", size=11))
GRID = "rgba(255,255,255,0.05)"


# Data ------------------------------------------------------------------------

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def race_folders():
    if not os.path.isdir(RACES_DIR):
        return []
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(RACES_DIR, "*"))
                  if os.path.isdir(p))


def load_prediction(folder):
    return load_json(os.path.join(RACES_DIR, folder, "prediction.json"))


def load_result(folder):
    return load_json(os.path.join(RACES_DIR, folder, "result.json"))


def load_config():
    return load_json(os.path.join(BASE_DIR, "config.json")) or {
        "accuracy_history": [], "last_calibrated_after_round": 0, "weights": {}}


def race_label(folder):
    if folder in RACE_META:
        return RACE_META[folder][0]
    pred = load_prediction(folder) or {}
    name = (pred.get("race") or {}).get("name")
    return name or folder.split("_", 1)[-1].replace("_", " ").title()


def banner_text(folder, race_info):
    name = race_info.get("name")
    circuit = race_info.get("circuit")
    if folder in RACE_META:
        name = name or RACE_META[folder][0]
        circuit = RACE_META[folder][1]
    parts = [name or race_label(folder)]
    if circuit:
        parts.append(circuit)
    if race_info.get("date"):
        parts.append(race_info["date"])
    return " | ".join(parts)


def pole_sitter(pred):
    for e in (pred or {}).get("predictions", []):
        if e.get("grid_pos") == 1:
            return e.get("driver")
    return None


def baseline_record(folders):
    """How often the pole sitter won, over races with a result."""
    hits = raced = 0
    for f in folders:
        res = load_result(f)
        pole = pole_sitter(load_prediction(f))
        if not res or not res.get("result") or not pole:
            continue
        raced += 1
        hits += same_driver(res["result"][0].get("driver", ""), pole)
    return hits, raced


def same_driver(a, b):
    """Early 2026 files wrote "Kimi Antonelli", results use the full name."""
    return a == b or (a.split()[-1:] == b.split()[-1:] and a.split()[-1:] != [])


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


FEATURE_LABELS = {"gap": "quali gap to fastest", "log_grid": "grid slot",
                  "field_gap": "margin to next car", "sprint_pos": "sprint finish",
                  "teammate_gap": "gap to teammate", "ot": "overtaking index"}


def surname(name):
    return name.split()[-1]


# Pieces ----------------------------------------------------------------------

def html(s):
    st.markdown(s, unsafe_allow_html=True)


def badge(correct):
    color, text = ("#00ff88", "CORRECT") if correct else ("#ff5555", "MISS")
    return (f'<div style="font-size:11px;color:{color};font-weight:900;'
            f'letter-spacing:2px;margin-top:6px;">{text}</div>')


def section(title, subtitle, color):
    html(f"""<div style="margin:24px 0 8px 0;padding:10px 16px;background:{color}0d;
         border-left:3px solid {color};border-radius:4px;">
         <span style="font-size:11px;letter-spacing:3px;color:{color};font-weight:900;">{title}</span>
         <span style="font-size:11px;color:#666;margin-left:10px;">{subtitle}</span></div>""")


def xgb_detail(xgb, row):
    """Second line under an XGBoost pick. v2 has a quali gap, v1 a finish slot."""
    if xgb.get("version") == 2:
        return f"P{row['grid_pos']} grid | {row.get('quali_gap', '')}s off pole"
    return f"P{row['grid_pos']} grid | predicted finish P{row['predicted_position']}"


def xgb_subtitle(xgb):
    rows = xgb.get("trained_rows", 0)
    if xgb.get("version") == 2:
        return f"V2 WIN CLASSIFIER | {rows} ROWS, 2022-2026 | MONOTONE"
    return f"V1 POSITION REGRESSOR | {rows} ROWS, 2026"


def actual_line(driver, result):
    if not result:
        return ""
    ap = next((r for r in result["result"] if r["driver"] == driver), None)
    if ap and ap.get("status") == "Retired":
        return "<div style='font-size:11px;color:#DC0000;margin-top:6px;'>Retired</div>"
    if ap and ap.get("pos"):
        return f"<div style='font-size:11px;color:#FFD700;margin-top:6px;'>Actual: P{ap['pos']}</div>"
    return ""


def winner_card(title, subtitle, color, row, pct, detail, badge_html):
    team_color = TEAM_COLORS.get(row["team"], color)
    html(f"""
    <div style="background:#111128;border:2px solid {team_color}88;border-radius:12px;padding:24px;text-align:center;height:340px;">
        <div style="font-size:10px;letter-spacing:3px;color:{color};font-weight:900;">{title}</div>
        <div style="font-size:9px;color:#666;margin-top:2px;">{subtitle}</div>
        <div style="font-size:46px;margin-top:8px;">🥇</div>
        <div style="font-size:22px;font-weight:900;color:white;font-family:monospace;margin-top:6px;">{row['driver']}</div>
        <div style="font-size:13px;color:{team_color};margin-top:2px;">{row['team']}</div>
        <div style="font-size:44px;font-weight:900;color:{color};font-family:monospace;margin-top:10px;line-height:1;">{pct}%</div>
        <div style="font-size:11px;color:#888;">{detail}</div>
        {badge_html}
    </div>""")


def podium_card(row, medal, big, pct, detail, result):
    c = TEAM_COLORS.get(row["team"], "#666")
    if big:
        return (f"<div style='background:#111128;border:2px solid {c}66;border-radius:12px;padding:24px;text-align:center;'>"
                f"<div style='font-size:36px;'>{medal}</div>"
                f"<div style='font-size:22px;font-weight:900;color:white;font-family:monospace;margin-top:4px;'>{row['driver']}</div>"
                f"<div style='font-size:13px;color:{c};margin-top:2px;'>{row['team']}</div>"
                f"<div style='font-size:42px;font-weight:900;color:{c};font-family:monospace;margin-top:10px;'>{pct}%</div>"
                f"<div style='font-size:11px;color:#888;'>{detail}</div>{actual_line(row['driver'], result)}</div>")
    return (f"<div style='background:#111128;border:1px solid {c}44;border-radius:12px;padding:18px;text-align:center;margin-top:40px;'>"
            f"<div style='font-size:24px;'>{medal}</div>"
            f"<div style='font-size:16px;font-weight:900;color:white;font-family:monospace;margin-top:4px;'>{row['driver']}</div>"
            f"<div style='font-size:12px;color:{c};margin-top:2px;'>{row['team']}</div>"
            f"<div style='font-size:28px;font-weight:900;color:{c};font-family:monospace;margin-top:8px;'>{pct}%</div>"
            f"<div style='font-size:10px;color:#888;'>{detail}</div>{actual_line(row['driver'], result)}</div>")


def podium_row(rows, pct_fn, detail_fn, result):
    l, c, r = st.columns([2, 3, 2])
    with l:
        html(podium_card(rows[1], "🥈", False, pct_fn(rows[1]), detail_fn(rows[1]), result))
    with c:
        html(podium_card(rows[0], "🥇", True, pct_fn(rows[0]), detail_fn(rows[0]), result))
    with r:
        html(podium_card(rows[2], "🥉", False, pct_fn(rows[2]), detail_fn(rows[2]), result))


def bar_chart(fig, title, height, ytitle=None):
    fig.update_layout(title=title, height=height, **CHART,
                      xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, title=ytitle))
    if not fig.layout.margin.l:
        fig.update_layout(margin=dict(t=50, b=60))
    st.plotly_chart(fig, use_container_width=True)


# Tabs ------------------------------------------------------------------------

def race_tab(selected):
    pred = load_prediction(selected)
    result = load_result(selected)
    preds = pred["predictions"]
    mc_top = preds[0]
    xgb = pred.get("xgboost") or {}
    xgb_ok = bool(xgb.get("available"))
    xgb_preds = xgb.get("predictions", []) if xgb_ok else []
    actual = result["result"][0]["driver"] if result and result.get("result") else None

    html(f"""<div style="background:#0a0a1a;border:1px solid rgba(255,255,255,0.08);
         border-radius:10px;padding:14px 20px;margin-bottom:16px;">
         <div style="font-size:11px;color:#555;letter-spacing:2px;">RACE</div>
         <div style="font-size:18px;font-weight:700;color:white;font-family:monospace;">
         {banner_text(selected, pred.get("race") or {})}</div></div>""")

    # Winner cards
    col_mc, col_xgb = st.columns(2)
    with col_mc:
        winner_card("MONTE CARLO WINNER", "100K SIMULATIONS", MC_COLOR, mc_top, mc_top["win_pct"],
                    f"P{mc_top['grid_pos']} grid | {mc_top.get('podium_pct', '')}% podium | {mc_top.get('dnf_pct', '')}% DNF",
                    badge(actual == mc_top["driver"]) if actual else "")
    with col_xgb:
        if xgb_ok:
            x = xgb_preds[0]
            winner_card("XGBOOST WINNER", xgb_subtitle(xgb), XGB_COLOR, x,
                        round(x["win_prob"] * 100, 2), xgb_detail(xgb, x),
                        badge(actual == x["driver"]) if actual else "")
        else:
            html("""<div style="background:#111128;border:1px dashed rgba(255,215,0,0.3);border-radius:12px;padding:24px;text-align:center;height:340px;display:flex;flex-direction:column;justify-content:center;">
                 <div style="font-size:10px;letter-spacing:3px;color:#FFD700;font-weight:900;">XGBOOST</div>
                 <div style="font-size:46px;margin-top:14px;opacity:0.3;">🥇</div>
                 <div style="font-size:14px;color:#888;margin-top:14px;">Not running yet this round</div></div>""")

    if actual:
        team = next((r.get("team") for r in result["result"] if r["driver"] == actual), "")
        html(f"""<div style="background:linear-gradient(90deg,rgba(255,215,0,0.18),rgba(255,215,0,0.04));
             border:2px solid rgba(255,215,0,0.5);border-radius:10px;padding:18px 28px;margin-top:18px;text-align:center;">
             <div style="font-size:11px;letter-spacing:4px;color:#FFD700;font-weight:900;">ACTUAL RACE WINNER</div>
             <div style="font-size:32px;font-weight:900;color:white;font-family:monospace;margin-top:4px;">{actual}</div>
             <div style="font-size:14px;color:{TEAM_COLORS.get(team, '#FFD700')};">{team}</div></div>""")

    # Agreement
    if xgb_ok:
        agree = mc_top["driver"] == xgb_preds[0]["driver"]
        color, title, text = (("#00ff88", "MODELS AGREE", "Both models picked the same winner") if agree
                              else ("#ff8800", "MODELS DISAGREE", "The two models picked different winners"))
        html(f"""<div style="border:1px solid {color}66;background:{color}14;border-radius:10px;
             padding:14px 20px;margin-top:18px;"><span style="color:{color};font-weight:900;letter-spacing:2px;">{title}</span>
             <span style="color:#cfcfcf;margin-left:12px;">{text}</span></div>""")

    # Podiums
    section("MONTE CARLO PODIUM", "100K simulation prediction", MC_COLOR)
    podium_row(preds[:3], lambda r: r["win_pct"],
               lambda r: f"P{r['grid_pos']} grid | {r.get('podium_pct', '')}% podium", result)
    if xgb_ok and len(xgb_preds) >= 3:
        section("XGBOOST PODIUM", xgb_subtitle(xgb).lower(), XGB_COLOR)
        podium_row(xgb_preds[:3], lambda r: round(r["win_prob"] * 100, 2),
                   lambda r: xgb_detail(xgb, r), result)

    # Win probability, both models
    st.markdown("")
    top = preds[:10]
    xgb_pct = {r["driver"]: round(r["win_prob"] * 100, 2) for r in xgb_preds}
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Monte Carlo", x=[surname(d["driver"]) for d in top],
                         y=[d["win_pct"] for d in top], marker_color=MC_COLOR,
                         text=[f"{d['win_pct']}%" for d in top], textposition="outside"))
    if xgb_pct:
        fig.add_trace(go.Bar(name="XGBoost", x=[surname(d["driver"]) for d in top],
                             y=[xgb_pct.get(d["driver"], 0) for d in top], marker_color=XGB_COLOR,
                             text=[f"{xgb_pct.get(d['driver'], 0)}%" for d in top], textposition="outside"))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.15, x=0))
    bar_chart(fig, "Win probability, top 10 by Monte Carlo", 380, "Win %")

    # Podium and DNF
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=[surname(d["driver"]) for d in top],
                               y=[d.get("podium_pct", 0) for d in top],
                               marker_color=[TEAM_COLORS.get(d["team"], "#666") for d in top]))
        bar_chart(fig, "Monte Carlo podium %", 280)
    with c2:
        fig = go.Figure(go.Bar(x=[surname(d["driver"]) for d in top],
                               y=[d.get("dnf_pct", 0) for d in top],
                               marker_color=["#DC0000" if d.get("dnf_pct", 0) > 15 else "#FF8700"
                                             if d.get("dnf_pct", 0) > 10 else "#555" for d in top]))
        bar_chart(fig, "Monte Carlo DNF risk %", 280)

    # Full grid
    st.markdown("### Full Grid")
    table = []
    for i, p in enumerate(preds, 1):
        row = {"#": i, "Driver": p["driver"], "Team": p["team"], "Grid": f"P{p['grid_pos']}",
               "MC Win%": p["win_pct"], "MC Podium%": p.get("podium_pct"), "MC DNF%": p.get("dnf_pct")}
        if xgb_pct:
            row["XGB Win%"] = xgb_pct.get(p["driver"])
        if result:
            ap = next((r for r in result["result"] if r["driver"] == p["driver"]), None)
            row["Actual"] = ("Retired" if ap and ap.get("status") == "Retired"
                             else f"P{ap['pos']}" if ap and ap.get("pos") else "?")
        table.append(row)
    st.dataframe(table, use_container_width=True, hide_index=True)

    # XGBoost feature importance for this race
    imp = xgb.get("feature_importance") or {}
    if imp:
        items = sorted(imp.items(), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(y=[FEATURE_LABELS.get(k, k.replace("_", " ")) for k, _ in items],
                               x=[round(v * 100, 1) for _, v in items], orientation="h",
                               marker_color=XGB_COLOR, marker_opacity=0.75,
                               text=[f"{round(v * 100, 1)}%" for _, v in items], textposition="outside"))
        fig.update_layout(margin=dict(l=170, t=40, b=40))
        bar_chart(fig, "XGBoost feature importance, this race", 90 + 28 * len(items))


def season_tab(config, folders):
    history = config.get("accuracy_history", [])
    if not history:
        st.info("No race history yet.")
        return
    n = len(history)
    mc_hits = sum(1 for h in history if h.get("correct"))
    xgb_rounds = [h for h in history if h.get("xgb_winner_correct") is not None]
    xgb_hits = sum(1 for h in xgb_rounds if h["xgb_winner_correct"])
    pole_hits, pole_races = baseline_record(folders)
    avg_pod = sum(h.get("podium_overlap", 0) for h in history) / n

    c1, c2, c3, c4 = st.columns(4)
    def rate(h, t):
        return f"{h}/{t} ({100 * h / t:.0f}%)" if t else "n/a"
    c1.metric("Monte Carlo winners", rate(mc_hits, n))
    c2.metric("XGBoost winners", rate(xgb_hits, len(xgb_rounds)))
    c3.metric("Always-pole baseline", rate(pole_hits, pole_races))
    c4.metric("MC avg podium hits", f"{avg_pod:.1f}/3")

    def mean(key):
        v = [h[key] for h in history if key in h]
        return (sum(v) / len(v), len(v)) if v else (None, 0)
    mc_ll, xgb_ll, pole_ll = mean("mc_log_loss"), mean("xgb_log_loss"), mean("pole_log_loss")
    if mc_ll[0] is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Monte Carlo log loss", f"{mc_ll[0]:.3f}")
        c2.metric("XGBoost log loss", f"{xgb_ll[0]:.3f}" if xgb_ll[0] else "n/a")
        c3.metric("Pole baseline log loss", f"{pole_ll[0]:.3f}" if pole_ll[0] else "n/a")
        c4.metric("Rounds scored", mc_ll[1])
        st.caption("Log loss is minus the log of the probability a model gave the actual winner. "
                   "Lower is better. 50% on the winner scores 0.69, 10% scores 2.30. "
                   "Monte Carlo and XGBoost were both rebuilt at R17.")

    scorecard_section(folders)

    # Running accuracy, both models and the pole baseline
    rounds, mc_run, xgb_run, pole_run = [], [], [], []
    mh = xh = xn = ph = pn = 0
    pole_by_round = {}
    for f in folders:
        res, pred = load_result(f), load_prediction(f)
        if res and res.get("result") and pole_sitter(pred):
            pole_by_round[int(f[:2])] = same_driver(res["result"][0]["driver"], pole_sitter(pred))
    for i, h in enumerate(history, 1):
        rounds.append(h["round"])
        mh += bool(h.get("correct"))
        mc_run.append(round(100 * mh / i, 1))
        if h.get("xgb_winner_correct") is not None:
            xn += 1
            xh += h["xgb_winner_correct"]
        xgb_run.append(round(100 * xh / xn, 1) if xn else None)
        if h["round"] in pole_by_round:
            pn += 1
            ph += pole_by_round[h["round"]]
        pole_run.append(round(100 * ph / pn, 1) if pn else None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rounds, y=mc_run, name="Monte Carlo", mode="lines+markers",
                             line=dict(color=MC_COLOR, width=3)))
    fig.add_trace(go.Scatter(x=rounds, y=xgb_run, name="XGBoost", mode="lines+markers",
                             line=dict(color=XGB_COLOR, width=3)))
    fig.add_trace(go.Scatter(x=rounds, y=pole_run, name="Always pole", mode="lines",
                             line=dict(color="#888", width=2, dash="dash")))
    fig.update_yaxes(range=[0, 105])
    fig.update_layout(legend=dict(orientation="h", y=-0.18, x=0))
    bar_chart(fig, "Running winner accuracy", 400, "Win rate %")

    st.markdown("### Race by race")
    rows = []
    for h in history:
        rows.append({
            "Round": h["round"], "Race": h.get("race", ""),
            "Monte Carlo": f"{h.get('predicted_winner', '')} ({h.get('predicted_win_pct', '')}%)",
            "MC": "Correct" if h.get("correct") else "Miss",
            "XGB": ("Correct" if h["xgb_winner_correct"] else "Miss")
                   if h.get("xgb_winner_correct") is not None else "n/a",
            "Actual": h.get("actual_winner", ""),
            "MC log loss": fmt(h.get("mc_log_loss")), "XGB log loss": fmt(h.get("xgb_log_loss")),
            "Pole log loss": fmt(h.get("pole_log_loss")),
            "MC podium hits": f"{h.get('podium_overlap', '')}/3",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


METRIC_ROWS = [
    ("winner_accuracy", "Winner accuracy", "pct", "Top pick won the race. Higher is better."),
    ("top3_hit", "Winner in top 3", "pct", "Actual winner was among the model's three most likely. Higher is better."),
    ("avg_p_winner", "Avg probability on winner", "pct", "How much probability the model put on the driver who won. Higher is better."),
    ("log_loss", "Log loss", "num", "Minus the log of the probability on the winner. Lower is better."),
    ("brier", "Brier score", "num", "Squared error across every driver's probability. Lower is better."),
    ("roc_auc", "ROC AUC", "num", "Chance a winner is ranked above a random non-winner. 0.5 is a coin flip, 1.0 is perfect."),
    ("avg_winner_rank", "Avg rank of winner", "num1", "Where the actual winner sat in the model's order. Lower is better."),
]
MODEL_NAMES = {"mc": "Monte Carlo", "xgb": "XGBoost", "pole": "Always pole"}
MODEL_COLORS = {"mc": MC_COLOR, "xgb": XGB_COLOR, "pole": "#888888"}
REBUILD_ROUND = 17


def scored_races(folders, first=1, last=99):
    races = []
    for f in folders:
        if not f[:2].isdigit() or not first <= int(f[:2]) <= last:
            continue
        pred, res = load_prediction(f), load_result(f)
        if pred and res and res.get("result"):
            races.append((pred, res["result"][0]["driver"]))
    return races


def scorecard_table(card):
    def show(v, kind):
        if v is None:
            return "n/a"
        return f"{100 * v:.0f}%" if kind == "pct" else f"{v:.1f}" if kind == "num1" else f"{v:.3f}"
    rows = []
    for key, label, kind, help_ in METRIC_ROWS:
        row = {"Metric": label}
        for m in ("mc", "xgb", "pole"):
            row[MODEL_NAMES[m]] = show(card[m][key], kind) if m in card else "n/a"
        row["What it means"] = help_
        rows.append(row)
    rows.append({"Metric": "Rounds scored",
                 **{MODEL_NAMES[m]: str(card[m]["rounds"]) if m in card else "0" for m in ("mc", "xgb", "pole")},
                 "What it means": "XGBoost started at R4."})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def scorecard_section(folders):
    st.markdown("### Model scorecard")
    new = scored_races(folders, first=REBUILD_ROUND)
    old = scored_races(folders, last=REBUILD_ROUND - 1)
    if new:
        st.markdown(f"**Rebuilt models, R{REBUILD_ROUND} onward**")
        scorecard_table(probscore.scorecard(new))
    else:
        st.caption(f"Both models were rebuilt before R{REBUILD_ROUND}. Their scorecard appears here "
                   f"once R{REBUILD_ROUND} has a result.")
    if old:
        st.markdown(f"**As published, R1 to R{REBUILD_ROUND - 1}** (Monte Carlo before recalibration, XGBoost v1)")
        card = probscore.scorecard(old)
        scorecard_table(card)
        st.caption("ROC AUC runs high for every model because most drivers have almost no chance "
                   "and every model ranks them low. Compare the models with each other, not with 1.0.")

    allc = probscore.scorecard(scored_races(folders))
    if allc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                                 line=dict(color="#555", dash="dot")))
        for m in ("mc", "xgb", "pole"):
            if m not in allc:
                continue
            rows = probscore.calibration(allc[m]["_scores"], allc[m]["_labels"])
            fig.add_trace(go.Scatter(
                x=[r["predicted"] for r in rows], y=[r["actual"] for r in rows],
                mode="lines+markers", name=MODEL_NAMES[m],
                line=dict(color=MODEL_COLORS[m], width=2), marker=dict(size=9),
                text=[f"{r['bin']}: {r['n']} driver-races" for r in rows],
                hovertemplate="%{text}<br>predicted %{x:.0%}, won %{y:.0%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", range=[0, 1], title="Predicted win probability")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0))
        bar_chart(fig, "Calibration, all scored rounds: when a model says X%, how often did that driver win", 420, "Actual win rate")
        st.caption("Points on the dotted line are well calibrated. Above the line, the model was too cautious. "
                   "Below, too confident. Bins with few drivers move a lot.")


def model_tab(config):
    weights = config.get("weights", {})
    if weights:
        st.markdown("### Monte Carlo feature weights")
        st.caption(f"Self-calibrating. Last calibrated after R{config.get('last_calibrated_after_round', 0)}.")
        items = sorted(weights.items(), key=lambda kv: kv[1])
        fig = go.Figure(go.Bar(y=[k.replace("_", " ") for k, _ in items],
                               x=[round(v * 100, 1) for _, v in items], orientation="h",
                               marker_color=MC_COLOR, marker_opacity=0.75,
                               text=[f"{round(v * 100, 1)}%" for _, v in items], textposition="outside"))
        fig.update_layout(margin=dict(l=120, t=20, b=40))
        bar_chart(fig, "", 520)
    temps = config.get("regulation_params", {}).get("track_type_temperatures")
    if temps:
        st.markdown("### Monte Carlo softmax temperature by track type")
        st.caption(config["regulation_params"].get("temperature_note", ""))
        st.dataframe([{"Track type": k, "Temperature": v} for k, v in temps.items()],
                     use_container_width=True, hide_index=True)
    st.markdown("### XGBoost v2")
    st.caption("Win classifier trained on 2022-2026 timing data with monotone constraints. "
               "Refit after every race. Feature importance for each race is on the Race Prediction tab.")


def run_panel(folders):
    """Local only: build a prediction from a race folder's data.py."""
    with st.sidebar:
        st.markdown("### Run prediction")
        ready = [f for f in folders if os.path.exists(os.path.join(RACES_DIR, f, "data.py"))]
        if not ready:
            st.info("No race folders with data.py yet.")
            return
        race = st.selectbox("Race", ready, index=len(ready) - 1, format_func=race_label)
        st.caption("Writes prediction.json. Build data.py first with fetch_race_data.py. "
                   "Results go through fetch_race_data.py --result --score.")
        if st.button("Run 100K simulations + XGBoost", type="primary"):
            import engine
            with st.spinner("Simulating..."):
                engine.predict(race)
            st.rerun()


# Entry -----------------------------------------------------------------------

def render(local=False):
    st.set_page_config(page_title="F1 2026 Predictor", page_icon="🏎️", layout="wide")
    html("""<style>.block-container { max-width: 1200px; padding-top: 3rem; }
         div[data-testid="stMetricValue"] { font-size: 26px; font-family: monospace; }</style>""")

    config = load_config()
    folders = race_folders()
    predicted = [f for f in folders if load_prediction(f)]

    html(f"""
    <div style="background:linear-gradient(90deg,rgba(220,0,0,0.08),rgba(0,210,190,0.08),rgba(54,113,198,0.08));
                border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:20px 28px;margin-bottom:20px;
                display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div style="font-size:9px;letter-spacing:3px;color:#555;">MONTE CARLO VS XGBOOST</div>
            <div style="font-size:28px;font-weight:900;color:white;font-family:monospace;">F1 2026 RACE PREDICTOR</div>
            <div style="font-size:11px;color:#666;">100K Monte Carlo + XGBoost v2 | Zero betting data | Predictions committed before lights out</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:9px;letter-spacing:2px;color:#555;">RACES PREDICTED</div>
            <div style="font-size:32px;font-weight:900;color:#00D2BE;font-family:monospace;">{len(predicted)}</div>
            <div style="font-size:9px;color:#555;">Calibrated after R{config.get('last_calibrated_after_round', 0)}</div>
        </div>
    </div>""")

    if local:
        run_panel(folders)

    tab_race, tab_season, tab_model = st.tabs(["Race Prediction", "Season Performance", "Models"])
    with tab_race:
        if not predicted:
            st.warning("No predictions yet.")
        else:
            selected = st.selectbox("Select Race", predicted, index=len(predicted) - 1,
                                    format_func=race_label)
            race_tab(selected)
    with tab_season:
        season_tab(config, folders)
    with tab_model:
        model_tab(config)

    html("<div style='text-align:center;color:#555;font-size:12px;margin-top:24px;'>"
         "Built by Brinda Bhanderi | Predictions committed to GitHub before each race | "
         "<a href='https://github.com/brinda0301/F1-predictions' style='color:#00D2BE;'>Repo</a></div>")
