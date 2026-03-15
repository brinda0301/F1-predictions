import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, Legend, Cell, PieChart, Pie } from "recharts";

const API = "http://localhost:8000";
const TEAM_COLORS = {
  "Mercedes": "#00D2BE", "Ferrari": "#DC0000", "McLaren": "#FF8700",
  "Red Bull": "#3671C6", "Racing Bulls": "#6692FF", "Audi": "#FF0000",
  "Haas": "#B6BABD", "Alpine": "#0090FF", "Williams": "#005AFF",
  "Aston Martin": "#006F62", "Cadillac": "#1E1E1E",
};

function useFetch(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  useEffect(() => {
    setLoading(true);
    fetch(API + url)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [url]);
  return { data, loading, error };
}

function Card({ children, style }) {
  return (
    <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "16px 20px", ...style }}>
      {children}
    </div>
  );
}

function Label({ children, color }) {
  return <div style={{ fontSize: 10, letterSpacing: 2, color: color || "#666", fontWeight: 700, marginBottom: 10 }}>{children}</div>;
}

export default function Dashboard() {
  const [view, setView] = useState("predict");
  const [selectedRound, setSelectedRound] = useState(2);

  const { data: home } = useFetch("/");
  const { data: races } = useFetch("/races");
  const { data: raceData, loading: raceLoading } = useFetch(`/races/${selectedRound}`);
  const { data: accuracy } = useFetch("/accuracy");
  const { data: weights } = useFetch("/weights");

  const predictions = raceData?.prediction?.predictions || [];
  const result = raceData?.result?.result || [];
  const topDrivers = predictions.slice(0, 10);

  const barData = topDrivers.map(d => ({
    name: d.driver.split(" ").pop(),
    driver: d.driver,
    win: d.win_pct,
    podium: d.podium_pct,
    dnf: d.dnf_pct,
    color: TEAM_COLORS[d.team] || "#666",
  }));

  const weightData = weights ? Object.entries(weights.weights)
    .map(([k, v]) => ({ name: k.replace(/_/g, " "), weight: Math.round(v * 100) }))
    .sort((a, b) => b.weight - a.weight) : [];

  const accuracyData = accuracy?.history?.map(h => ({
    race: h.race.replace(" Grand Prix", ""),
    correct: h.correct ? 1 : 0,
    podium_overlap: h.podium_overlap,
    error: h.mean_position_error || 0,
    winner: h.actual_winner?.split(" ").pop(),
    predicted: h.predicted_winner?.split(" ").pop(),
    win_pct: h.predicted_win_pct,
  })) || [];

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(135deg, #0a0a1a 0%, #0f0f25 50%, #0d0d20 100%)", color: "#e0e0e0", fontFamily: "'JetBrains Mono', 'SF Mono', monospace" }}>

      {/* HEADER */}
      <div style={{ background: "linear-gradient(90deg, rgba(220,0,0,0.08) 0%, rgba(0,210,190,0.08) 50%, rgba(54,113,198,0.08) 100%)", borderBottom: "1px solid rgba(255,255,255,0.06)", padding: "16px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", maxWidth: 1200, margin: "0 auto" }}>
          <div>
            <div style={{ fontSize: 9, letterSpacing: 3, color: "#555" }}>FASTAPI + REACT + MONTE CARLO</div>
            <div style={{ fontSize: 24, fontWeight: 900, background: "linear-gradient(90deg, #DC0000, #FF8700, #00D2BE)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              F1 2026 RACE PREDICTOR
            </div>
          </div>
          <div style={{ textAlign: "right", fontSize: 11 }}>
            <div style={{ color: "#00D2BE", fontWeight: 700 }}>{home?.model_version || "loading..."}</div>
            <div style={{ color: "#666" }}>Winner accuracy: {home?.winner_accuracy || "..."}</div>
            <div style={{ color: "#555", fontSize: 9 }}>API: {API}</div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 24px" }}>

        {/* NAV */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
          {[
            { id: "predict", label: "PREDICTIONS" },
            { id: "accuracy", label: "ACCURACY" },
            { id: "weights", label: "WEIGHTS" },
            { id: "calendar", label: "CALENDAR" },
          ].map(tab => (
            <button key={tab.id} onClick={() => setView(tab.id)}
              style={{ padding: "8px 18px", fontSize: 11, letterSpacing: 2, fontWeight: 600,
                background: view === tab.id ? "rgba(0,210,190,0.15)" : "rgba(255,255,255,0.03)",
                border: view === tab.id ? "1px solid rgba(0,210,190,0.4)" : "1px solid rgba(255,255,255,0.08)",
                color: view === tab.id ? "#00D2BE" : "#666", borderRadius: 6, cursor: "pointer", fontFamily: "inherit" }}>
              {tab.label}
            </button>
          ))}
        </div>

        {/* PREDICTIONS VIEW */}
        {view === "predict" && (
          <>
            {/* Race selector */}
            <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
              {races?.races?.filter(r => r.has_prediction).map(r => (
                <button key={r.round} onClick={() => setSelectedRound(r.round)}
                  style={{ padding: "6px 14px", fontSize: 11, fontWeight: 600,
                    background: selectedRound === r.round ? "rgba(0,210,190,0.15)" : "rgba(255,255,255,0.03)",
                    border: selectedRound === r.round ? "1px solid rgba(0,210,190,0.4)" : "1px solid rgba(255,255,255,0.06)",
                    color: selectedRound === r.round ? "#00D2BE" : "#888",
                    borderRadius: 6, cursor: "pointer", fontFamily: "inherit" }}>
                  R{r.round} {r.name.replace(" Grand Prix", "")}
                </button>
              ))}
            </div>

            {raceLoading ? (
              <Card><div style={{ color: "#666", padding: 40, textAlign: "center" }}>Loading race data...</div></Card>
            ) : predictions.length === 0 ? (
              <Card><div style={{ color: "#666", padding: 40, textAlign: "center" }}>No prediction data for this race</div></Card>
            ) : (
              <>
                {/* Winner hero */}
                <Card style={{ background: "linear-gradient(135deg, rgba(0,210,190,0.08) 0%, rgba(0,210,190,0.02) 100%)", border: "1px solid rgba(0,210,190,0.2)", marginBottom: 16, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ fontSize: 9, letterSpacing: 3, color: "#00D2BE", marginBottom: 4 }}>
                      PREDICTED WINNER {result.length > 0 && predictions[0]?.driver === result.find(r => r.pos === 1)?.driver ? "  ✓  CORRECT" : result.length > 0 ? "  ✗" : ""}
                    </div>
                    <div style={{ fontSize: 28, fontWeight: 900, color: "#fff" }}>{predictions[0]?.driver}</div>
                    <div style={{ fontSize: 12, color: TEAM_COLORS[predictions[0]?.team] || "#888" }}>
                      {predictions[0]?.team} | P{predictions[0]?.grid_pos} on Grid | DNF Risk: {predictions[0]?.dnf_pct}%
                    </div>
                    {result.length > 0 && (
                      <div style={{ fontSize: 11, color: "#FFD700", marginTop: 6 }}>
                        Actual winner: {result.find(r => r.pos === 1)?.driver || "Unknown"}
                      </div>
                    )}
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 44, fontWeight: 900, color: "#00D2BE", lineHeight: 1 }}>{predictions[0]?.win_pct}%</div>
                    <div style={{ fontSize: 10, color: "#888", letterSpacing: 1 }}>WIN PROBABILITY</div>
                  </div>
                </Card>

                {/* Bar chart */}
                <Card style={{ marginBottom: 16 }}>
                  <Label>TOP 10 WIN PROBABILITY</Label>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={barData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                      <XAxis dataKey="name" tick={{ fill: "#888", fontSize: 10, fontFamily: "inherit" }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
                      <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8, color: "#e0e0e0", fontSize: 12 }} />
                      <Bar dataKey="win" name="Win%" radius={[4, 4, 0, 0]} maxBarSize={44}>
                        {barData.map((e, i) => <Cell key={i} fill={e.color} fillOpacity={0.8} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Card>

                {/* Full table */}
                <Card>
                  <Label>FULL GRID ({predictions.length} DRIVERS)</Label>
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                      <thead>
                        <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                          {["#", "Driver", "Team", "Grid", "Win%", "Podium%", "DNF%"].map(h => (
                            <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: "#555", fontWeight: 600, fontSize: 10, letterSpacing: 1 }}>{h}</th>
                          ))}
                          {result.length > 0 && <th style={{ textAlign: "left", padding: "8px 10px", color: "#FFD700", fontWeight: 600, fontSize: 10 }}>ACTUAL</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {predictions.map((p, i) => {
                          const actualPos = result.find(r => r.driver === p.driver)?.pos;
                          const actualStatus = result.find(r => r.driver === p.driver)?.status;
                          return (
                            <tr key={p.driver} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                              <td style={{ padding: "8px 10px", fontWeight: 700, color: i < 3 ? "#00D2BE" : "#555" }}>{i + 1}</td>
                              <td style={{ padding: "8px 10px", fontWeight: 600, color: "#e0e0e0" }}>
                                <span style={{ display: "inline-block", width: 3, height: 14, borderRadius: 2, background: TEAM_COLORS[p.team] || "#666", marginRight: 8, verticalAlign: "middle" }} />
                                {p.driver}
                              </td>
                              <td style={{ padding: "8px 10px", color: TEAM_COLORS[p.team] || "#888", fontSize: 11 }}>{p.team}</td>
                              <td style={{ padding: "8px 10px", color: "#888" }}>P{p.grid_pos}</td>
                              <td style={{ padding: "8px 10px", fontWeight: 700, color: p.win_pct > 5 ? "#00D2BE" : p.win_pct > 1 ? "#FFD700" : "#555" }}>{p.win_pct}%</td>
                              <td style={{ padding: "8px 10px", color: "#aaa" }}>{p.podium_pct}%</td>
                              <td style={{ padding: "8px 10px", color: p.dnf_pct > 20 ? "#DC0000" : p.dnf_pct > 15 ? "#FF8700" : "#555" }}>{p.dnf_pct}%</td>
                              {result.length > 0 && (
                                <td style={{ padding: "8px 10px", fontWeight: 700, color: actualPos === 1 ? "#00D2BE" : actualPos && actualPos <= 3 ? "#FFD700" : actualPos ? "#aaa" : "#DC0000" }}>
                                  {actualPos ? `P${actualPos}` : actualStatus?.includes("DN") ? "DNF" : "NC"}
                                </td>
                              )}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </>
            )}
          </>
        )}

        {/* ACCURACY VIEW */}
        {view === "accuracy" && (
          <>
            {/* Summary cards */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12, marginBottom: 16 }}>
              {[
                { label: "RACES", value: accuracy?.races_completed || 0, color: "#e0e0e0" },
                { label: "WINNER ACCURACY", value: accuracy?.winner_accuracy || "0/0", color: "#00D2BE" },
                { label: "AVG PODIUM OVERLAP", value: accuracy?.avg_podium_overlap ? `${accuracy.avg_podium_overlap}/3` : "N/A", color: "#FFD700" },
                { label: "AVG POSITION ERROR", value: accuracy?.avg_position_error || "N/A", color: "#FF8700" },
              ].map((card, i) => (
                <Card key={i}>
                  <div style={{ fontSize: 9, letterSpacing: 2, color: "#666" }}>{card.label}</div>
                  <div style={{ fontSize: 28, fontWeight: 900, color: card.color, marginTop: 4 }}>{card.value}</div>
                </Card>
              ))}
            </div>

            {/* Race-by-race accuracy */}
            <Card>
              <Label>RACE BY RACE RESULTS</Label>
              {accuracyData.length === 0 ? (
                <div style={{ color: "#555", padding: 20 }}>No accuracy data yet. Submit race results to start tracking.</div>
              ) : (
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                      {["Race", "Predicted", "Win%", "Actual", "Correct", "Podium"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "10px", color: "#555", fontWeight: 600, fontSize: 10, letterSpacing: 1 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {accuracyData.map((r, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                        <td style={{ padding: "10px", color: "#e0e0e0", fontWeight: 600 }}>{r.race}</td>
                        <td style={{ padding: "10px", color: "#aaa" }}>{r.predicted}</td>
                        <td style={{ padding: "10px", color: "#00D2BE" }}>{r.win_pct}%</td>
                        <td style={{ padding: "10px", color: "#FFD700", fontWeight: 600 }}>{r.winner}</td>
                        <td style={{ padding: "10px" }}>
                          <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 10, fontWeight: 700,
                            background: r.correct ? "rgba(0,210,190,0.15)" : "rgba(220,0,0,0.15)",
                            color: r.correct ? "#00D2BE" : "#DC0000",
                            border: `1px solid ${r.correct ? "rgba(0,210,190,0.3)" : "rgba(220,0,0,0.3)"}` }}>
                            {r.correct ? "YES" : "NO"}
                          </span>
                        </td>
                        <td style={{ padding: "10px", color: "#aaa" }}>{r.podium_overlap}/3</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </>
        )}

        {/* WEIGHTS VIEW */}
        {view === "weights" && (
          <>
            <Card style={{ marginBottom: 16 }}>
              <Label>CURRENT MODEL WEIGHTS (SELF-CALIBRATING)</Label>
              <div style={{ fontSize: 11, color: "#888", marginBottom: 16 }}>
                Last calibrated after Round {weights?.last_calibrated || 0}. Weights adjust automatically after each race result.
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={weightData} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 120 }}>
                  <XAxis type="number" tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${v}%`} />
                  <YAxis type="category" dataKey="name" tick={{ fill: "#aaa", fontSize: 11, fontFamily: "inherit" }} axisLine={false} tickLine={false} width={120} />
                  <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8, color: "#e0e0e0", fontSize: 12 }} />
                  <Bar dataKey="weight" fill="#00D2BE" fillOpacity={0.7} radius={[0, 4, 4, 0]} maxBarSize={22} />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card>
              <Label>2026 REGULATION PARAMETERS</Label>
              {weights?.regulation_params && Object.entries(weights.regulation_params).map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid rgba(255,255,255,0.03)", fontSize: 12 }}>
                  <span style={{ color: "#888" }}>{k.replace(/_/g, " ")}</span>
                  <span style={{ color: "#00D2BE", fontWeight: 600 }}>{v}</span>
                </div>
              ))}
            </Card>
          </>
        )}

        {/* CALENDAR VIEW */}
        {view === "calendar" && (
          <Card>
            <Label>2026 RACE CALENDAR</Label>
            {races?.races?.map((r, i) => (
              <div key={i} onClick={() => { setSelectedRound(r.round); setView("predict"); }}
                style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 8px",
                  borderBottom: "1px solid rgba(255,255,255,0.03)", cursor: r.has_prediction ? "pointer" : "default" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <span style={{ fontSize: 11, color: "#555", fontWeight: 700, width: 30 }}>R{r.round}</span>
                  <div>
                    <div style={{ fontSize: 12, color: "#e0e0e0", fontWeight: 600 }}>{r.name}</div>
                    <div style={{ fontSize: 10, color: "#666" }}>{r.circuit} | {r.date} | {r.format}</div>
                  </div>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  {r.has_prediction && (
                    <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 9, fontWeight: 700,
                      background: "rgba(0,210,190,0.1)", color: "#00D2BE", border: "1px solid rgba(0,210,190,0.2)" }}>
                      PREDICTED
                    </span>
                  )}
                  {r.has_result && (
                    <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 9, fontWeight: 700,
                      background: "rgba(255,215,0,0.1)", color: "#FFD700", border: "1px solid rgba(255,215,0,0.2)" }}>
                      RESULT
                    </span>
                  )}
                  {!r.has_prediction && !r.has_result && (
                    <span style={{ padding: "2px 8px", borderRadius: 4, fontSize: 9, color: "#555", border: "1px solid rgba(255,255,255,0.06)" }}>
                      UPCOMING
                    </span>
                  )}
                </div>
              </div>
            ))}
          </Card>
        )}

        {/* FOOTER */}
        <div style={{ marginTop: 20, padding: "12px 20px", background: "rgba(255,255,255,0.01)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 10, fontSize: 10, color: "#444", lineHeight: 1.6 }}>
          <span style={{ color: "#00D2BE", fontWeight: 600 }}>F1 2026 Race Predictor</span> &nbsp;
          Self-calibrating ML model. Zero betting data. 100K Monte Carlo simulations per race.
          Weights auto-adjust after each race result via gradient descent.
          Built with FastAPI + React + Recharts + NumPy.
        </div>
      </div>
    </div>
  );
}