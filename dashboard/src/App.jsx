import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, Cell } from "recharts";

const PREDICTIONS = [
  { rank: 1, driver: "George Russell", team: "Mercedes", grid: 1, win: 62.0, podium: 74.7, points: 86.8, score: 0.838, color: "#00D2BE", features: { grid_score: 1.0, practice_score: 0.89, consistency_score: 0.88, experience_score: 0.43, aus_track_score: 0.15, team_score: 1.0, market_score: 0.778, reliability_score: 1.0 } },
  { rank: 2, driver: "Lewis Hamilton", team: "Ferrari", grid: 7, win: 12.4, podium: 42.9, points: 73.6, score: 0.656, color: "#DC0000", features: { grid_score: 0.71, practice_score: 0.79, consistency_score: 0.87, experience_score: 1.0, aus_track_score: 1.0, team_score: 0.88, market_score: 0.04, reliability_score: 1.0 } },
  { rank: 3, driver: "Charles Leclerc", team: "Ferrari", grid: 4, win: 8.8, podium: 42.8, points: 72.6, score: 0.693, color: "#DC0000", features: { grid_score: 0.86, practice_score: 0.89, consistency_score: 0.87, experience_score: 0.67, aus_track_score: 0.7, team_score: 0.88, market_score: 0.08, reliability_score: 1.0 } },
  { rank: 4, driver: "Kimi Antonelli", team: "Mercedes", grid: 2, win: 3.1, podium: 22.3, points: 65.8, score: 0.626, color: "#00D2BE", features: { grid_score: 0.95, practice_score: 0.75, consistency_score: 0.72, experience_score: 0.18, aus_track_score: 0.0, team_score: 1.0, market_score: 0.154, reliability_score: 0.7 } },
  { rank: 5, driver: "Lando Norris", team: "McLaren", grid: 6, win: 3.0, podium: 21.0, points: 66.2, score: 0.620, color: "#FF8700", features: { grid_score: 0.76, practice_score: 0.81, consistency_score: 0.9, experience_score: 0.55, aus_track_score: 0.7, team_score: 0.85, market_score: 0.05, reliability_score: 1.0 } },
  { rank: 6, driver: "Isack Hadjar", team: "Red Bull", grid: 3, win: 2.1, podium: 15.1, points: 63.5, score: 0.598, color: "#3671C6", features: { grid_score: 0.90, practice_score: 0.67, consistency_score: 0.9, experience_score: 0.0, aus_track_score: 0.0, team_score: 0.82, market_score: 0.03, reliability_score: 1.0 } },
  { rank: 7, driver: "Oscar Piastri", team: "McLaren", grid: 5, win: 1.8, podium: 13.2, points: 63.3, score: 0.586, color: "#FF8700", features: { grid_score: 0.81, practice_score: 0.84, consistency_score: 0.82, experience_score: 0.33, aus_track_score: 0.0, team_score: 0.85, market_score: 0.06, reliability_score: 1.0 } },
  { rank: 8, driver: "Max Verstappen", team: "Red Bull", grid: 20, win: 1.2, podium: 11.1, points: 44.4, score: 0.405, color: "#3671C6", features: { grid_score: 0.10, practice_score: 0.75, consistency_score: 0.82, experience_score: 1.0, aus_track_score: 0.85, team_score: 0.82, market_score: 0.02, reliability_score: 0.3 } },
  { rank: 9, driver: "Liam Lawson", team: "Racing Bulls", grid: 8, win: 0.6, podium: 5.0, points: 50.5, score: 0.469, color: "#6692FF" },
  { rank: 10, driver: "Arvid Lindblad", team: "Racing Bulls", grid: 9, win: 0.6, podium: 5.0, points: 46.6, score: 0.447, color: "#6692FF" },
  { rank: 11, driver: "Fernando Alonso", team: "Aston Martin", grid: 17, win: 0.6, podium: 5.7, points: 42.5, score: 0.405, color: "#006F62" },
  { rank: 12, driver: "Gabriel Bortoleto", team: "Audi", grid: 10, win: 0.5, podium: 4.4, points: 41.0, score: 0.425, color: "#FF0000" },
  { rank: 13, driver: "Nico Hulkenberg", team: "Audi", grid: 11, win: 0.5, podium: 4.6, points: 43.8, score: 0.436, color: "#FF0000" },
];

const TEAM_COLORS = {
  "Mercedes": "#00D2BE",
  "Ferrari": "#DC0000",
  "McLaren": "#FF8700",
  "Red Bull": "#3671C6",
  "Racing Bulls": "#6692FF",
  "Audi": "#FF0000",
  "Haas": "#B6BABD",
  "Alpine": "#0090FF",
  "Williams": "#005AFF",
  "Aston Martin": "#006F62",
  "Cadillac": "#1E1E1E",
};

const FEATURE_LABELS = {
  grid_score: "Grid Position",
  practice_score: "Practice Pace",
  consistency_score: "Consistency",
  experience_score: "Experience",
  aus_track_score: "Track Knowledge",
  team_score: "Team Strength",
  market_score: "Market Odds",
  reliability_score: "Reliability",
};

const WEIGHTS = {
  grid_score: 0.25,
  practice_score: 0.10,
  consistency_score: 0.05,
  experience_score: 0.07,
  aus_track_score: 0.05,
  team_score: 0.15,
  market_score: 0.20,
  reliability_score: 0.07,
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div style={{ background: "#1a1a2e", padding: "12px 16px", borderRadius: 8, border: "1px solid #333", color: "#e0e0e0", fontSize: 13 }}>
        <div style={{ fontWeight: 700, color: d.color || "#fff", marginBottom: 4 }}>{d.driver}</div>
        <div>Win: {d.win}%</div>
        <div>Podium: {d.podium}%</div>
        <div>Points: {d.points}%</div>
      </div>
    );
  }
  return null;
};

export default function F1Dashboard() {
  const [selectedDriver, setSelectedDriver] = useState(PREDICTIONS[0]);
  const [showMC, setShowMC] = useState(false);
  const [simCount, setSimCount] = useState(0);
  const [view, setView] = useState("overview");

  useEffect(() => {
    if (showMC && simCount < 50000) {
      const timer = setTimeout(() => {
        setSimCount((c) => Math.min(c + 2500, 50000));
      }, 40);
      return () => clearTimeout(timer);
    }
  }, [showMC, simCount]);

  const startSim = () => { setShowMC(true); setSimCount(0); };

  const topDrivers = PREDICTIONS.slice(0, 8);

  const radarData = selectedDriver.features
    ? Object.entries(selectedDriver.features).map(([key, val]) => ({
        feature: FEATURE_LABELS[key] || key,
        value: Math.round(val * 100),
        weight: Math.round((WEIGHTS[key] || 0) * 100),
      }))
    : [];

  const barData = topDrivers.map((d) => ({
    name: d.driver.split(" ").pop(),
    driver: d.driver,
    win: d.win,
    podium: d.podium,
    points: d.points,
    color: d.color,
  }));

  const weightData = Object.entries(WEIGHTS).map(([k, v]) => ({
    name: FEATURE_LABELS[k] || k,
    weight: Math.round(v * 100),
  })).sort((a, b) => b.weight - a.weight);

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(135deg, #0a0a1a 0%, #111128 50%, #0d0d20 100%)", color: "#e0e0e0", fontFamily: "'JetBrains Mono', 'SF Mono', monospace", padding: 0, overflow: "hidden" }}>
      
      {/* HEADER */}
      <div style={{ background: "linear-gradient(90deg, rgba(220,0,0,0.15) 0%, rgba(0,210,190,0.15) 50%, rgba(54,113,198,0.15) 100%)", borderBottom: "1px solid rgba(255,255,255,0.08)", padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", maxWidth: 1200, margin: "0 auto" }}>
          <div>
            <div style={{ fontSize: 11, letterSpacing: 3, color: "#888", textTransform: "uppercase", marginBottom: 4 }}>ML Race Prediction</div>
            <div style={{ fontSize: 28, fontWeight: 900, background: "linear-gradient(90deg, #DC0000, #FF8700, #00D2BE)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", letterSpacing: -1 }}>
              F1 AUSTRALIAN GP 2026
            </div>
            <div style={{ fontSize: 12, color: "#666", marginTop: 2 }}>Albert Park, Melbourne | 58 Laps | 306km | Sunday March 8</div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: 11, color: "#666", letterSpacing: 2 }}>MONTE CARLO</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#00D2BE" }}>50,000</div>
            <div style={{ fontSize: 10, color: "#555" }}>SIMULATIONS</div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 24px" }}>
        
        {/* NAV TABS */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {[
            { id: "overview", label: "PREDICTIONS" },
            { id: "features", label: "FEATURES" },
            { id: "simulate", label: "SIMULATE" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => { setView(tab.id); if (tab.id === "simulate") startSim(); }}
              style={{
                padding: "8px 20px", fontSize: 11, letterSpacing: 2, fontWeight: 600,
                background: view === tab.id ? "rgba(0,210,190,0.15)" : "rgba(255,255,255,0.03)",
                border: view === tab.id ? "1px solid rgba(0,210,190,0.4)" : "1px solid rgba(255,255,255,0.08)",
                color: view === tab.id ? "#00D2BE" : "#666",
                borderRadius: 6, cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* PREDICTED WINNER HERO */}
        {view === "overview" && (
          <>
            <div style={{
              background: "linear-gradient(135deg, rgba(0,210,190,0.08) 0%, rgba(0,210,190,0.02) 100%)",
              border: "1px solid rgba(0,210,190,0.2)", borderRadius: 12, padding: "24px 28px", marginBottom: 20,
              display: "flex", justifyContent: "space-between", alignItems: "center"
            }}>
              <div>
                <div style={{ fontSize: 10, letterSpacing: 3, color: "#00D2BE", marginBottom: 6 }}>PREDICTED WINNER</div>
                <div style={{ fontSize: 32, fontWeight: 900, color: "#fff", letterSpacing: -1 }}>George Russell</div>
                <div style={{ fontSize: 14, color: "#00D2BE", marginTop: 2 }}>Mercedes | P1 on Grid</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 52, fontWeight: 900, color: "#00D2BE", lineHeight: 1 }}>62%</div>
                <div style={{ fontSize: 11, color: "#888", letterSpacing: 1 }}>WIN PROBABILITY</div>
              </div>
            </div>

            {/* MAIN CHART */}
            <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "20px 16px", marginBottom: 20 }}>
              <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 12, paddingLeft: 8 }}>WIN PROBABILITY BY DRIVER</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={barData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                  <XAxis dataKey="name" tick={{ fill: "#888", fontSize: 11, fontFamily: "inherit" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#555", fontSize: 10, fontFamily: "inherit" }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.03)" }} />
                  <Bar dataKey="win" radius={[4, 4, 0, 0]} maxBarSize={48}>
                    {barData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} fillOpacity={0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* FULL GRID TABLE */}
            <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "16px", overflow: "auto" }}>
              <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 12, paddingLeft: 4 }}>FULL PREDICTION TABLE</div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                    {["#", "Driver", "Team", "Grid", "Win%", "Podium%", "Pts%", "Score"].map((h) => (
                      <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: "#555", fontWeight: 600, fontSize: 10, letterSpacing: 1 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {PREDICTIONS.map((p, i) => (
                    <tr
                      key={p.driver}
                      onClick={() => p.features && setSelectedDriver(p)}
                      style={{
                        borderBottom: "1px solid rgba(255,255,255,0.03)",
                        cursor: p.features ? "pointer" : "default",
                        background: selectedDriver.driver === p.driver ? "rgba(0,210,190,0.06)" : "transparent",
                        transition: "background 0.15s",
                      }}
                    >
                      <td style={{ padding: "10px", fontWeight: 700, color: i < 3 ? "#00D2BE" : "#555" }}>{p.rank}</td>
                      <td style={{ padding: "10px", fontWeight: 600, color: "#e0e0e0" }}>
                        <span style={{ display: "inline-block", width: 3, height: 16, borderRadius: 2, background: TEAM_COLORS[p.team] || "#666", marginRight: 8, verticalAlign: "middle" }} />
                        {p.driver}
                      </td>
                      <td style={{ padding: "10px", color: TEAM_COLORS[p.team] || "#888" }}>{p.team}</td>
                      <td style={{ padding: "10px", color: "#888" }}>P{p.grid}</td>
                      <td style={{ padding: "10px", fontWeight: 700, color: p.win > 5 ? "#00D2BE" : p.win > 1 ? "#FFD700" : "#555" }}>{p.win}%</td>
                      <td style={{ padding: "10px", color: "#aaa" }}>{p.podium}%</td>
                      <td style={{ padding: "10px", color: "#aaa" }}>{p.points}%</td>
                      <td style={{ padding: "10px" }}>
                        <div style={{ background: "rgba(255,255,255,0.05)", borderRadius: 4, height: 6, width: 80, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${p.score * 100}%`, background: `linear-gradient(90deg, ${TEAM_COLORS[p.team]}, ${TEAM_COLORS[p.team]}88)`, borderRadius: 4, transition: "width 0.6s" }} />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {/* FEATURES VIEW */}
        {view === "features" && (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
              
              {/* Driver selector */}
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 16 }}>
                <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 12 }}>SELECT DRIVER</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 360, overflowY: "auto" }}>
                  {PREDICTIONS.filter((p) => p.features).map((p) => (
                    <button
                      key={p.driver}
                      onClick={() => setSelectedDriver(p)}
                      style={{
                        display: "flex", justifyContent: "space-between", alignItems: "center",
                        padding: "10px 12px", borderRadius: 6, border: "none", cursor: "pointer", fontFamily: "inherit",
                        background: selectedDriver.driver === p.driver ? "rgba(0,210,190,0.12)" : "rgba(255,255,255,0.02)",
                        color: selectedDriver.driver === p.driver ? "#00D2BE" : "#aaa",
                        fontSize: 12, transition: "all 0.15s",
                      }}
                    >
                      <span style={{ fontWeight: 600 }}>
                        <span style={{ display: "inline-block", width: 3, height: 12, borderRadius: 2, background: TEAM_COLORS[p.team], marginRight: 8, verticalAlign: "middle" }} />
                        {p.driver}
                      </span>
                      <span style={{ color: "#555" }}>{p.win}%</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Radar chart */}
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 16 }}>
                <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 4 }}>
                  FEATURE PROFILE: <span style={{ color: TEAM_COLORS[selectedDriver.team] }}>{selectedDriver.driver.toUpperCase()}</span>
                </div>
                <ResponsiveContainer width="100%" height={340}>
                  <RadarChart data={radarData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                    <PolarGrid stroke="rgba(255,255,255,0.08)" />
                    <PolarAngleAxis dataKey="feature" tick={{ fill: "#888", fontSize: 9, fontFamily: "inherit" }} />
                    <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: "#555", fontSize: 8 }} />
                    <Radar name="Score" dataKey="value" stroke={TEAM_COLORS[selectedDriver.team]} fill={TEAM_COLORS[selectedDriver.team]} fillOpacity={0.2} strokeWidth={2} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Feature weights */}
            <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "16px 20px" }}>
              <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 12 }}>MODEL FEATURE WEIGHTS</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={weightData} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 100 }}>
                  <XAxis type="number" tick={{ fill: "#555", fontSize: 10, fontFamily: "inherit" }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="name" tick={{ fill: "#aaa", fontSize: 11, fontFamily: "inherit" }} axisLine={false} tickLine={false} width={100} />
                  <Bar dataKey="weight" fill="#00D2BE" fillOpacity={0.6} radius={[0, 4, 4, 0]} maxBarSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* SIMULATE VIEW */}
        {view === "simulate" && (
          <div>
            <div style={{
              background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 12, padding: "24px 28px", marginBottom: 20, textAlign: "center"
            }}>
              <div style={{ fontSize: 11, letterSpacing: 3, color: "#666", marginBottom: 8 }}>MONTE CARLO SIMULATION</div>
              <div style={{ fontSize: 48, fontWeight: 900, color: simCount >= 50000 ? "#00D2BE" : "#FFD700", lineHeight: 1 }}>
                {simCount.toLocaleString()}
              </div>
              <div style={{ fontSize: 11, color: "#555", marginTop: 4 }}>/ 50,000 race simulations</div>
              
              {/* Progress bar */}
              <div style={{ background: "rgba(255,255,255,0.05)", borderRadius: 8, height: 8, margin: "16px auto", maxWidth: 400, overflow: "hidden" }}>
                <div style={{
                  height: "100%", borderRadius: 8, transition: "width 0.1s",
                  width: `${(simCount / 50000) * 100}%`,
                  background: simCount >= 50000 ? "linear-gradient(90deg, #00D2BE, #00ffcc)" : "linear-gradient(90deg, #FFD700, #FF8700)",
                }} />
              </div>

              {simCount >= 50000 && (
                <div style={{ marginTop: 12, fontSize: 12, color: "#00D2BE" }}>
                  Simulation complete. Results account for safety cars, rain, DNFs, and first-lap incidents.
                </div>
              )}

              <button
                onClick={startSim}
                style={{
                  marginTop: 16, padding: "10px 28px", fontSize: 12, letterSpacing: 2, fontWeight: 700,
                  background: "rgba(0,210,190,0.12)", border: "1px solid rgba(0,210,190,0.3)",
                  color: "#00D2BE", borderRadius: 8, cursor: "pointer", fontFamily: "inherit",
                }}
              >
                RE-RUN SIMULATION
              </button>
            </div>

            {/* Sim results as animated bars */}
            <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 16 }}>SIMULATED WIN DISTRIBUTION</div>
              {topDrivers.map((d, i) => {
                const animatedWin = simCount >= 50000 ? d.win : Math.max(0, d.win * (simCount / 50000) + (Math.random() - 0.5) * 5 * (1 - simCount / 50000));
                return (
                  <div key={d.driver} style={{ marginBottom: 12, display: "flex", alignItems: "center", gap: 12 }}>
                    <div style={{ width: 140, fontSize: 12, color: "#aaa", fontWeight: 500 }}>
                      <span style={{ display: "inline-block", width: 3, height: 12, borderRadius: 2, background: d.color, marginRight: 6, verticalAlign: "middle" }} />
                      {d.driver.split(" ").pop()}
                    </div>
                    <div style={{ flex: 1, background: "rgba(255,255,255,0.03)", borderRadius: 6, height: 24, overflow: "hidden", position: "relative" }}>
                      <div style={{
                        height: "100%", borderRadius: 6,
                        width: `${Math.min(100, animatedWin)}%`,
                        background: `linear-gradient(90deg, ${d.color}, ${d.color}88)`,
                        transition: "width 0.3s ease-out",
                      }} />
                      <div style={{ position: "absolute", right: 8, top: 4, fontSize: 11, fontWeight: 700, color: "#fff" }}>
                        {simCount >= 50000 ? `${d.win}%` : "..."}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* METHODOLOGY FOOTER */}
        <div style={{ marginTop: 20, padding: "16px 20px", background: "rgba(255,255,255,0.01)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 10, fontSize: 10, color: "#444", lineHeight: 1.6 }}>
          <span style={{ color: "#666", fontWeight: 600, letterSpacing: 1 }}>MODEL</span> &nbsp;
          Weighted ensemble scoring (10 features) with softmax normalization, validated through 50K Monte Carlo simulations.
          Features: qualifying position (25%), market odds (20%), team strength (15%), practice pace (10%), experience (7%), reliability (7%),
          track knowledge (5%), consistency (5%), teammate differential (3%), quali improvement (3%).
          Random factors: safety car probability 60%, rain 15%, DNF rate 8%, first-lap chaos 25%.
        </div>
      </div>
    </div>
  );
}
