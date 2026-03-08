import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, Cell, LineChart, Line } from "recharts";

const MODELS = {
  v1: {
    name: "v1 Baseline", tag: "predict.py", sims: 50000, features: 10,
    betting: "20% weight", temp: 0.15, poleWinRate: "60% (historical)", color: "#666",
    drivers: [
      { driver: "George Russell", team: "Mercedes", grid: 1, win: 62.0, podium: 74.7 },
      { driver: "Lewis Hamilton", team: "Ferrari", grid: 7, win: 12.4, podium: 42.9 },
      { driver: "Charles Leclerc", team: "Ferrari", grid: 4, win: 8.8, podium: 42.8 },
      { driver: "Kimi Antonelli", team: "Mercedes", grid: 2, win: 3.1, podium: 22.3 },
      { driver: "Lando Norris", team: "McLaren", grid: 6, win: 3.0, podium: 21.0 },
      { driver: "Isack Hadjar", team: "Red Bull", grid: 3, win: 2.1, podium: 15.1 },
      { driver: "Oscar Piastri", team: "McLaren", grid: 5, win: 1.8, podium: 13.2 },
      { driver: "Max Verstappen", team: "Red Bull", grid: 20, win: 1.2, podium: 11.1 },
    ],
  },
  v2: {
    name: "v2 Pure Data", tag: "predictv2.py", sims: 100000, features: 13,
    betting: "None", temp: 0.12, poleWinRate: "60% (historical)", color: "#FF8700",
    drivers: [
      { driver: "George Russell", team: "Mercedes", grid: 1, win: 78.06, podium: 89.0 },
      { driver: "Lewis Hamilton", team: "Ferrari", grid: 7, win: 11.47, podium: 48.63 },
      { driver: "Kimi Antonelli", team: "Mercedes", grid: 2, win: 3.45, podium: 39.18 },
      { driver: "Charles Leclerc", team: "Ferrari", grid: 4, win: 3.04, podium: 37.46 },
      { driver: "Lando Norris", team: "McLaren", grid: 6, win: 2.35, podium: 14.84 },
      { driver: "Oscar Piastri", team: "McLaren", grid: 5, win: 1.25, podium: 21.71 },
      { driver: "Isack Hadjar", team: "Red Bull", grid: 3, win: 2.3, podium: 14.09 },
      { driver: "Max Verstappen", team: "Red Bull", grid: 20, win: 0.43, podium: 12.37 },
    ],
  },
  v3: {
    name: "v3 Regulation-Aware", tag: "predictv3.py", sims: 100000, features: 10,
    betting: "None", temp: 0.14, poleWinRate: "45% (active aero)", color: "#00D2BE",
    drivers: [
      { driver: "George Russell", team: "Mercedes", grid: 1, win: 59.1, podium: 74.04, dnf: 8.55 },
      { driver: "Lewis Hamilton", team: "Ferrari", grid: 7, win: 7.57, podium: 30.78, dnf: 10.12 },
      { driver: "Charles Leclerc", team: "Ferrari", grid: 4, win: 6.63, podium: 28.52, dnf: 10.41 },
      { driver: "Oscar Piastri", team: "McLaren", grid: 5, win: 4.76, podium: 22.95, dnf: 11.34 },
      { driver: "Kimi Antonelli", team: "Mercedes", grid: 2, win: 4.2, podium: 22.14, dnf: 11.37 },
      { driver: "Lando Norris", team: "McLaren", grid: 6, win: 2.35, podium: 14.84, dnf: 12.50 },
      { driver: "Isack Hadjar", team: "Red Bull", grid: 3, win: 2.3, podium: 14.09, dnf: 13.87 },
      { driver: "Max Verstappen", team: "Red Bull", grid: 20, win: 0.56, podium: 5.07, dnf: 17.97 },
    ],
  },
};

const TEAM_COLORS = { "Mercedes": "#00D2BE", "Ferrari": "#DC0000", "McLaren": "#FF8700", "Red Bull": "#3671C6" };
const TOP_DRIVERS = ["George Russell", "Lewis Hamilton", "Charles Leclerc", "Kimi Antonelli", "Oscar Piastri", "Lando Norris", "Isack Hadjar", "Max Verstappen"];

const CHANGES = [
  { from: "v1", to: "v2", items: [
    { change: "Removed betting odds (was 20% weight)", type: "removed" },
    { change: "Switched from position rank to time gap in seconds", type: "improved" },
    { change: "Added qualifying extraction, career win rate, rain skill features", type: "added" },
    { change: "Doubled simulations: 50K to 100K", type: "improved" },
  ]},
  { from: "v2", to: "v3", items: [
    { change: "Pole win rate: 60% to 45% (active aero)", type: "regulation" },
    { change: "Added energy management readiness feature", type: "regulation" },
    { change: "Added start procedure feature (Ferrari advantage)", type: "regulation" },
    { change: "Higher DNF rates for new engine partnerships", type: "regulation" },
    { change: "Active aero overtaking boost in simulation", type: "regulation" },
    { change: "Lap 1 incidents: 30% to 35% (22-car grid)", type: "regulation" },
    { change: "DNF tracking added to output", type: "added" },
  ]},
];

export default function F1ComparisonDashboard() {
  const [selectedDriver, setSelectedDriver] = useState("George Russell");
  const [view, setView] = useState("compare");

  const driverComparison = Object.entries(MODELS).map(([key, model]) => {
    const d = model.drivers.find((d) => d.driver === selectedDriver);
    return { model: model.name, win: d ? d.win : 0, podium: d ? d.podium : 0, color: model.color };
  });

  const multiDriverData = TOP_DRIVERS.map((name) => {
    const row = { driver: name.split(" ").pop() };
    Object.entries(MODELS).forEach(([key, model]) => {
      const d = model.drivers.find((d) => d.driver === name);
      row[key] = d ? d.win : 0;
    });
    return row;
  });

  const lineData = [
    { version: "v1 Baseline", ...Object.fromEntries(TOP_DRIVERS.slice(0, 5).map((n) => [n.split(" ").pop(), MODELS.v1.drivers.find((d) => d.driver === n)?.win || 0])) },
    { version: "v2 Pure Data", ...Object.fromEntries(TOP_DRIVERS.slice(0, 5).map((n) => [n.split(" ").pop(), MODELS.v2.drivers.find((d) => d.driver === n)?.win || 0])) },
    { version: "v3 Reg-Aware", ...Object.fromEntries(TOP_DRIVERS.slice(0, 5).map((n) => [n.split(" ").pop(), MODELS.v3.drivers.find((d) => d.driver === n)?.win || 0])) },
  ];

  const driverColors = { Russell: "#00D2BE", Hamilton: "#DC0000", Leclerc: "#DC0000", Antonelli: "#00D2BE", Piastri: "#FF8700" };

  return (
    <div style={{ minHeight: "100vh", background: "linear-gradient(135deg, #0a0a1a 0%, #0f0f25 50%, #0d0d20 100%)", color: "#e0e0e0", fontFamily: "'JetBrains Mono', 'SF Mono', monospace" }}>
      <div style={{ background: "linear-gradient(90deg, rgba(102,102,102,0.1) 0%, rgba(255,135,0,0.1) 33%, rgba(0,210,190,0.1) 66%)", borderBottom: "1px solid rgba(255,255,255,0.06)", padding: "20px 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <div style={{ fontSize: 10, letterSpacing: 3, color: "#888", marginBottom: 4 }}>MODEL COMPARISON DASHBOARD</div>
          <div style={{ fontSize: 28, fontWeight: 900, background: "linear-gradient(90deg, #666, #FF8700, #00D2BE)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>F1 AUSTRALIAN GP 2026</div>
          <div style={{ fontSize: 12, color: "#555", marginTop: 4 }}>3 model versions. v1 baseline. v2 pure data. v3 2026 regulation-aware.</div>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 24px" }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
          {[{ id: "compare", label: "COMPARE" }, { id: "evolution", label: "EVOLUTION" }, { id: "changes", label: "CHANGELOG" }, { id: "cards", label: "MODEL CARDS" }].map((tab) => (
            <button key={tab.id} onClick={() => setView(tab.id)}
              style={{ padding: "8px 18px", fontSize: 11, letterSpacing: 2, fontWeight: 600, background: view === tab.id ? "rgba(0,210,190,0.15)" : "rgba(255,255,255,0.03)", border: view === tab.id ? "1px solid rgba(0,210,190,0.4)" : "1px solid rgba(255,255,255,0.08)", color: view === tab.id ? "#00D2BE" : "#666", borderRadius: 6, cursor: "pointer", fontFamily: "inherit" }}>
              {tab.label}
            </button>
          ))}
        </div>

        {view === "compare" && (<>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
            {Object.entries(MODELS).map(([key, model]) => (
              <div key={key} style={{ background: "rgba(255,255,255,0.02)", border: `1px solid ${model.color}33`, borderRadius: 12, padding: "16px 20px" }}>
                <div style={{ fontSize: 10, letterSpacing: 2, color: model.color, fontWeight: 700 }}>{model.name.toUpperCase()}</div>
                <div style={{ fontSize: 36, fontWeight: 900, color: model.color, lineHeight: 1, marginTop: 6 }}>{model.drivers[0].win}%</div>
                <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>{model.drivers[0].driver}</div>
                <div style={{ marginTop: 8, fontSize: 10, color: "#555", lineHeight: 1.6 }}>
                  Sims: {model.sims.toLocaleString()} | Features: {model.features}<br/>Betting: {model.betting} | Pole rate: {model.poleWinRate}
                </div>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
            {TOP_DRIVERS.map((name) => (
              <button key={name} onClick={() => setSelectedDriver(name)}
                style={{ padding: "6px 14px", fontSize: 11, fontWeight: 600, background: selectedDriver === name ? "rgba(0,210,190,0.12)" : "rgba(255,255,255,0.03)", border: selectedDriver === name ? "1px solid rgba(0,210,190,0.4)" : "1px solid rgba(255,255,255,0.06)", color: selectedDriver === name ? "#00D2BE" : "#888", borderRadius: 6, cursor: "pointer", fontFamily: "inherit" }}>
                {name.split(" ").pop()}
              </button>
            ))}
          </div>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "20px 16px", marginBottom: 20 }}>
            <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 12, paddingLeft: 8 }}>{selectedDriver.toUpperCase()} WIN% ACROSS VERSIONS</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={driverComparison} margin={{ top: 5, right: 20, bottom: 5, left: 20 }}>
                <XAxis dataKey="model" tick={{ fill: "#888", fontSize: 11, fontFamily: "inherit" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8, color: "#e0e0e0", fontSize: 12 }} />
                <Bar dataKey="win" name="Win%" radius={[6, 6, 0, 0]} maxBarSize={60}>
                  {driverComparison.map((e, i) => (<Cell key={i} fill={e.color} fillOpacity={0.8} />))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "20px 16px" }}>
            <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 12, paddingLeft: 8 }}>ALL DRIVERS: WIN% BY VERSION</div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={multiDriverData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                <XAxis dataKey="driver" tick={{ fill: "#888", fontSize: 11, fontFamily: "inherit" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8, color: "#e0e0e0", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11, color: "#888" }} />
                <Bar dataKey="v1" name="v1 Baseline" fill="#666" fillOpacity={0.6} radius={[3, 3, 0, 0]} maxBarSize={24} />
                <Bar dataKey="v2" name="v2 Pure Data" fill="#FF8700" fillOpacity={0.7} radius={[3, 3, 0, 0]} maxBarSize={24} />
                <Bar dataKey="v3" name="v3 Reg-Aware" fill="#00D2BE" fillOpacity={0.8} radius={[3, 3, 0, 0]} maxBarSize={24} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>)}

        {view === "evolution" && (<>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "20px 16px", marginBottom: 20 }}>
            <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 12, paddingLeft: 8 }}>WIN% EVOLUTION ACROSS VERSIONS</div>
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={lineData} margin={{ top: 10, right: 30, bottom: 10, left: 10 }}>
                <XAxis dataKey="version" tick={{ fill: "#888", fontSize: 11, fontFamily: "inherit" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}%`} />
                <Tooltip contentStyle={{ background: "#1a1a2e", border: "1px solid #333", borderRadius: 8, color: "#e0e0e0", fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {Object.entries(driverColors).map(([name, color]) => (
                  <Line key={name} type="monotone" dataKey={name} stroke={color} strokeWidth={2.5} dot={{ r: 5, fill: color }} activeDot={{ r: 7 }} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "20px 24px" }}>
            <div style={{ fontSize: 11, letterSpacing: 2, color: "#666", marginBottom: 16 }}>KEY OBSERVATIONS</div>
            {[
              { label: "Russell win%", val: "62% → 78% → 59%", note: "v2 overestimated (old 60% pole rate). v3 corrected to 45% for active aero." },
              { label: "Hamilton win%", val: "12.4% → 11.5% → 7.6%", note: "Dropped after removing betting odds. Bookmakers valued his experience more than data does." },
              { label: "Verstappen win%", val: "1.2% → 0.4% → 0.6%", note: "v3 gives slight bump via adaptability (survived 3 reg changes) and overtake mode from P20." },
              { label: "DNF tracking", val: "None → None → 8-27%", note: "v3 added DNF risk. New engines (Cadillac 26%, Red Bull/Ford 14%) create realistic variance." },
            ].map((o, i) => (
              <div key={i} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: i < 3 ? "1px solid rgba(255,255,255,0.04)" : "none" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: "#e0e0e0" }}>{o.label}</span>
                  <span style={{ fontSize: 12, color: "#00D2BE", fontWeight: 600 }}>{o.val}</span>
                </div>
                <div style={{ fontSize: 11, color: "#888" }}>{o.note}</div>
              </div>
            ))}
          </div>
        </>)}

        {view === "changes" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {CHANGES.map((block, bi) => (
              <div key={bi} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: "20px 24px" }}>
                <div style={{ fontSize: 11, letterSpacing: 2, color: "#00D2BE", marginBottom: 14 }}>{block.from.toUpperCase()} → {block.to.toUpperCase()}</div>
                {block.items.map((item, i) => {
                  const c = { removed: "#DC0000", improved: "#FF8700", added: "#00D2BE", regulation: "#FFD700" };
                  const l = { removed: "REMOVED", improved: "IMPROVED", added: "ADDED", regulation: "2026 REG" };
                  return (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                      <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1, padding: "3px 8px", borderRadius: 4, background: `${c[item.type]}15`, color: c[item.type], border: `1px solid ${c[item.type]}33`, minWidth: 65, textAlign: "center" }}>{l[item.type]}</span>
                      <span style={{ fontSize: 12, color: "#ccc" }}>{item.change}</span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        )}

        {view === "cards" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
            {Object.entries(MODELS).map(([key, model]) => (
              <div key={key} style={{ background: "rgba(255,255,255,0.02)", border: `1px solid ${model.color}33`, borderRadius: 12, padding: "20px" }}>
                <div style={{ fontSize: 13, fontWeight: 900, color: model.color, letterSpacing: 1, marginBottom: 4 }}>{model.name}</div>
                <div style={{ fontSize: 10, color: "#555", marginBottom: 16 }}>{model.tag}</div>
                {[
                  ["Simulations", model.sims.toLocaleString()], ["Features", model.features],
                  ["Betting data", model.betting], ["Softmax temp", model.temp], ["Pole win rate", model.poleWinRate],
                ].map(([label, val], i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid rgba(255,255,255,0.03)", fontSize: 11 }}>
                    <span style={{ color: "#888" }}>{label}</span><span style={{ color: "#e0e0e0", fontWeight: 600 }}>{val}</span>
                  </div>
                ))}
                <div style={{ marginTop: 16, fontSize: 10, letterSpacing: 1, color: "#555", marginBottom: 8 }}>TOP 5</div>
                {model.drivers.slice(0, 5).map((d, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 11, color: "#aaa" }}>{d.driver.split(" ").pop()}</span>
                    <span style={{ fontSize: 12, fontWeight: 700, color: model.color }}>{d.win}%</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        <div style={{ marginTop: 20, padding: "16px 20px", background: "rgba(255,255,255,0.01)", border: "1px solid rgba(255,255,255,0.04)", borderRadius: 10, fontSize: 10, color: "#444", lineHeight: 1.6 }}>
          <span style={{ color: "#00D2BE", fontWeight: 600 }}>F1 2026 Australian GP</span> &nbsp;
          3 model iterations. v1: baseline with betting odds. v2: pure F1 data, time gaps. v3: 2026 regulation-aware with active aero, energy management, DNF tracking. Zero betting market data in final version.
        </div>
      </div>
    </div>
  );
}