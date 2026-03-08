# F1 2026 Race Winner Prediction

ML model to predict Formula 1 race winners. Built for the 2026 Australian Grand Prix season opener. 100% data-driven. Zero betting market data. 2026 regulation-aware.

## Australian GP Prediction (March 8, 2026)

**George Russell (Mercedes) - 59.1% win probability** from P1 on grid.

100,000 Monte Carlo simulations. 10 features. All derived from 2026 qualifying, practice, and pre-season data.

| # | Driver | Team | Grid | Win% | Podium% | DNF% |
|---|--------|------|------|------|---------|------|
| 1 | George Russell | Mercedes | P1 | 59.10 | 74.04 | 8.55 |
| 2 | Lewis Hamilton | Ferrari | P7 | 7.57 | 30.78 | 10.12 |
| 3 | Charles Leclerc | Ferrari | P4 | 6.63 | 28.52 | 10.41 |
| 4 | Oscar Piastri | McLaren | P5 | 4.76 | 22.95 | 11.34 |
| 5 | Kimi Antonelli | Mercedes | P2 | 4.20 | 22.14 | 11.37 |
| 6 | Lando Norris | McLaren | P6 | 2.35 | 14.84 | 12.50 |
| 7 | Isack Hadjar | Red Bull | P3 | 2.30 | 14.09 | 13.87 |
| 8 | Max Verstappen | Red Bull | P20 | 0.56 | 5.07 | 17.97 |

## How It Works

**1. Data Collection.** Qualifying lap times, practice session times (FP1/FP2/FP3), pre-season testing data, driver career stats. All from 2026 sessions.

**2. Feature Engineering.** 10 normalized features per driver. Uses qualifying time gaps in seconds (not position rank). A driver 0.3s off pole scores very differently from one 2.5s off.

**3. Model Scoring.** Weighted ensemble with softmax normalization. Temperature calibrated to 2026-adjusted pole win rate (45%, down from historical 60% due to active aero).

**4. Monte Carlo Simulation.** 100,000 race simulations with 2026-specific events: safety cars (55%), virtual safety cars (25%), rain (15%), lap 1 incidents (35% with 22-car grid), energy management uncertainty, higher DNF rates for new engine partnerships.

## 2026 Regulation Adjustments

The 2026 season introduced the biggest rule change in F1 history. The model accounts for:

| Regulation Change | Model Adjustment |
|-------------------|-----------------|
| Active aero replaces DRS | Pole win rate reduced 60% to 45%. Overtake factor 1.4x. |
| Overtake Mode (within 1s) | Pursuing drivers get energy boost in simulation |
| 300% more battery power | Energy management uncertainty added as noise |
| New engine partnerships | Higher DNF rates for Ford/Red Bull, Honda/Aston Martin, Audi, Cadillac |
| 22-car grid (Cadillac added) | Lap 1 incident probability raised to 35% |
| New start procedure | Team-specific start advantage (Ferrari +0.3) |
| Lighter cars (768kg, -30kg) | Higher softmax temperature reflecting more uncertainty |

Historical data is used ONLY where track physics transfer (Albert Park wall locations, corner layout, safety car probability). Old-era grid win rates, team pecking orders, and tire patterns are NOT carried forward.

## Features (v3, zero betting data)

| Feature | Weight | Source | Transfers from old regs? |
|---------|--------|--------|--------------------------|
| Qualifying pace (time gap) | 26% | 2026 qualifying | Yes (physics) |
| FP2 race pace | 16% | 2026 FP2 session | Yes (physics) |
| Teammate qualifying gap | 8% | 2026 qualifying | Yes (same car comparison) |
| Energy management readiness | 8% | Pre-season testing laps | No (new for 2026) |
| Grid-position win rate | 8% | Historical, adjusted for active aero | Partially |
| Qualifying extraction | 7% | 2026 quali vs practice delta | Yes (driver skill) |
| Reliability | 7% | 2026 weekend events | Yes (mechanical) |
| Adaptability | 6% | Reg-change survival count | Yes (driver trait) |
| Practice trend | 6% | 2026 FP1 to FP3 progression | Yes (physics) |
| Start procedure | 5% | 2026 pre-season reports | No (new for 2026) |

## Model Evolution

| Version | File | Russell Win% | Betting Data | Key Change |
|---------|------|-------------|-------------|------------|
| v1 | predict.py | 62% | 20% weight | Baseline prototype |
| v2 | predict_v2.py | 78% | None | Pure data, time-gap features |
| v3 | predict_v3.py | 59% | None | 2026 regulation-aware |

v1 used betting odds as 20% of the model. v2 removed betting data and used time gaps instead of position ranks. v3 adjusted for 2026 regulation changes: active aero reducing pole advantage, energy management uncertainty, higher DNF rates for new engine partnerships.

## Project Structure

```
f1-aus-gp-predictor/
├── src/
│   ├── predict_v3.py         # Current model (2026 regulation-aware)
│   ├── predict_v2.py         # v2 (pure data-driven)
│   ├── predict.py            # v1 (baseline)
│   ├── data_loader.py        # Static race data
│   ├── fastf1_loader.py      # FastF1 API integration
│   ├── features.py           # Base feature engineering
│   ├── features_enhanced.py  # Enhanced features with FastF1
│   ├── model.py              # Scoring functions
│   └── monte_carlo.py        # Simulation engine
├── dashboard/
│   ├── src/App.jsx           # React prediction dashboard
│   └── package.json
├── data/
│   ├── predictions_v3.json   # v3 output
│   ├── predictions_v2.json   # v2 output
│   └── predictions.json      # v1 output
├── tests/
│   └── test_model.py         # Unit tests
├── docs/
│   └── methodology.md
├── requirements.txt
└── README.md
```

## Quick Start

```bash
git clone https://github.com/brinda0301/F1-predictions.git
cd F1-predictions

python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt

# Run v3 (recommended, 2026 regulation-aware)
python src/predict_v3.py

# Run v2 (pure data-driven, no reg adjustments)
python src/predict_v2.py

# Run v1 (baseline with betting odds)
python src/predict.py

# Tests
python tests/test_model.py
```

### React Dashboard

```bash
cd dashboard
npm install
npm run dev
```

## Extending to Full 2026 Season

After each race, the model improves:

**Race 1 (Australia):** Compare prediction vs actual result. Calibrate pole win rate, DNF rates, energy management impact.

**Races 2-5 (China, Japan, Bahrain, Saudi Arabia):** Build a training dataset of 2026-era outcomes. Replace estimated parameters with learned ones.

**Race 6+:** Switch from weighted ensemble to gradient boosting (XGBoost) trained on 2026 data. Add tire strategy and pit stop features from FastF1 telemetry.

The model becomes more accurate each race because it replaces 2026 estimates with 2026 facts.

## Tech Stack

**Model:** Python, NumPy, FastF1, Pandas

**Dashboard:** React, Recharts, Vite

**Data:** F1 live timing API (via FastF1), official session results

## License

MIT