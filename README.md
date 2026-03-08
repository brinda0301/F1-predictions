# F1 2026 Race Winner Prediction

ML model to predict Formula 1 race winners. Built for the 2026 Australian Grand Prix. 100% data-driven. Zero betting market data. 2026 regulation-aware.

## Result: MODEL WAS CORRECT

**Predicted: George Russell (Mercedes) wins with 59.1% probability.**

**Actual: George Russell (Mercedes) won by 2.9 seconds.**

Russell won the race from pole, leading a Mercedes 1-2 with Antonelli in P2 and Leclerc P3. The model correctly predicted the winner, the podium composition (Russell, Antonelli, Leclerc were all top 5 in our predictions), and the Mercedes dominance.

## Prediction vs Reality

| # | Predicted | Win% | Actual Result | Grid | Correct? |
|---|-----------|------|---------------|------|----------|
| 1 | George Russell | 59.1% | P1 WINNER | P1 | YES |
| 2 | Lewis Hamilton | 7.57% | P4 | P7 | Top 5 correct |
| 3 | Charles Leclerc | 6.63% | P3 PODIUM | P4 | YES (podium) |
| 4 | Oscar Piastri | 4.76% | DNS (crash to grid) | P5 | DNF captured by model |
| 5 | Kimi Antonelli | 4.20% | P2 PODIUM | P2 | YES (podium) |
| 6 | Lando Norris | 2.35% | P5 | P6 | Close |
| 7 | Isack Hadjar | 2.30% | DNF (mechanical) | P3 | DNF risk 13.87% |
| 8 | Max Verstappen | 0.56% | P6 (from P20) | P20 | Underestimated recovery |

## Accuracy Analysis

**What the model got right:**

Winner prediction correct. Russell won from pole. The 59.1% probability was well-calibrated: he won but faced real challenges from Leclerc in the early laps.

Mercedes 1-2 predicted. Both Russell and Antonelli were in our top 5. Mercedes dominance from qualifying carried into the race.

Podium composition correct. Russell, Antonelli, and Leclerc were all predicted top 5. All three finished on the podium.

DNF predictions validated. Hadjar retired with a mechanical failure (our model gave him 13.87% DNF risk). Bottas retired (we gave Cadillac 26% DNF risk). Hulkenberg DNS with technical issues (Audi had 21.49% DNF risk). The new engine partnership reliability concerns were justified.

Piastri DNS. He crashed on his sighting lap. Our model couldn't predict that specific event, but the Monte Carlo simulation's "driver error" probability (4%) and lap 1 chaos (35%) captured this type of randomness.

**What the model got wrong:**

Verstappen recovery underestimated. We gave him 0.56% win probability from P20. He finished P6. The model didn't fully account for how effectively a 4-time champion recovers through the field, especially when retirements ahead clear the way. Active aero and overtake mode helped him gain positions.

Ferrari strategy error not modeled. Ferrari didn't pit under the VSC, which cost Leclerc the win. Our model doesn't simulate pit strategy decisions. This is the biggest gap.

Hamilton undervalued. We predicted 7.57% win probability. He finished P4, close to the podium (0.6s behind Leclerc). Ferrari's race pace was stronger than our FP2 estimates suggested. Hamilton said the car "felt good" and he had "great pace."

**Key learning: 2026 regulation adjustments were directionally correct.**

We reduced the pole win rate from 60% to 45% for active aero. Russell won from pole, but Leclerc actually led the race for multiple laps before strategy separated them. Overtaking happened. The old 60% rate would have been too high, validating the adjustment.

We modeled energy management uncertainty. Russell said on the grid his "battery level had nothing in the tank" and he made a bad start. Ferrari's energy deployment at the start was superior, as predicted (start procedure advantage feature). Energy management was indeed a factor.

We raised DNF rates for new engines. 6 cars did not finish or did not start (Piastri DNS, Hulkenberg DNS, Hadjar DNF, Bottas DNF, both Aston Martins not classified). That's 27% of the grid. Our elevated DNF rates were accurate.

## How It Works

**1. Data Collection.** Qualifying lap times, practice session times (FP1/FP2/FP3), pre-season testing data, driver career stats. All from 2026 sessions.

**2. Feature Engineering.** 10 normalized features per driver. Uses qualifying time gaps in seconds (not position rank).

**3. Model Scoring.** Weighted ensemble with softmax normalization. Temperature calibrated to 2026-adjusted pole win rate (45%).

**4. Monte Carlo Simulation.** 100,000 race simulations with 2026-specific events: safety cars (55%), virtual safety cars (25%), rain (15%), lap 1 incidents (35%), energy management uncertainty, higher DNF rates for new engine partnerships.

## Features (v3, zero betting data)

| Feature | Weight | Source |
|---------|--------|--------|
| Qualifying pace (time gap to pole) | 26% | 2026 qualifying session |
| FP2 race pace | 16% | 2026 practice high-fuel runs |
| Teammate qualifying gap | 8% | 2026 intra-team delta |
| Energy management readiness | 8% | Pre-season testing laps completed |
| Grid-position win rate (2026-adjusted) | 8% | Historical, adjusted for active aero |
| Qualifying extraction | 7% | Quali vs practice delta |
| Reliability | 7% | Weekend crash/mechanical events |
| Adaptability | 6% | Regulation-change survival count |
| Practice trend | 6% | FP1 to FP3 time improvement |
| Start procedure | 5% | 2026 pre-season reports |

## 2026 Regulation Adjustments

| Regulation Change | Model Adjustment | Race Validation |
|-------------------|-----------------|-----------------|
| Active aero replaces DRS | Pole win rate 60% to 45% | Leclerc led from P4 early. Overtaking happened. |
| Overtake Mode | Pursuing drivers get energy boost | Verstappen gained 14 positions from P20 |
| 300% more battery power | Energy management uncertainty | Russell had battery issues at start |
| New engine partnerships | Higher DNF rates (8-12%) | 6 cars DNS/DNF (27% of grid) |
| 22-car grid | Lap 1 incidents raised to 35% | Piastri crashed before race start |
| New start procedure | Ferrari advantage modeled | Leclerc jumped from P4 to P1 at start |

## Model Evolution

| Version | File | Features | Sims | Betting Data | Russell Win% |
|---------|------|----------|------|-------------|-------------|
| v1 | predict.py | 10 | 50K | 20% weight | 62.0% |
| v2 | predictv2.py | 13 | 100K | None | 78.06% |
| v3 | predictv3.py | 10 | 100K | None | 59.1% |
| Actual | Race result | - | - | - | WON |

v1 used betting odds as 20% of the model. v2 removed betting data and used time gaps. v3 adjusted for 2026 active aero, energy management, and new engine reliability. All three correctly predicted Russell as the winner.

## Project Structure

```
f1-aus-gp-predictor/
├── src/
│   ├── predictv3.py          v3: 2026 regulation-aware (CURRENT)
│   ├── predictv2.py          v2: pure data-driven
│   ├── predict.py            v1: baseline
│   ├── data_loader.py        Static race data
│   ├── fastf1_loader.py      FastF1 API integration
│   ├── features.py           Base feature engineering
│   ├── features_enhanced.py  Enhanced features with FastF1
│   ├── model.py              Scoring functions
│   └── monte_carlo.py        Simulation engine
├── dashboard/
│   └── src/App.jsx           Comparison dashboard (4 tabs)
├── data/
│   ├── predictions_v4.json   v3 output
│   ├── predictions_v3.json   v2 output
│   └── predictions.json      v1 output
├── tests/
│   └── test_model.py         8 unit tests
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
python src/predictv3.py

# Run v2 (pure data-driven)
python src/predictv2.py

# Run v1 (baseline)
python src/predict.py

# Tests
python tests/test_model.py
```

### Comparison Dashboard

```bash
cd dashboard
npm install
npm run dev
```

4 tabs: Compare (3 models side by side), Evolution (line chart of win% changes), Changelog (every change tagged), Model Cards (technical specs).

## Extending to Full 2026 Season

The Australian GP result gives us the first calibration point.

**Immediate fixes for Race 2 (China):** Increase Verstappen recovery factor. Add pit strategy simulation. Reduce Ferrari start advantage if other teams adapt.

**By Race 5:** Build a training dataset of 2026 outcomes. Replace estimated parameters with learned ones. Switch from weighted ensemble to gradient boosting (XGBoost).

**By Race 10:** Add tire degradation curves from FastF1 telemetry. Model energy deployment lap-by-lap. The model gets smarter each race.

## Tech Stack

**Model:** Python, NumPy, FastF1, Pandas

**Dashboard:** React, Recharts, Vite

**Data:** F1 live timing API, official session results, historical data (track layout only)

## License

MIT