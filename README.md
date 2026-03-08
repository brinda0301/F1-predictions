# F1 Australian GP 2026 Winner Prediction

ML model to predict the winner of the 2026 Formula 1 Australian Grand Prix. 100% data-driven. Zero betting market data.

## Prediction

**George Russell (Mercedes) - 78.06% win probability** from P1 on grid.

100,000 Monte Carlo race simulations. 13 engineered features. All derived from official F1 qualifying, practice, and historical race data.

| # | Driver | Team | Grid | Win% | Podium% |
|---|--------|------|------|------|---------|
| 1 | George Russell | Mercedes | P1 | 78.06 | 89.00 |
| 2 | Lewis Hamilton | Ferrari | P7 | 11.47 | 48.63 |
| 3 | Kimi Antonelli | Mercedes | P2 | 3.45 | 39.18 |
| 4 | Charles Leclerc | Ferrari | P4 | 3.04 | 37.46 |
| 5 | Lando Norris | McLaren | P6 | 1.28 | 22.30 |
| 6 | Oscar Piastri | McLaren | P5 | 1.25 | 21.71 |
| 7 | Isack Hadjar | Red Bull | P3 | 0.52 | 11.24 |
| 8 | Max Verstappen | Red Bull | P20 | 0.43 | 12.37 |

## How It Works

**Stage 1: Data Collection.** Qualifying lap times, FP1/FP2/FP3 session times, historical Albert Park results (2007-2025), driver career stats, and team race pace from long-run simulations.

**Stage 2: Feature Engineering.** 13 features per driver, all normalized 0.0 to 1.0.

**Stage 3: Model Scoring.** Weighted ensemble with softmax normalization (temperature = 0.12, calibrated to Albert Park's historical 60% pole-win rate).

**Stage 4: Monte Carlo Simulation.** 100,000 race simulations with safety cars (60%), virtual safety cars (25%), rain (15%), lap 1 incidents (30%), mechanical DNFs (6%), driver errors (3%), and strategy variance.

## Features (13 total, zero betting data)

| Feature | Weight | Source |
|---------|--------|--------|
| Qualifying pace (time gap to pole) | 28% | Official qualifying times |
| Team race pace (FP2 long-run deficit) | 14% | Practice session data |
| Historical grid-position win rate | 12% | Albert Park results 1996-2025 |
| FP2 race simulation pace | 10% | Practice high-fuel runs |
| Qualifying vs practice extraction | 5% | Delta between Q and FP times |
| Teammate qualifying gap | 5% | Intra-team time difference |
| Career win rate | 5% | Official career statistics |
| Reliability | 5% | Weekend crash/mechanical data |
| Practice improvement trend | 4% | FP1 to FP3 time progression |
| Recent form (2025 season) | 4% | Last season results |
| Australian GP track record | 3% | Circuit-specific win/podium rate |
| Lap 1 start safety | 3% | Grid position risk factor |
| Rain skill | 2% | Experience-based wet ability |

Key design choice: qualifying pace uses the actual time gap in seconds (e.g. 0.8s off pole), not grid position rank. A driver 0.3s off pole is far more competitive than one 2.5s off. Position numbers alone hide this.

## Project Structure

```
f1-aus-gp-predictor/
├── src/
│   ├── predict_v3.py         # Main model (v3, pure data-driven)
│   ├── predict.py            # Original model (v1)
│   ├── data_loader.py        # Static race data
│   ├── fastf1_loader.py      # FastF1 API integration
│   ├── features.py           # Base feature engineering
│   ├── features_enhanced.py  # Enhanced features with FastF1
│   ├── model.py              # Scoring functions
│   └── monte_carlo.py        # Simulation engine
├── dashboard/
│   ├── src/App.jsx           # React prediction dashboard
│   ├── package.json
│   └── index.html
├── data/
│   ├── predictions_v3.json   # v3 model output
│   └── predictions.json      # v1 model output
├── tests/
│   └── test_model.py         # Unit tests
├── docs/
│   └── methodology.md        # Detailed methodology
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

# Run v3 model (recommended)
python src/predict_v3.py

# Run original model
python src/predict.py

# Run tests
python tests/test_model.py
```

### React Dashboard

```bash
cd dashboard
npm install
npm run dev
```

## Tech Stack

**Model:** Python, NumPy, FastF1, Pandas

**Dashboard:** React, Recharts, Vite

**Data:** F1 live timing API (via FastF1), official session results, historical race data (1996-2025)

## Model Versions

| Version | Features | Simulations | Betting Data | Accuracy Driver |
|---------|----------|-------------|--------------|-----------------|
| v1 | 10 | 50,000 | 20% weight | Position-based |
| v3 | 13 | 100,000 | None | Time-gap-based |

v3 produces sharper predictions because it uses time gaps instead of position ranks and derives all signal from F1 data.

## License

MIT