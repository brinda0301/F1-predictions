# F1 Australian GP 2026 Winner Prediction

ML model to predict the winner of the 2026 Formula 1 Australian Grand Prix using weighted ensemble scoring, FastF1 telemetry, and Monte Carlo simulation.

## Prediction Result

**George Russell (Mercedes) - 62% win probability** from P1 on grid.

50,000 Monte Carlo race simulations. 13 engineered features. Real qualifying and practice data from the 2026 Australian GP weekend.

## How It Works

The model runs in 4 stages:

**1. Data Collection** - Pulls qualifying results, practice session lap times, sector times, speed trap data, and weather conditions from F1's live timing API via FastF1. Falls back to manually collected data if the API is unavailable.

**2. Feature Engineering** - Transforms raw timing data into 13 normalized features (0.0 to 1.0 scale) for each of the 22 drivers on the grid.

**3. Model Scoring** - Weighted ensemble scoring with softmax normalization converts features into win probabilities.

**4. Monte Carlo Simulation** - 50,000 race simulations model random events: safety cars (60% probability), rain (15%), mechanical DNFs (8%), and first-lap incidents (25%).

## Features

| Feature | Weight | Description |
|---------|--------|-------------|
| Grid Position | 22-25% | Qualifying result (pole = 1.0) |
| Market Odds | 18-20% | Betting implied probability from sportsbooks |
| Team Strength | 13-15% | Car performance from testing and practice |
| Practice Pace | 8-10% | Average FP1/FP2/FP3 ranking |
| Experience | 6-7% | Career wins, podiums, poles, seasons |
| Reliability | 6-7% | Weekend mechanical issues or crashes |
| Sector Balance | 5% | Consistency across track sectors (FastF1) |
| Track Knowledge | 4-5% | Historical Australian GP results |
| Consistency | 4-5% | Practice session variance |
| Speed Advantage | 4% | Straight-line speed vs field (FastF1) |
| Lap Completion | 4% | Laps completed vs field max (FastF1) |
| Teammate Gap | 3% | Intra-team qualifying delta |
| Quali Improvement | 3% | Practice-to-qualifying delta |

Weights shift depending on whether FastF1 telemetry data is available (13 features) or the model uses static data only (10 features).

## Project Structure

```
f1-aus-gp-predictor/
├── src/
│   ├── data_loader.py        # Static race data (grid, stats, odds)
│   ├── fastf1_loader.py      # FastF1 API integration for telemetry
│   ├── features.py           # Base feature engineering (10 features)
│   ├── features_enhanced.py  # Enhanced features with FastF1 (13 features)
│   ├── model.py              # Weighted ensemble + softmax scoring
│   ├── monte_carlo.py        # Race simulation engine
│   └── predict.py            # Main entry point
├── dashboard/
│   ├── src/App.jsx           # React prediction dashboard
│   ├── package.json
│   └── index.html
├── data/
│   └── predictions.json      # Model output
├── tests/
│   └── test_model.py         # 8 unit tests
├── docs/
│   └── methodology.md        # Detailed methodology
├── notebooks/                # EDA notebooks
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

### Python Model

```bash
git clone https://github.com/brinda0301/F1-predictions.git
cd F1-predictions
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python src/predict.py
```

### Run Tests

```bash
python tests/test_model.py
```

### React Dashboard

```bash
cd dashboard
npm install
npm run dev
```

## FastF1 Integration

The model uses [FastF1](https://docs.fastf1.dev/) to pull real telemetry from F1's live timing API:

- Qualifying lap times with sector breakdown (S1, S2, S3)
- Speed trap readings at key track points
- Practice session lap counts and pace
- Weather conditions (temperature, wind, rain)

FastF1 data is cached locally in `data/f1_cache/` to avoid repeated API calls. If FastF1 is unavailable or the API has no data for the session, the model falls back to static data in `data_loader.py`.

```python
from fastf1_loader import load_race_weekend

data = load_race_weekend(2026, "Australia")
# Returns: grid, practice, lap_times, weather, source
```

## Tech Stack

**Model:** Python, NumPy, FastF1, Pandas

**Dashboard:** React, Recharts, Vite

**Data Sources:** F1 live timing API (via FastF1), official qualifying/practice results, betting odds (Caesars, DraftKings, BetMGM), historical race data (2017-2025)

## License

MIT
