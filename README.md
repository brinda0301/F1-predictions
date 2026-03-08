# F1 Australian GP 2026 Winner Prediction

ML model to predict the winner of the 2026 Formula 1 Australian Grand Prix using weighted ensemble scoring and Monte Carlo simulation.

## Model Overview

**10 engineered features** feed into a softmax scoring function, validated through **50,000 Monte Carlo race simulations**.

| Feature | Weight | Source |
|---------|--------|--------|
| Grid Position | 25% | Qualifying results |
| Market Odds | 20% | Betting implied probability |
| Team Strength | 15% | Testing + practice data |
| Practice Pace | 10% | FP1, FP2, FP3 rankings |
| Experience | 7% | Career wins, podiums, seasons |
| Reliability | 7% | Weekend mechanical/crash issues |
| Track Knowledge | 5% | Historical Australian GP results |
| Consistency | 5% | Practice session variance |
| Teammate Gap | 3% | Intra-team qualifying delta |
| Quali Improvement | 3% | Practice-to-qualifying delta |

Monte Carlo simulations model: safety cars (60%), rain (15%), DNFs (8%), first-lap incidents (25%).

## Prediction Result

George Russell (Mercedes) - **62% win probability** from P1 on grid.

## Project Structure

```
f1-aus-gp-predictor/
├── src/
│   ├── features.py          # Feature engineering pipeline
│   ├── model.py             # Ensemble scoring + softmax
│   ├── monte_carlo.py       # Race simulation engine
│   ├── data_loader.py       # Raw data definitions
│   └── predict.py           # Main entry point
├── dashboard/
│   ├── src/
│   │   └── App.jsx          # React prediction dashboard
│   ├── package.json
│   └── index.html
├── data/
│   └── predictions.json     # Model output
├── tests/
│   └── test_model.py        # Unit tests
├── notebooks/
│   └── exploration.ipynb    # EDA notebook
├── docs/
│   └── methodology.md       # Detailed methodology writeup
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Quick Start

### Python Model

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/f1-aus-gp-predictor.git
cd f1-aus-gp-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run prediction
python src/predict.py
```

### React Dashboard

```bash
cd dashboard
npm install
npm run dev
```

## Tech Stack

**Model:** Python, NumPy, JSON

**Dashboard:** React, Recharts, Vite, Tailwind CSS

## Data Sources

- Formula1.com official qualifying and practice results
- Historical Australian GP results (2017-2025)
- Pre-race betting odds (Caesars, DraftKings, BetMGM)
- Pre-season testing performance data

## License

MIT
