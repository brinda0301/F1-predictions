# F1 2026 Race Predictor

Self-calibrating ML model that predicts F1 race winners using qualifying, practice, sprint, tyre, circuit, and weather data. Built around the 2026 FIA regulation overhaul.

## What It Does

Predicts race outcomes using two models that run side by side:

- **Monte Carlo**: 100,000 simulations using 18 weighted features. Self-calibrates feature weights after each race using gradient descent.
- **XGBoost**: trained on past race features and finishing positions. Re-trains from scratch each race using all completed races as data.

One Streamlit dashboard. One command. Dual model comparison with agreement banner, dual podium cards, win probability charts, DNF risk, and feature importance.

## Race Results

| Round | Race | MC Pick | XGB Pick | Actual | MC ✓ | XGB ✓ |
| :---: | --- | --- | --- | --- | :---: | :---: |
| 1 | Australian GP | Russell (59.1%) | n/a | Russell | ✅ | n/a |
| 2 | Chinese GP | Hamilton (55.25%) | n/a | Antonelli | ❌ | n/a |
| 3 | Japanese GP | Antonelli (40.11%) | n/a | Antonelli | ✅ | n/a |
| 4 | Miami GP | Antonelli (19.2%) | Norris | Antonelli | ✅ | ❌ |
| 5 | Canadian GP | Russell (41.1%) | Russell | Antonelli | ❌ | ❌ |
| 6 | Monaco GP | Hamilton (27.16%) | Antonelli (68.9%) | Antonelli | ❌ | ✅ |
| 7 | Barcelona-Catalunya GP | Hamilton (28.27%) | Russell (34.8%) | Race in progress | — | — |

**Season after 6 races**: MC 3/6 winners correct (50%). XGBoost 1/3 since debut.

**2026 streaks**: Mercedes wins 6/6. Antonelli 5 consecutive wins. Pole-to-win 5/6 (broken at Canada when Russell DNF'd from the lead).

## R7 Barcelona: Models Disagree (Third Race in a Row)

Russell took pole, ending Antonelli's three-pole streak. Hamilton P2 split the Mercedes duo. Antonelli P3 starts off the front row for the first time in 2026.

Monte Carlo backs Hamilton 28.27% because:
- 6 Barcelona/Spanish GP wins (most active driver)
- 19 F1 seasons (max adaptability)
- track_history dominates for him

XGBoost backs Russell 34.8% because:
- Pole position
- Training data: 5 of 6 races went to pole sitter
- quali_pace dominance carries over from Monaco

The thesis test: history vs grid. The race answers it.

## R6 Monaco: XGBoost's First Correct Call

The Antonelli win at Monaco was XGBoost's breakthrough moment. The data-driven model picked him at 68.9% from pole. Monte Carlo backed Hamilton at 27.16% based on Monaco track history. Antonelli won. XGBoost validated its quali_pace dominance thesis.

Pre-race odds:
- Monte Carlo: Hamilton 27.16%, Verstappen 21.50%, Antonelli 18.16%
- XGBoost: Antonelli 68.9%, Verstappen pos 3.93, Hamilton pos 4.08

Actual top 3: Antonelli, Hamilton, Gasly (Gasly reinstated to P3 after Alpine's appeal).

What got missed:
- Verstappen DNF on lap 1 (stalled at lights out, both models had him P2)
- Gasly P3 (neither model had him in top 6, Alpine's appeal flipped the official result)

## FastF1 Integration

Added at R6 Monaco. The `fetch_race_data.py` script pulls qualifying, FP1, sprint, and previous race finishes directly from F1's official timing data.

What FastF1 auto-fetches:
- GRID with positions and Q3 times
- FP1_TIMES from practice
- SPRINT_RESULT (if sprint weekend)
- TEAM_PACE_DEFICIT computed from quali times
- DRIVER_EXPERIENCE r1_finish from previous race result.json

Driver name normalization handles "Kimi Antonelli" → "Andrea Kimi Antonelli", "Alexander Albon" → "Alex Albon", "Oliver Bearman" → "Ollie Bearman", "Nico Hülkenberg" → "Nico Hulkenberg".

What still needs hand-editing after generation:
- WEATHER forecast (no API for race-day predictions)
- CIRCUIT type (high_speed / street / balanced)
- TYRE_COMPOUNDS estimates
- CIRCUIT_HISTORY veteran adjustments
- START_PROCEDURE per-team form
- ENERGY_READINESS per-team 2026 hybrid management

Usage:

```bash
python fetch_race_data.py 07_barcelona --year 2026 --round 7 --circuit Barcelona
```

Then hand-edit the five fields above before running prediction.

## XGBoost Performance

| Round | XGBoost Pick | Actual Winner | Correct |
| :---: | --- | --- | :---: |
| 4 | Norris | Antonelli | ❌ |
| 5 | Russell | Antonelli | ❌ |
| 6 | Antonelli (68.9%) | Antonelli | ✅ |
| 7 | Russell (34.8%) | Race in progress | — |

Training set growth: 66 rows (R4) → 88 (R5) → 110 (R6) → 125 (R7).
MAE: 0.292 (R4) → 0.280 (R5) → 0.303 (R6) → 0.357 (R7).

Slight MAE regression after Monaco came from training on chaotic data (lap 1 DNF, late crashes, appeal-driven podium change). XGBoost is sharpening overall but Monaco data added noise.

Inspired by [Mariana Antaya](https://www.linkedin.com/in/mar-antaya/) who runs XGBoost for her F1 predictions.

## 18 Features

| Feature | Category |
| --- | --- |
| quali_pace | Car speed |
| race_pace | Car speed |
| grid_win_rate | Car speed |
| practice_pace | Car speed |
| sprint_score | Driver skill |
| teammate_gap | Driver skill |
| adaptability | Driver skill |
| start_score | Race factor |
| reliability | Race factor |
| energy_score | 2026 reg-specific |
| tyre_management | Tyre/pit |
| pit_execution | Tyre/pit |
| tyre_compound_fit | Tyre/pit |
| fuel_quality | 2026 reg-specific |
| dirty_air | 2026 reg-specific |
| circuit_fit | 2026 reg-specific |
| track_temp | 2026 reg-specific |
| track_history | History |

Weights adjust after each race based on prediction error. Learning rate decays each round so early races cause bigger shifts. Calibration sign bug fixed before R5.

## 2026 FIA Regulation Constants

| Constant | Value | What Changed |
| --- | :---: | --- |
| POLE_WIN_RATE | 0.45 | Active aero replaced DRS |
| DIRTY_AIR_RETENTION | 0.90 | Cars keep 90% downforce when following (was 70%) |
| OVERTAKE_BOOST | 1.4 | Overtake Mode replaces DRS |
| ENERGY_NOISE | 0.06 | 350kW MGU-K, 50/50 power split |
| WEIGHT_VARIANCE | 0.015 | Cars 76kg lighter (724kg) |

Plus per-team constants for fuel suppliers (sustainable fuel mandatory), DNF rates, pit crew speeds, and tyre management.

## Setup

Requires Python 3.12, Windows or Linux.

```bash
git clone https://github.com/brinda0301/F1-predictions.git
cd F1-predictions
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate      # Linux/Mac
pip install -r requirements.txt
```

## Usage

Fetch race data from FastF1:

```bash
python fetch_race_data.py 07_barcelona --year 2026 --round 7 --circuit Barcelona
```

Hand-edit the generated data.py to add CIRCUIT type, WEATHER forecast, TYRE_COMPOUNDS, and CIRCUIT_HISTORY.

Run prediction:

```bash
python engine.py 07_barcelona
```

Launch the dashboard:

```bash
streamlit run app.py
```

Submit a race result and calibrate:

```bash
python -c "from engine import calibrate; calibrate(7)"
```

## File Structure

```
f1-gp-predictor/
├── engine.py              # Prediction engine: Monte Carlo + XGBoost + calibration
├── app.py                 # Streamlit dashboard with dual model comparison
├── fetch_race_data.py     # FastF1 data fetcher
├── config.json            # Feature weights + accuracy history + regulation params
├── requirements.txt
├── .fastf1_cache/         # Local cache for FastF1 API calls
└── races/
    ├── 01_australia/
    ├── 02_china/
    ├── 03_japan/
    ├── 04_miami/
    ├── 05_canada/
    ├── 06_monaco/
    └── 07_barcelona/
        ├── data.py
        └── prediction.json
```

## Tech Stack

- Python 3.12
- NumPy (Monte Carlo simulations)
- XGBoost + scikit-learn (data-driven model)
- FastF1 (official F1 timing data ingestion)
- Streamlit (dashboard)
- Plotly (charts)

## What's Next

- **Round 8**: Canadian GP, June 26 to 28, 2026 (calendar reshuffle moves it to back-to-back with Barcelona)
- XGBoost training set grows to ~150 rows after R7
- The first race where XGBoost gets 4 races of post-calibration weights to train against
- Track whether the pole position dominance pattern holds across track types

---

Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).