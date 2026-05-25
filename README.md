# F1 2026 Race Predictor

Self-calibrating ML model that predicts F1 race winners using qualifying, practice, sprint, tyre, circuit, and weather data. Built around the 2026 FIA regulation overhaul.

## What It Does

Predicts race outcomes using two models that run side by side:

- **Monte Carlo**: 100,000 simulations using 18 weighted features. Self-calibrates feature weights after each race using gradient descent.
- **XGBoost**: trained on past race features and finishing positions. Re-trains from scratch each race using all completed races as data.

One Streamlit dashboard. One command. Dual model comparison with agreement banner, podium cards, win probability charts, DNF risk, and feature importance.

## Race Results

| Round | Race | Predicted Winner | Actual Winner | Correct | Podium Hits |
| :---: | --- | --- | --- | :---: | :---: |
| 1 | Australian GP | Russell (59.1%) | Russell | ✅ | 2/3 |
| 2 | Chinese GP | Hamilton (55.25%) | Antonelli | ❌ | 3/3 |
| 3 | Japanese GP | Antonelli (40.11%) | Antonelli | ✅ | 1/3 |
| 4 | Miami GP | Antonelli (19.2%) | Antonelli | ✅ | 2/3 |
| 5 | Canadian GP | Russell (41.1%) | Antonelli | ❌ | 1/3 |

**Season after 5 races**: 3/5 winners correct (60%). 9/15 podium drivers hit (60%).

**2026 streaks**: Mercedes wins 5/5. Antonelli 4 consecutive wins. Pole-to-win broken at Canada when Russell retired from the lead with a power unit failure on lap 30.

## XGBoost Performance

| Round | XGBoost Pick | Actual Winner | Correct |
| :---: | --- | --- | :---: |
| 4 | Norris | Antonelli | ❌ |
| 5 | Russell | Antonelli | ❌ |

XGBoost record: 0/2 in two races since debut.

Training data: 88 rows after R4. MAE dropped from 0.29 at Miami to 0.28 at Canada. The model is sharpening but still under-weights driver form across the season.

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

Weights adjust after each race based on prediction error. Learning rate decays each round so early races cause bigger shifts.

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

Run prediction for a specific race:

```bash
python engine.py 05_canada
```

Launch the dashboard:

```bash
streamlit run app.py
```

Submit a race result and calibrate (from the dashboard or CLI):

```bash
python -c "from engine import calibrate; calibrate(5)"
```

## File Structure

```
f1-gp-predictor/
├── engine.py              # Prediction engine: Monte Carlo + XGBoost + calibration
├── app.py                 # Streamlit dashboard
├── config.json            # Feature weights + accuracy history + regulation params
├── requirements.txt
└── races/
    ├── 01_australia/
    ├── 02_china/
    ├── 03_japan/
    ├── 04_miami/
    └── 05_canada/
        ├── data.py
        ├── prediction.json
        └── result.json
```

## Tech Stack

- Python 3.12
- NumPy (Monte Carlo simulations)
- XGBoost + scikit-learn (data-driven model)
- Streamlit (dashboard)
- Plotly (charts)

## What's Next

- **Round 6**: Monaco GP, June 4 to 7, 2026
- XGBoost training set grows to 110 rows
- Calibrated weights from R5 lock in the lesson that pole-sitter DNFs happen
- Goal: 4/6 winners on Monte Carlo, first correct call from XGBoost

---

Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).