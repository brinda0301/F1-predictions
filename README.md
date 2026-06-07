# F1 2026 Race Predictor
 
Self-calibrating ML model that predicts F1 race winners using qualifying, practice, sprint, tyre, circuit, and weather data. Built around the 2026 FIA regulation overhaul.
 
## What It Does
 
Predicts race outcomes using two models that run side by side:
 
- **Monte Carlo**: 100,000 simulations using 18 weighted features. Self-calibrates feature weights after each race using gradient descent.
- **XGBoost**: trained on past race features and finishing positions. Re-trains from scratch each race using all completed races as data.
One Streamlit dashboard. One command. Dual model comparison with agreement banner, podium cards, win probability charts, DNF risk, and feature importance.
 
## Race Results
 
| Round | Race | MC Pick | XGB Pick | Actual | MC ✓ | XGB ✓ |
| :---: | --- | --- | --- | --- | :---: | :---: |
| 1 | Australian GP | Russell (59.1%) | n/a | Russell | ✅ | n/a |
| 2 | Chinese GP | Hamilton (55.25%) | n/a | Antonelli | ❌ | n/a |
| 3 | Japanese GP | Antonelli (40.11%) | n/a | Antonelli | ✅ | n/a |
| 4 | Miami GP | Antonelli (19.2%) | Norris | Antonelli | ✅ | ❌ |
| 5 | Canadian GP | Russell (41.1%) | Russell | Antonelli | ❌ | ❌ |
| 6 | Monaco GP | Hamilton (27.16%) | Antonelli (68.9%) | Race upcoming | — | — |
 
**Season after 5 races**: 3/5 winners correct (60%). XGBoost: 0/2 since debut.
 
**2026 streaks**: Mercedes wins 5/5. Antonelli 4 consecutive wins. Pole-to-win broken at Canada when Russell DNF'd from the lead.
 
## R6 Monaco: Models Disagree
 
The biggest split of the season so far. Monte Carlo and XGBoost picked different winners.
 
Monte Carlo backs Hamilton 27.16% because:
- 3 Monaco wins (most of any active driver)
- 19 F1 seasons (max adaptability)
- Track history is the strongest predictive feature at Monaco
XGBoost backs Antonelli 68.9% because:
- Pole position
- quali_pace is XGBoost's dominant feature at 45.8% importance
- track_history hasn't predicted any 2026 outcome yet in training data
If history matters at Monaco, Hamilton wins from P3. If pole position matters more, Antonelli converts and extends his streak to 5.
 
## FastF1 Integration
 
Added at R6 Monaco. The `fetch_race_data.py` script pulls qualifying, FP1, sprint, and previous race finishes directly from F1's official timing data. Replaces the manual data.py typing workflow.
 
What FastF1 auto-fetches:
- GRID with positions and Q3 times
- FP1_TIMES from practice
- SPRINT_RESULT (if sprint weekend)
- TEAM_PACE_DEFICIT computed from quali times
- DRIVER_EXPERIENCE r1_finish from previous race result.json
What still needs hand-editing after generation:
- WEATHER forecast (no API for race-day predictions)
- CIRCUIT type (high_speed / street / balanced)
- TYRE_COMPOUNDS estimates
- CIRCUIT_HISTORY veteran adjustments
- START_PROCEDURE per-team form
- ENERGY_READINESS per-team 2026 hybrid management
Usage:
 
```bash
python fetch_race_data.py 06_monaco --year 2026 --round 8 --circuit Monaco
```
 
Then hand-edit the five fields above before running prediction.
 
## XGBoost Performance
 
| Round | XGBoost Pick | Actual Winner | Correct |
| :---: | --- | --- | :---: |
| 4 | Norris | Antonelli | ❌ |
| 5 | Russell | Antonelli | ❌ |
| 6 | Antonelli | Race upcoming | — |
 
Training set: 110 rows after R5 (was 88 after R4, 66 after R3).
MAE: improving from 0.29 (Miami) to 0.28 (Canada) to 0.30 (Monaco).
 
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
python fetch_race_data.py 06_monaco --year 2026 --round 8 --circuit Monaco
```
 
Hand-edit the generated data.py to add CIRCUIT type, WEATHER forecast, TYRE_COMPOUNDS, and CIRCUIT_HISTORY.
 
Run prediction:
 
```bash
python engine.py 06_monaco
```
 
Launch the dashboard:
 
```bash
streamlit run app.py
```
 
Submit a race result and calibrate:
 
```bash
python -c "from engine import calibrate; calibrate(6)"
```
 
## File Structure
 
```
f1-gp-predictor/
├── engine.py              # Prediction engine: Monte Carlo + XGBoost + calibration
├── app.py                 # Streamlit dashboard
├── fetch_race_data.py     # FastF1 data fetcher (auto-generates data.py)
├── config.json            # Feature weights + accuracy history + regulation params
├── requirements.txt
├── .fastf1_cache/         # Local cache for FastF1 API calls
└── races/
    ├── 01_australia/
    ├── 02_china/
    ├── 03_japan/
    ├── 04_miami/
    ├── 05_canada/
    └── 06_monaco/
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
 
- **Round 7**: Spanish GP, June 11 to 14, 2026
- XGBoost training set grows to ~130 rows after R6
- First race where calibration has run with the fixed sign logic for two consecutive races
- Track if XGBoost's quali_pace dominance generalises to Barcelona's high-speed layout
---

Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).