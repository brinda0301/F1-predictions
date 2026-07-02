# F1 2026 Race Predictor
 
Self-calibrating ML model predicts F1 race winners using qualifying, practice, sprint, tyre, circuit, and weather data. Built around the 2026 FIA regulation overhaul.
 
## What It Does
 
Two models run side by side:
 
- Monte Carlo: 100,000 simulations across 18 weighted features. Self-calibrates feature weights after each race using gradient descent.
- XGBoost: trains on past race features and finishing positions. Re-trains from scratch each race using all completed races.
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
| 7 | Barcelona-Catalunya GP | Hamilton (28.27%) | Russell (34.8%) | Hamilton | ✅ | ❌ |
| 8 | Austrian GP | Russell (48.21%) | Russell (35.1%) | Race upcoming | — | — |
 
Season after 7 races: Monte Carlo 4/7 winners correct (57%). XGBoost 1/4 since debut (25%).
 
2026 streaks:
- Mercedes wins: 6/7 (broken at Barcelona by Hamilton)
- Antonelli consecutive wins: 5 (broken at Barcelona by engine failure)
- Pole-to-win 2026: 5/7 (broken at Canada and Barcelona)
## R8 Austria: Models Agree After 3 Races of Disagreement
 
Both models pick Russell. First model agreement since R5 Canada.
 
- Monte Carlo: Russell 48.21%
- XGBoost: Russell 35.1% (predicted finish 1.98)
The disagreement moves below the winner. Monte Carlo backs Hamilton P2 (12.51%) and Verstappen P3 (8.57%) on Austria track history. Verstappen has 5 Austria wins. XGBoost backs Antonelli P2 and Leclerc P3 on grid position and recent form.
 
Verstappen at P5 grid with 5 Austria wins tests the track_history feature. Antonelli lost pole to a yellow flag situation in Q3. Hamilton P3 comes off his Barcelona Ferrari breakthrough win.
 
## Model Improvements Shipped Before R8
 
Two changes added before Austria to improve winner probability calibration.
 
### 1. DNF Discount
 
Winner probability now factors in DNF risk. Formula: raw_win_pct * (1 - dnf_pct/100). Then renormalize so probabilities sum to 100.
 
Impact at Austria:
- Russell rose from 40.58% to 48.21% (his DNF risk is 6.07%, others are higher)
- Drivers with mechanical concerns get downweighted
Catches the exact failure modes that hit the model at R5 Canada (Russell DNF from lead), R6 Monaco (Verstappen lap 1 stall), and R7 Barcelona (Antonelli engine on lap 63).
 
### 2. Track-Dependent Softmax Temperature
 
The old model used one temperature (0.11) for every track. Monaco should not behave like Monza.
 
New per-track-type temperatures:
- Street circuits: 0.07 (tighter, favorite gets higher probability)
- High-speed circuits: 0.10
- Balanced circuits: 0.12
- Wet races: 0.18 (wider spread, more chaos)
Austria is high_speed, so temperature drops from 0.11 to 0.10. Russell's win probability sharpens as the front-of-field favorite.
 
## R7 Barcelona: Hamilton's First Ferrari Win
 
Monte Carlo validated its track-history thesis. Pre-race odds had Hamilton at 28.27% based on his 6 Barcelona wins and 19 F1 seasons. Hamilton won by 19.561 seconds over Russell.
 
The story:
- Hamilton ended his 22-month win drought (last win Belgium 2024 with Mercedes)
- Ferrari pitted aggressively on lap 12 then lap 28, undercut Russell on the second stop
- Antonelli was P2 with 3 laps to go after overtaking Russell, then engine failure ended his perfect run
- Mercedes 6/6 winning streak ended
- First all-British podium since 1968 (Hamilton, Russell, Norris)
XGBoost predicted podium: Russell, Hamilton, Norris. Actual podium: Hamilton, Russell, Norris. XGBoost identified all 3 podium drivers but had the top 2 reversed. Strong podium accuracy, missed winner.
 
Championship impact: Antonelli lead cut from 66 to 41 points.
 
## R6 Monaco: XGBoost's First Correct Call
 
Antonelli won Monaco from pole. XGBoost predicted him at 68.9%. Monte Carlo backed Hamilton at 27.16% on Monaco track history. Antonelli won.
 
Actual podium: Antonelli, Hamilton, Gasly (reinstated to P3 after Alpine's successful appeal).
 
## FastF1 Integration
 
Added at R6 Monaco. The `fetch_race_data.py` script pulls qualifying, FP1, sprint, and previous race finishes directly from F1's official timing data.
 
Auto-fetched:
- GRID with positions and Q3 times
- FP1_TIMES from practice
- SPRINT_RESULT (if sprint weekend)
- TEAM_PACE_DEFICIT computed from quali times
- DRIVER_EXPERIENCE r1_finish from previous race result.json
Driver name normalization handles "Kimi Antonelli" to "Andrea Kimi Antonelli", "Alexander Albon" to "Alex Albon", "Oliver Bearman" to "Ollie Bearman", "Nico Hülkenberg" to "Nico Hulkenberg".
 
Team name normalization handles "RB F1 Team" to "Racing Bulls", "Alpine F1 Team" to "Alpine", "Cadillac F1 Team" to "Cadillac", "Kick Sauber" to "Audi".
 
Hand-edit after generation:
- WEATHER forecast (no API for race-day predictions)
- CIRCUIT type (high_speed / street / balanced)
- TYRE_COMPOUNDS estimates
- CIRCUIT_HISTORY veteran adjustments
Usage:
 
```bash
python fetch_race_data.py 08_austria --year 2026 --round 8 --circuit Austria
```
 
## XGBoost Performance
 
| Round | XGBoost Pick | Actual Winner | Winner ✓ | Podium Hits |
| :---: | --- | --- | :---: | :---: |
| 4 | Norris | Antonelli | ❌ | 2/3 |
| 5 | Russell | Antonelli | ❌ | 1/3 |
| 6 | Antonelli (68.9%) | Antonelli | ✅ | 2/3 |
| 7 | Russell (34.8%) | Hamilton | ❌ | 3/3 |
| 8 | Russell (35.1%) | Race upcoming | — | — |
 
Training set growth: 66 rows (R4) to 88 (R5) to 110 (R6) to 125 (R7) to 147 (R8).
MAE: 0.292 (R4) to 0.280 (R5) to 0.303 (R6) to 0.357 (R7) to 0.331 (R8).
 
MAE improved at Austria as training data grew and the model sharpened.
 
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
 
## Setup
 
Requires Python 3.12, Windows or Linux.
 
```bash
git clone https://github.com/brinda0301/F1-predictions.git
cd F1-predictions
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
 
## Usage
 
Fetch race data:
 
```bash
python fetch_race_data.py 08_austria --year 2026 --round 8 --circuit Austria
```
 
Hand-edit the generated data.py for CIRCUIT type, WEATHER, TYRE_COMPOUNDS, and CIRCUIT_HISTORY.
 
Run prediction:
 
```bash
python engine.py 08_austria
```
 
Launch dashboard:
 
```bash
streamlit run app.py
```
 
Submit result and calibrate:
 
```bash
python -c "from engine import calibrate; calibrate(8)"
```
 
## File Structure
 
```
f1-gp-predictor/
├── engine.py              # Monte Carlo + XGBoost + calibration
├── app.py                 # Streamlit dashboard
├── fetch_race_data.py     # FastF1 data fetcher
├── config.json            # weights, accuracy history, regulation params
├── requirements.txt
├── .fastf1_cache/
└── races/
    ├── 01_australia/
    ├── 02_china/
    ├── 03_japan/
    ├── 04_miami/
    ├── 05_canada/
    ├── 06_monaco/
    ├── 07_barcelona/
    └── 08_austria/
        ├── data.py
        └── prediction.json
```
 
## Tech Stack
 
- Python 3.12
- NumPy (Monte Carlo simulations)
- XGBoost + scikit-learn
- FastF1 (official F1 timing data)
- Streamlit (dashboard)
- Plotly (charts)
## What Comes Next
 
R9: British GP, July 3 to 5, 2026, Silverstone.
 
Backlog priorities before R9:
- Recent DNF rate per driver over last 5 races (currently static per team)
- Brier score logging after each race for probability calibration tracking
XGBoost training set grows to ~168 rows after R8.
---

Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).