# F1 2026 Race Predictor
 
A self-calibrating ML system that predicts F1 race winners from qualifying, practice, sprint, tyre, circuit, and weather data. Every prediction is committed to GitHub before the race, timestamped and public, so the track record cannot be edited after the fact.
 
**Live dashboard: [f1-predictions-bb.streamlit.app](https://f1-predictions-bb.streamlit.app/)**
 
## What It Does
 
Two models run side by side on every race:
 
- **Monte Carlo**: 100,000 simulations across 18 weighted features. Self-calibrates its feature weights after each race using gradient descent.
- **XGBoost**: trains on past race features and finishing positions. Re-trains from scratch each race using all completed races.
The public dashboard shows both predictions, the actual result, a Correct or Miss badge per model, and running season accuracy.
 
## Season Track Record
 
| Round | Race | Monte Carlo | XGBoost | Actual | MC | XGB |
| :---: | --- | --- | --- | --- | :---: | :---: |
| 1 | Australian GP | Russell (59.1%) | n/a | Russell | Correct | n/a |
| 2 | Chinese GP | Hamilton (55.25%) | n/a | Antonelli | Miss | n/a |
| 3 | Japanese GP | Antonelli (40.11%) | n/a | Antonelli | Correct | n/a |
| 4 | Miami GP | Antonelli (19.2%) | Norris | Antonelli | Correct | Miss |
| 5 | Canadian GP | Russell (41.1%) | Russell | Antonelli | Miss | Miss |
| 6 | Monaco GP | Hamilton (27.16%) | Antonelli (68.9%) | Antonelli | Miss | Correct |
| 7 | Barcelona-Catalunya GP | Hamilton (28.27%) | Russell (34.8%) | Hamilton | Correct | Miss |
| 8 | Austrian GP | Russell (48.21%) | Russell (35.1%) | Russell | Correct | Correct |
| 9 | British GP | Antonelli (48.84%) | Antonelli (57.4%) | Leclerc | Miss | Miss |
 
**After 9 races**: Monte Carlo 5/9 winners correct (56%). XGBoost 2/6 since debut (33%). Average podium drivers hit: 2.0 of 3.
 
Three races this season were decided by mechanical failure, not pace: Russell's power unit at Canada, Antonelli's engine at Barcelona, Antonelli's wheel shield at Britain. No model predicts a part breaking from qualifying data.
 
## The 18 Features
 
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
| energy_score | 2026 regulation |
| tyre_management | Tyre and pit |
| pit_execution | Tyre and pit |
| tyre_compound_fit | Tyre and pit |
| fuel_quality | 2026 regulation |
| dirty_air | 2026 regulation |
| circuit_fit | 2026 regulation |
| track_temp | 2026 regulation |
| track_history | History |
 
Weights adjust after each race based on prediction error. The learning rate decays each round, so early races cause bigger shifts.
 
## Model Improvements Shipped
 
### DNF Discount (before R8)
 
Winner probability now factors in DNF risk. Raw win probability is multiplied by (1 minus DNF probability), then renormalized so all probabilities sum to 100. At Austria this raised Russell from 40.58% to 48.21% because his mechanical risk was lower than his rivals'.
 
### Track-Dependent Softmax Temperature (before R8)
 
The model used a single temperature (0.11) for every circuit. Monaco should not behave like Monza. New per-track-type values:
 
| Track type | Temperature | Effect |
| --- | :---: | --- |
| Street | 0.07 | Tighter spread, favourite gets higher probability |
| High speed | 0.10 | Moderately tight |
| Balanced | 0.12 | Standard spread |
| Wet | 0.18 | Wide spread, more chaos |
 
## Key Race Analyses
 
### R9 Britain: Both Models Wrong
 
Both models picked Antonelli. He finished P16. Wheelspin at the start dropped him behind both Ferraris. He recovered to P2 on fresher hards and was closing on Leclerc at over a second per lap when a wheel shield failure broke the car. A track limits penalty finished the job.
 
The real model gap this exposed was Leclerc. He qualified P2, 0.175s off pole, with the strongest race-trim Ferrari of the weekend. Both models ranked him outside the top 3 because his two recent DNFs dragged down his form scores. The model punished him for mechanical failures he did not cause. That is a feature design flaw, not bad luck. Fix queued: split driver-caused DNFs from mechanical DNFs so pace scores are not penalized for parts breaking.
 
### R7 Barcelona: Monte Carlo's Track-History Thesis Validated
 
Monte Carlo backed Hamilton at 28.27% based on his 6 Barcelona wins and 19 seasons of experience. He won by 19.561 seconds, his first victory for Ferrari, ending Mercedes' 6-race winning streak. XGBoost had all 3 podium drivers correct but the top 2 in the wrong order.
 
### R6 Monaco: XGBoost's First Correct Call
 
XGBoost predicted Antonelli at 68.9% from pole. Monte Carlo backed Hamilton at 27.16% on Monaco track history. Antonelli won. The data-driven model proved that qualifying pace dominates at street circuits where overtaking is rare.
 
## FastF1 Integration
 
`fetch_race_data.py` pulls qualifying, FP1, sprint results, and previous race finishes directly from F1's official timing data.
 
Auto-fetched: grid with Q3 times, FP1 lap times, sprint results, team pace deficit computed from qualifying, and driver form from the previous race result.
 
Name normalization handles API inconsistencies: "Kimi Antonelli" becomes "Andrea Kimi Antonelli", "Alexander Albon" becomes "Alex Albon", "RB F1 Team" becomes "Racing Bulls", "Kick Sauber" becomes "Audi".
 
Hand-edited per race: weather forecast, circuit type, tyre compounds, circuit history.
 
```bash
python fetch_race_data.py 09_britain --year 2026 --round 9 --circuit British
```
 
## XGBoost Performance
 
| Round | Pick | Actual | Winner | Podium Hits | Training Rows | MAE |
| :---: | --- | --- | :---: | :---: | :---: | :---: |
| 4 | Norris | Antonelli | Miss | 2/3 | 66 | 0.292 |
| 5 | Russell | Antonelli | Miss | 1/3 | 88 | 0.280 |
| 6 | Antonelli | Antonelli | Correct | 2/3 | 110 | 0.303 |
| 7 | Russell | Hamilton | Miss | 3/3 | 125 | 0.357 |
| 8 | Russell | Russell | Correct | 2/3 | 147 | 0.331 |
| 9 | Antonelli | Leclerc | Miss | 2/3 | 169 | 0.336 |
 
## 2026 Regulation Constants
 
| Constant | Value | What Changed |
| --- | :---: | --- |
| POLE_WIN_RATE | 0.45 | Active aero replaced DRS |
| DIRTY_AIR_RETENTION | 0.90 | Cars keep 90% downforce when following, up from 70% |
| OVERTAKE_BOOST | 1.4 | Overtake Mode replaces DRS |
| ENERGY_NOISE | 0.06 | 350kW MGU-K, 50/50 power split |
| WEIGHT_VARIANCE | 0.015 | Cars 76kg lighter at 724kg |
 
## Setup
 
```bash
git clone https://github.com/brinda0301/F1-predictions.git
cd F1-predictions
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
 
## Usage
 
Fetch race data from the FastF1 API:
 
```bash
python fetch_race_data.py 09_britain --year 2026 --round 9 --circuit British
```
 
Hand-edit the generated data.py for circuit type, weather, tyre compounds, and circuit history. Then run the prediction:
 
```bash
python engine.py 09_britain
```
 
Launch the local dashboard:
 
```bash
streamlit run app.py
```
 
After the race, submit the result and recalibrate:
 
```bash
python -c "from engine import calibrate; calibrate(9)"
```
 
## Project Structure
 
```
F1-predictions/
├── engine.py              Monte Carlo + XGBoost + self-calibration
├── app.py                 Local dashboard, runs predictions
├── app_public.py          Public read-only dashboard, deployed to Streamlit Cloud
├── fetch_race_data.py     FastF1 data pipeline
├── config.json            Feature weights, accuracy history, regulation params
├── requirements.txt
└── races/
    ├── 01_australia/ ... 09_britain/
    │   ├── data.py         Race inputs
    │   ├── prediction.json Locked before the race
    │   └── result.json     Actual outcome
```
 
## Tech Stack
 
Python 3.12, NumPy, XGBoost, scikit-learn, FastF1, Streamlit, Plotly.
 
Deployed free on Streamlit Community Cloud. Every push to main rebuilds the live dashboard automatically.
 
## Roadmap
 
- **Backtest harness**: replay the model against 2024 and 2025 seasons to validate across 60-plus races instead of 9
- **DNF cause split**: separate driver-caused DNFs from mechanical failures so pace scores are not penalized for parts breaking
- **Brier score logging**: track probability calibration quality with a single number after each race
- **XGBoost accuracy history**: log XGBoost results to config so the season chart shows both models
Next race: Belgian GP, Spa-Francorchamps, July 17 to 19, 2026.
 
---

Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).