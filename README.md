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
| 10 | Belgian GP | Antonelli (34.45%) | Verstappen (44.18%) | Antonelli | Correct | Miss |
| 11 | Hungarian GP | Hamilton (29.82%) | Hamilton | Norris | Miss | Miss |
| 12 | Dutch GP | Norris (36.0%) | Norris (42.69%) | Norris | Correct | Correct |
| 13 | Italian GP | Russell (15.48%) | Russell (49.54%) | pending | pending | pending |
 
**After 12 scored races**: Monte Carlo 7/12 winners correct (58%). XGBoost 3/9 since debut (33%). Average podium drivers hit: 1.9 of 3.

**The baseline it has to beat**: always picking the pole sitter gets 9/12 (75%). The model is 16.7 points behind. At 12 races a two-race gap is well inside noise, so neither figure supports a claim yet, but the comparison is the bar and it is published on the dashboard rather than left for a reader to compute. Beating it over a full season is the goal; the backtest in the roadmap is what makes that measurable.
 
Four races this season were decided by mechanical failure, not pace: Russell's power unit at Canada, Antonelli's engine at Barcelona, Antonelli's wheel shield at Britain, Russell's retirement at Belgium. No model predicts a part breaking from qualifying data.
 
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
 
### R13 Monza: The Model Ranks a Back-Row Start Above Pole

Pierre Gasly took a shock pole for Alpine, a team the model ranked ninth on season pace but which posted the fastest single-lap time of the weekend at a 0.0 pace deficit. Monte Carlo puts Gasly seventh at 7.07%.

Two flaws surfaced at once, both measurable from the feature dump.

**Stale hand-set constants beat live measurement.** Gasly reads `quali_pace` 1.0 and `race_pace` 1.0, the maximum on both. But `energy_score` 0.6, `tyre_management` 0.6, `pit_execution` 0.53 and `circuit_fit` 0.6 are hand-tuned Alpine priors from when the car was midfield. Those drag his model score to 0.7011 against Russell's 0.7268. When a team finds pace, the priors override the evidence, which is backwards.

**Grid position barely registers.** Kimi Antonelli starts P22 after a power unit penalty and the model ranks him fourth at 9.72%, above the pole sitter. His `grid_win_rate` reads 0.0017 against Russell's 0.0788, so the feature reads the grid correctly. It carries 0.0717 weight, so an 18-place gap moves his score from 0.7268 to 0.6909. Moving him from P20 to P22 during a grid correction *raised* his win probability, from 9.21% to 9.72%.

A model that improves a driver's chances when you move him further back is not weighting grid position enough. Same finding as Hungary, louder. The fix is track-dependent grid weighting, queued in the roadmap.

### R12 Zandvoort: Best Race of the Season

Both models called Norris. Monte Carlo hit all three podium drivers, swapping only second and third, for a mean position error of 0.67.

Worth noting what nearly cost it. Antonelli finished second, and on the pre-fix run he sat sixth at 4.57% because a FastF1 name mismatch dropped him to the `practice_pace` fallback. Reconciling the name moved him to 9.42% and onto the predicted podium. The bug is described under Data Pipeline; the lesson is that a silent fallback cost a correct podium call and threw no error.

### R11 Hungary: Both Models Missed the Pole Sitter

Both models picked Hamilton from P5 after a three-place impeding penalty. He finished P5. Norris won from pole, and Monte Carlo had him second at 22.57%, XGBoost fourth.

The model leaned on Hamilton's eight Hungaroring wins through `track_history`, weighted 0.0244. Testing the penalty in isolation showed why that was the wrong read: dropping Hamilton three places moved his win probability from 32.45% to 30.3%. A three-place drop at a circuit where nobody overtakes cost him two points of probability. `quali_pace` carries 0.2014 and reads lap time, which a penalty never changes, while `grid_win_rate` carries 0.0717 and is the only feature reading grid position. Roughly seven percent of the feature mass moves when a penalty lands.

That is defensible at Spa. At the Hungaroring it is close to blind. Fix queued: make grid weighting track-dependent, the way softmax temperature already is.

### R9 Britain: Both Models Wrong
 
Both models picked Antonelli. He finished P16. Wheelspin at the start dropped him behind both Ferraris. He recovered to P2 on fresher hards and was closing on Leclerc at over a second per lap when a wheel shield failure broke the car. A track limits penalty finished the job.
 
The real model gap this exposed was Leclerc. He qualified P2, 0.175s off pole, with the strongest race-trim Ferrari of the weekend. Both models ranked him outside the top 3 because his two recent DNFs dragged down his form scores. The model punished him for mechanical failures he did not cause. That is a feature design flaw, not bad luck. Fix queued: split driver-caused DNFs from mechanical DNFs so pace scores are not penalized for parts breaking.
 
### R7 Barcelona: Monte Carlo's Track-History Thesis Validated
 
Monte Carlo backed Hamilton at 28.27% based on his 6 Barcelona wins and 19 seasons of experience. He won by 19.561 seconds, his first victory for Ferrari, ending Mercedes' 6-race winning streak. XGBoost had all 3 podium drivers correct but the top 2 in the wrong order.
 
### R6 Monaco: XGBoost's First Correct Call
 
XGBoost predicted Antonelli at 68.9% from pole. Monte Carlo backed Hamilton at 27.16% on Monaco track history. Antonelli won. The data-driven model proved that qualifying pace dominates at street circuits where overtaking is rare.
 
## Data Pipeline

`fetch_race_data.py` builds race files from the Ergast-compatible timing API at `api.jolpi.ca`. The earlier version read FastF1 only, which pulls fresh sessions from `livetiming.formula1.com`. That host blocks many networks and lags for hours after a session ends, so a Saturday-evening prediction could not be built on it. The API serves qualifying, sprint and race classifications from anywhere within an hour or two. FastF1 is now optional and supplies FP1 times alone, which no public API exposes.

Auto-fetched: grid with qualifying times, sprint results, team pace deficit computed from qualifying, driver form from the previous race result, and the sprint-weekend flag.

Grid penalties apply from the command line rather than by hand editing. Non-penalised drivers keep relative order and fill from the front, then each penalised driver takes their target slot, matching how the FIA forms a grid when several penalties land at once.

```bash
python fetch_race_data.py 11_hungary --round 11 \
    --penalty "Lewis Hamilton:3" --penalty "Andrea Kimi Antonelli:3" \
    --pitlane "Sergio Perez"
```

Results and scoring run in one command. This writes `result.json` and appends the round to `accuracy_history`.

```bash
python fetch_race_data.py 11_hungary --round 11 --result --score
```

Name normalization keys on stable API ids rather than display names, which drift: `rb` resolves to Racing Bulls whether the API calls it "RB F1 Team" or "Racing Bulls". FastF1 driver names are reconciled against the grid by surname, so "Kimi Antonelli" maps to "Andrea Kimi Antonelli", "Oliver Bearman" to "Ollie Bearman", "Alexander Albon" to "Alex Albon".

That reconciliation matters more than it looks. The engine reads FP1 by grid name and assigns `practice_pace` 0.3 to any name it cannot find. Before the fix, three drivers at Zandvoort silently sat on that fallback, including Antonelli, who had topped the session. His win probability read 4.57% instead of 9.42%, and he dropped off the predicted podium. Nothing errored. The only trace was a suspiciously round number in the feature dump.

FP1 is written all-or-nothing for the same reason, and the threshold had to be tightened twice. At Monza the API returned times for 18 of 22 drivers, which cleared an 80 percent gate. The four absent drivers had sat out FP1 for rookie runs, and one of them was the pole sitter. Scored on the 0.3 fallback, Gasly dropped from 7.07% to 4.12% and Verstappen from 10.14% to 5.98%, while Russell, Hamilton and Leclerc each gained roughly 3.3 points they had not earned. The gate now sits at 95 percent and the script names every grid driver missing a time.

The deeper issue lives in the engine, not the fetcher: `practice_pace` defaults to 0.3, a low value, so a driver who did not run reads as a driver who was slow. Those are different things. Moving the default to the median of drivers who did run is on the roadmap.

Hand-edited per race: weather forecast, circuit type, tyre compounds, circuit history. These carry over from the previous `data.py`, so a re-fetch no longer wipes tuning.

## XGBoost Performance
 
| Round | Pick | Actual | Winner | Podium Hits | Training Rows | MAE |
| :---: | --- | --- | :---: | :---: | :---: | :---: |
| 4 | Norris | Antonelli | Miss | 2/3 | 66 | 0.292 |
| 5 | Russell | Antonelli | Miss | 1/3 | 88 | 0.280 |
| 6 | Antonelli | Antonelli | Correct | 2/3 | 110 | 0.303 |
| 7 | Russell | Hamilton | Miss | 3/3 | 125 | 0.357 |
| 8 | Russell | Russell | Correct | 2/3 | 147 | 0.331 |
| 9 | Antonelli | Leclerc | Miss | 2/3 | 169 | 0.336 |
| 10 | Verstappen | Antonelli | Miss | 3/3 | 191 | 0.321 |
| 11 | Hamilton | Norris | Miss | 1/3 | 213 | 0.316 |
| 12 | Norris | Norris | Correct | 2/3 | 235 | 0.396 |
| 13 | Russell | pending | pending | pending | 257 | 0.421 |
 
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

Build the race file. The folder is created automatically.

```bash
python fetch_race_data.py 13_italy --round 13
```

Apply grid penalties in the same command rather than editing by hand. This reproduced the FIA's Monza grid on all 22 positions:

```bash
python fetch_race_data.py 13_italy --round 13 \
    --penalty "Oscar Piastri:3" \
    --pitlane "Alex Albon" --pitlane "Andrea Kimi Antonelli"
```

Set the weather forecast and circuit history in the generated `data.py`, then run the prediction:

```bash
python engine.py 13_italy
```

Launch the local dashboard:

```bash
streamlit run app.py
```

After the race, write the result and score the round:

```bash
python fetch_race_data.py 13_italy --round 13 --result --score
```

## Project Structure
 
```
F1-predictions/
├── engine.py              Monte Carlo + XGBoost + self-calibration
├── app.py                 Local dashboard, runs predictions
├── app_public.py          Public read-only dashboard, deployed to Streamlit Cloud
├── fetch_race_data.py     Timing API pipeline: grid, sprint, penalties, results, scoring
├── config.json            Feature weights, accuracy history, regulation params
├── requirements.txt
└── races/
    ├── 01_australia/ ... 13_italy/
    │   ├── data.py         Race inputs
    │   ├── prediction.json Locked before the race
    │   └── result.json     Actual outcome
```
 
## Tech Stack
 
Python 3.12, NumPy, XGBoost, scikit-learn, FastF1, Streamlit, Plotly. Race data from the Ergast-compatible API at api.jolpi.ca.
 
Deployed free on Streamlit Community Cloud. Every push to main rebuilds the live dashboard automatically.
 
## Roadmap
 
- **Backtest harness**: replay the model against 2024 and 2025 seasons to validate across 60-plus races instead of 12. This is the top priority. At the current sample size the gap against the pole baseline is not statistically distinguishable from zero, so no accuracy claim here is worth much until the sample grows
- **DNF cause split**: separate driver-caused DNFs from mechanical failures so pace scores are not penalized for parts breaking
- **Brier score logging**: track probability calibration quality with a single number after each race
- **XGBoost accuracy history**: log XGBoost results to config so the season chart shows both models
- **Track-dependent grid weighting**: `grid_win_rate` carries the same 0.0717 weight at Monaco and Monza. At R13 this let a P22 start outrank the pole sitter. Circuits where overtaking is rare should weight starting position far higher, the way softmax temperature already varies by track type
- **Ensemble layer**: across recent races XGBoost identifies podium drivers while ordering them wrong, and Monte Carlo orders better than it selects. Let XGBoost pick the podium set and Monte Carlo rank it
- **Refresh hand-set team constants**: `ENERGY_READINESS`, `START_PROCEDURE`, `tyre_management` and `circuit_fit` are set by hand and rarely revisited. At R13 they held Alpine down while measured pace put the car on pole. Priors should decay toward measured performance as the season provides evidence
- **Practice-pace fallback**: a driver missing from `FP1_TIMES` scores 0.3, a low value, so sitting out a session for a rookie run reads as slowness. The median of drivers who did run would treat absence as no information instead of bad information
- **Dead XGBoost features**: at R12 the model assigned `race_pace` and `tyre_management` zero importance, while `tyre_compound_fit` and `energy_score` together carried 51%. With 235 rows at max_depth 3, that concentration needs investigating before more features are added
Next race: Spanish GP, Madring, September 9 to 11, 2026.
 
---
Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).