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
| 1 | Australian GP | George Russell (32.61%) | n/a | George Russell | Correct | n/a |
| 2 | Chinese GP | Lewis Hamilton (59.78%) | n/a | Kimi Antonelli | Miss | n/a |
| 3 | Japanese GP | Kimi Antonelli (49.28%) | n/a | Kimi Antonelli | Correct | n/a |
| 4 | Miami GP | Kimi Antonelli (20.47%) | Lando Norris | Kimi Antonelli | Correct | Miss |
| 5 | Canadian GP | George Russell (40.07%) | George Russell | Kimi Antonelli | Miss | Miss |
| 6 | Monaco GP | Lewis Hamilton (27.16%) | Kimi Antonelli | Kimi Antonelli | Miss | Correct |
| 7 | Spanish GP (Barcelona) | Lewis Hamilton (28.27%) | George Russell | Lewis Hamilton | Correct | Miss |
| 8 | Austrian GP | George Russell (48.21%) | George Russell | George Russell | Correct | Correct |
| 9 | British GP | Kimi Antonelli (48.84%) | Kimi Antonelli | Charles Leclerc | Miss | Miss |
| 10 | Belgian GP | Kimi Antonelli (34.45%) | Max Verstappen | Kimi Antonelli | Correct | Miss |
| 11 | Hungarian GP | Lewis Hamilton (29.82%) | Lewis Hamilton | Lando Norris | Miss | Miss |
| 12 | Dutch GP | Lando Norris (36.0%) | Lando Norris | Lando Norris | Correct | Correct |
| 13 | Italian GP | George Russell (15.48%) | George Russell | Kimi Antonelli | Miss | Miss |
| 14 | Spanish GP (Madring) | Lando Norris (25.33%) | Kimi Antonelli | Kimi Antonelli | Miss | Correct |
 
**After 14 scored races**: Monte Carlo 7/14 winners correct (50%). XGBoost 4/11 since debut (36%). Average podium drivers hit: 1.86 of 3.

**The baseline it has to beat**: always picking the pole sitter gets 9/14 (64%). The model is 14.3 points behind. At 14 races a two-race gap is well inside noise, so neither figure supports a claim yet, but the comparison is the bar and it is published on the dashboard rather than left for a reader to compute.

**These numbers were wrong until R14.** Results for R1-R9 were hand-entered and the midfield was 4-7 places out, up to 16 in places. Winners were right throughout, so the headline accuracy never moved, but two podiums were scored 3/3 that were really 2/3, and several mean position errors were badly understated: Canada was recorded as 0.45 and is 11.0, Britain as 0.36 and is 5.33. Every result is now pulled from the official timing API and verified against it. See Three Silent Data Bugs below.

**The baseline it has to beat**: always picking the pole sitter gets 9/13 (69%). The model is 15.4 points behind. At 13 races a two-race gap is well inside noise, so neither figure supports a claim yet, but the comparison is the bar and it is published on the dashboard rather than left for a reader to compute. Beating it over a full season is the goal; the backtest in the roadmap is what makes that measurable.

**Round 14 is the first split decision since Hungary.** Monte Carlo picks Norris from pole; XGBoost picks Antonelli from P2. FP1 is what divides them: Russell topped practice with Antonelli second and Norris only sixth, and XGBoost leans on `practice_pace` more heavily. Both predictions are committed before lights out, so the result says something about which model should own winner selection.
 
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
 
### R13 Monza: The Flaw Called It Better Than the Model Did

Antonelli won from the back of the grid, the first Italian to win at Monza since 1966. Monte Carlo picked Russell, who finished second, for a podium overlap of 2 and a mean position error of 1.67. XGBoost picked Russell too, overlap 1.

The interesting part is what the model said about the winner. Antonelli started P22 and Monte Carlo ranked him fourth at 9.72%, above the pole sitter. Before the race that read as an obvious defect, and it is written up below as one. He then won.

This does not vindicate the weighting, and the fix still stands. The reason is worth stating precisely: Monza is the easiest overtaking circuit on the calendar, so low grid weighting is *correct* there. The queued fix weights grid position by track type, which would leave Monza roughly as it is and raise it sharply at Monaco. The Monza result is evidence for that change, not against it.

What was wrong was the generalisation. "A back-row start outranking pole is indefensible" is true at Monaco and false at Monza. A model can be right for a structural reason and still need fixing, and the reverse is equally possible.

Pole also failed here. Gasly started first and finished seventh, so the baseline dropped to 9/13.

### R13 Monza: The Model Ranks a Back-Row Start Above Pole

Pierre Gasly took a shock pole for Alpine, a team the model ranked ninth on season pace but which posted the fastest single-lap time of the weekend at a 0.0 pace deficit. Monte Carlo puts Gasly seventh at 7.07%.

Two flaws surfaced at once, both measurable from the feature dump.

**Stale hand-set constants beat live measurement.** Gasly reads `quali_pace` 1.0 and `race_pace` 1.0, the maximum on both. But `energy_score` 0.6, `tyre_management` 0.6, `pit_execution` 0.53 and `circuit_fit` 0.6 are hand-tuned Alpine priors from when the car was midfield. Those drag his model score to 0.7011 against Russell's 0.7268. When a team finds pace, the priors override the evidence, which is backwards.

**Starting further back is rewarded, not merely under-penalised.** The Monte Carlo loop gives every driver outside the top five a recovery bonus that scales linearly with grid slot: P22 receives more than P20, and pole receives none at all. Antonelli started P22 and the model ranked him fourth at 9.72%, above the pole sitter. Moving him from P20 to P22 during a grid correction *raised* his win probability, from 9.21% to 9.72%, which is the signature of a bonus rather than a weak penalty.

This write-up originally blamed the `grid_win_rate` weight of 0.0717. That was wrong. A low weight would flatten the effect of grid position, not invert it. The recovery term is the cause, it is not track-dependent, and it applies the same boost at Monaco as at Monza.

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

### The Audit

A full pass over the repo after R14 found fourteen issues. The four that changed published numbers:

**Hand-entered results.** R1-R9 were typed by hand and the midfield was 4-7 places out. Winners were correct throughout, so headline accuracy never moved, but XGBoost had been training on scrambled labels for nine of fourteen races, and `r1_finish` for each following race read from them. All results now come from the API and are verified against it.

**In-sample MAE published as if it were prediction error.** See XGBoost Performance above.

**DNF counted twice.** The simulation already prevents a retired driver from winning, then win probability is discounted again by the same DNF rate. Teams with higher hand-set rates are penalised twice over.

**Reliability is not reliability.** The feature reads the previous race finishing position: top ten scores 0.95, anything lower 0.80, missing 0.50. Finishing eleventh on pace counts as unreliable, a crash that was not the driver's fault counts as unreliable, and a driver absent from the previous race takes the heaviest penalty of all.

Also found: the pole sitter is excluded from a random boost every other driver can receive, because the loop starts at index 1; `practice_pace` defaults to 0.3 for a missing FP1 time, so skipping a session reads as being slow; DNFs are labelled two different ways in the training set; and the hand-set team constants have never been revisited, which is why Alpine was held down at Monza while measured pace put them on pole.

### Three Silent Data Bugs

None of these threw an exception. Each was found by reading a published timing sheet against the generated file, and each changed the prediction.

**Driver name mismatch across sources.** FastF1 writes "Kimi Antonelli", the timing API writes "Andrea Kimi Antonelli". Same for Oliver against Ollie Bearman and Alexander against Alex Albon. The engine looks FP1 up by grid name, so three drivers fell to the `practice_pace` fallback of 0.3. Antonelli had topped the session at Zandvoort and was scored as if slowest: 4.57% instead of 9.42%, off the predicted podium. He finished second. Fixed by reconciling names against the grid by surname.

**Drivers who race without qualifying.** At Madrid the API returned 20 drivers, not 22. Bearman never left the garage after an FP3 crash and Stroll set no time, both cleared to race at the stewards' discretion. The qualifying classification only lists drivers who set a time, so both vanished from the grid entirely. Fixed with `--absent "Driver:Team"`, which places them behind the classified runners with `q_time` of None, which the engine already reads as neutral.

**Fastest lap read as latest session.** The original `best_lap` took Q3, then Q2, then Q1, assuming later sessions are quicker. Albon set 1:35.307 in Q1 and a slower 1:35.532 in Q2, so he was recorded two tenths off his real pace. Harmless at P16. The damaging case is a front-runner who banks a good Q2 lap and has Q3 ruined by a red flag, who would then be scored on the ruined lap. Fixed by taking the minimum across all three sessions.

The pattern matters more than any single bug. All three produced plausible numbers, none produced an error, and all three were caught by eye rather than by anything automated. That is the strongest argument in this repo for the backtest: 60-plus races surface distortions that 13 races hide.

The deeper issue lives in the engine, not the fetcher: `practice_pace` defaults to 0.3, a low value, so a driver who did not run reads as a driver who was slow. Those are different things. Moving the default to the median of drivers who did run is on the roadmap.

Hand-edited per race: weather forecast, circuit type, tyre compounds, circuit history. These carry over from the previous `data.py`, so a re-fetch no longer wipes tuning.

## XGBoost Performance

**Read the MAE column as training error, not prediction error.** It is computed on the same rows the model just fitted, so it measures memorisation. Held out properly, leave-one-race-out across all 14 rounds, the real figure is **3.93 positions**, eight times larger than the ~0.49 the table shows.

**And it loses to the naive baseline.** Predicting that every driver finishes exactly where they started scores **3.37 positions**. XGBoost scores 3.93. On honest data it is worse than assuming nobody overtakes.

That comparison was hidden until the results were fixed. Measured against the old hand-entered results it appeared to win, 3.45 against 3.81, because both the labels it trained on and the labels it was scored against carried the same errors. Correcting the data reversed the finding.

The in-sample number rising across the season, 0.29 up to 0.49, is not evidence of anything either. Training error creeping up as the dataset grows is ordinary regularisation.

Fixing this means either better features or accepting that 279 rows of 18 features cannot beat a one-line heuristic. The backtest is what decides which.

| Round | Pick | Actual | Winner | Podium Hits | Training Rows | MAE |
| :---: | --- | --- | :---: | :---: | :---: | :---: |
| 4 | Lando Norris | Kimi Antonelli | Miss | 1/3 | 66 | 0.292 |
| 5 | George Russell | Kimi Antonelli | Miss | 1/3 | 88 | 0.28 |
| 6 | Kimi Antonelli | Kimi Antonelli | Correct | 2/3 | 110 | 0.303 |
| 7 | George Russell | Lewis Hamilton | Miss | 3/3 | 125 | 0.357 |
| 8 | George Russell | George Russell | Correct | 2/3 | 147 | 0.331 |
| 9 | Kimi Antonelli | Charles Leclerc | Miss | 2/3 | 169 | 0.336 |
| 10 | Max Verstappen | Kimi Antonelli | Miss | 3/3 | 191 | 0.321 |
| 11 | Lewis Hamilton | Lando Norris | Miss | 1/3 | 213 | 0.316 |
| 12 | Lando Norris | Lando Norris | Correct | 2/3 | 235 | 0.396 |
| 13 | George Russell | Kimi Antonelli | Miss | 1/3 | 257 | 0.421 |
| 14 | Kimi Antonelli | Kimi Antonelli | Correct | 2/3 | 279 | 0.486 |

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
    ├── 01_australia/ ... 14_spain/
    │   ├── data.py         Race inputs
    │   ├── prediction.json Locked before the race
    │   └── result.json     Actual outcome
```
 
## Tech Stack
 
Python 3.12, NumPy, XGBoost, scikit-learn, FastF1, Streamlit, Plotly. Race data from the Ergast-compatible API at api.jolpi.ca.
 
Deployed free on Streamlit Community Cloud. Every push to main rebuilds the live dashboard automatically.
 
## Roadmap
 
- **Fix the engine bugs from the audit**: double-counted DNF, the recovery bonus that rewards starting further back, the pole sitter excluded from the random boost, and the reliability feature. These change future predictions only; published predictions are never regenerated
- **Tests**: three would have caught the bugs above before they shipped. Penalty reordering against a known published grid, name reconciliation against the three known mismatches, and a schema check that every grid driver appears in `FP1_TIMES` or is explicitly absent
- **Backtest harness**: replay the model against 2024 and 2025 seasons to validate across 60-plus races instead of 12. This is the top priority. At the current sample size the gap against the pole baseline is not statistically distinguishable from zero, so no accuracy claim here is worth much until the sample grows
- **DNF cause split**: separate driver-caused DNFs from mechanical failures so pace scores are not penalized for parts breaking
- **Brier score logging**: track probability calibration quality with a single number after each race
- **XGBoost accuracy history**: log XGBoost results to config so the season chart shows both models
- **Track-dependent grid weighting**: `grid_win_rate` carries the same 0.0717 weight at Monaco and Monza. At R13 this let a P22 start outrank the pole sitter. Circuits where overtaking is rare should weight starting position far higher, the way softmax temperature already varies by track type
- **Ensemble layer**: across recent races XGBoost identifies podium drivers while ordering them wrong, and Monte Carlo orders better than it selects. Let XGBoost pick the podium set and Monte Carlo rank it
- **Refresh hand-set team constants**: `ENERGY_READINESS`, `START_PROCEDURE`, `tyre_management` and `circuit_fit` are set by hand and rarely revisited. At R13 they held Alpine down while measured pace put the car on pole. Priors should decay toward measured performance as the season provides evidence
- **Practice-pace fallback**: a driver missing from `FP1_TIMES` scores 0.3, a low value, so sitting out a session for a rookie run reads as slowness. The median of drivers who did run would treat absence as no information instead of bad information
- **Dead XGBoost features**: at R12 the model assigned `race_pace` and `tyre_management` zero importance, while `tyre_compound_fit` and `energy_score` together carried 51%. With 235 rows at max_depth 3, that concentration needs investigating before more features are added
Next race: Azerbaijan GP, Baku, September 25 to 27, 2026.
 
---
Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).