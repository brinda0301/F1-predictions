# F1 2026 Race Predictor

Monte Carlo vs XGBoost, predicting every 2026 Grand Prix winner from timing data. Each prediction is committed to GitHub before lights out and never edited, then scored against a naive "always pick pole" baseline.

**Live dashboard: [f1-predictions-bb.streamlit.app](https://f1-predictions-bb.streamlit.app/)**

## Headline Findings

1. **The original model added nothing over the grid.** A 48-race backtest of 2024-2025 showed it picked the pole sitter in 45 races and tied "always pick pole" at 28 winners each. Fifteen of eighteen features changed nothing.
2. **Both models were rebuilt at R17 and tested on races they had not seen.** XGBoost v2 cut log loss from 1.461 to 1.062 and went from 5 to 8 correct winners over 13 rounds. Monte Carlo went from 1.390 to 1.083 on 2026 and from 1.821 to 1.445 on 2024-2025.
3. **Live since the rebuild, both models beat the pole baseline on log loss in all three races.** Average across R17-R19: Monte Carlo 1.352, XGBoost 1.731, always picking pole 1.966. Both models named the R17 and R18 winners, both from pole, and missed R19, won by Verstappen from P5.
4. **Winner counts barely move. Probability on the winner does.** Hit rate cannot separate a good model from the grid, so every round is scored on log loss.

## The Two Models

**Monte Carlo** (`engine.py`). Eighteen weighted features per driver feed a softmax, then 100,000 simulated races add safety cars, rain, retirements and strategy noise. Feature weights recalibrate after each race by gradient descent.

**XGBoost v2** (`xgb_model.py`, from R17). A win classifier trained on 2022-2026 races, 2,214 driver rows after R18. Six timing inputs: qualifying gap to the fastest lap, log grid slot, gap to teammate, sprint finish, circuit overtaking index, margin to the next-fastest car. Monotone constraints guarantee a faster lap, better grid slot or better sprint finish never lowers a driver's chances. Refit after every race.

XGBoost v1, R4-R16, was a position regressor on 2026 data alone. At R17 it ranked the pole sitter tenth after he won the sprint. Its code is in git history.

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
| 15 | Azerbaijan GP | George Russell (90.25%) | George Russell | George Russell | Correct | Correct |
| 16 | Bahrain GP in Malaysia (Sepang) | Max Verstappen (40.97%) | Isack Hadjar | Max Verstappen | Correct | Miss |
| 17 | Singapore GP | Max Verstappen (77.0%) | Max Verstappen (53.5%) | Max Verstappen | Correct | Correct |
| 18 | United States GP | Kimi Antonelli (66.7%) | Kimi Antonelli (58.3%) | Kimi Antonelli | Correct | Correct |
| 19 | Mexico City GP | Lewis Hamilton (40.5%) | Lewis Hamilton (44.9%) | Max Verstappen | Miss | Miss |
| 20 | Brazilian GP | Lando Norris (60.56%) | Lando Norris (77.0%) | pending | | |

**Kimi Antonelli clinched the 2026 title at Mexico City** with a third place and four races left, the youngest champion in F1 history at 20. He won 9 of 19 races. Monte Carlo picked him to win 5 times and was right 4 of those.

**After 19 races**: Monte Carlo 11/19 winners (58%). XGBoost 7/16 (44%). Always picking pole: 13/19 (68%).

**Log loss, R1-R16 as published** (lower is better): Monte Carlo 1.490, XGBoost v1 1.461 over 13 rounds, pole baseline 1.086. The pole baseline beat both published models. That gap is what the R17 upgrades target.

**Rebuilt models, live** (log loss, lower is better):

| Race | Winner | Monte Carlo | XGBoost | Pole baseline | Note |
| --- | --- | :---: | :---: | :---: | --- |
| R17 Singapore | Verstappen, from pole | 0.261 | 0.626 | 0.684 | XGBoost had Leclerc P2, correct. Monte Carlo had Hamilton P2 on experience, he finished P6 |
| R18 United States | Antonelli, from pole | 0.405 | 0.539 | 0.684 | Monte Carlo called the exact podium. Red Bull P3 and P6, as Monte Carlo's team constants implied |
| R19 Mexico City | Verstappen, from P5 | 3.390 | 4.029 | 4.531 | First race won from off pole since the rebuild. Monte Carlo had Verstappen 6th at 3.4%, XGBoost 5th at 1.8%. Both models had Russell P2, he finished P6. Norris, the pick questioned before the race, finished P5 |

Mexico is the case flagged in the season review: the sharper Monte Carlo puts little on drivers behind the favourite, so a P5 winner costs about 3.4. Both models still lost less than the pole baseline, which gives P5 about 1%. First of the six frozen races.

**R20 Brazil, wet qualifying.** Rain made the track about 3s faster from Q1 to Q3, so lap times from different sessions are not comparable. The prediction uses each driver's Q1 lap, the only session with a lap from all 22 drivers on the same track. Using last-session laps instead put Verstappen, P3 in Q1 and starting P21 after a 30-place engine penalty, 3.8s off pole and at 0.3%. With Q1 laps both models give him a real chance: Monte Carlo 8.4%, XGBoost 11.7%. Both still pick Norris from pole. The rejected version is recorded in `races/20_brazil/data.py`. Data choice only, no model settings changed.

Results R1-R9 were first entered by hand and re-pulled from the timing API at R14. Four races were decided by mechanical failure: Canada, Barcelona, Britain, Belgium.

## How Predictions Are Scored

- **Log loss**: minus the log of the probability given to the actual winner. 50% on the winner scores 0.69, 10% scores 2.30. A confident miss costs far more than a hedged one.
- **Brier score**: squared error across every driver's probability.
- **Pole baseline**: P(win | grid slot) from 91 races, 2022-2025. Pole wins 55%, P2 20%, P3 11%. Stored in `grid_prior.json`.

`score_round` writes all three for Monte Carlo, XGBoost and the baseline into `config.json` after each race.

## R17 Model Upgrades

Every change had to lower log loss on races the model had not seen.

### XGBoost v2

| | v1 (R4-R16) | v2 (from R17) |
| --- | --- | --- |
| Training data | 2026 only, 352 rows | 2022-2026, 2,170 rows |
| Target | Finishing position | Won the race, yes or no |
| Features | 18, ten hand-set constants | 6, all from timing data |
| Guardrails | None | Monotone constraints |
| Trees | 300 at depth 3 | 250 at depth 2 |

Walk-forward over the 13 rounds v1 ran, each round trained only on races before it:

| | Log loss | Winners |
| --- | :---: | :---: |
| XGBoost v1 (published) | 1.461 | 5/13 |
| Pole baseline, same rounds | 1.179 | |
| XGBoost v2 | 1.062 (1.046 with the pinned xgboost 3.2.0) | 8/13 |

### Monte Carlo

`mc_backtest.py` runs the simulation itself over the 16 completed 2026 rounds and 48 races rebuilt from 2024-2025.

| Change | 2026 | 2024-2025 |
| --- | :---: | :---: |
| Before | 1.390 | 1.821 |
| Recovery bonus removed | 1.387 | 1.786 |
| + softmax temperature halved | 1.198 | 1.445 |
| + track history off | 1.083 | n/a |

- **Recovery bonus removed.** It rewarded starting further back.
- **Temperature halved.** The softmax was too flat. The average probability on the eventual winner rose from 31% to 48% on 2026.
- **Track history off.** It gave Hamilton 35% from P3 at Singapore on old wins. Tested on 2026 only, since the 2024-2025 rebuild has no history data.
- **Pole time fixed.** `quali_pace` now measures gaps from the fastest lap, not from whoever starts first.

Checked together after the cleanup: these settings score 1.355 across all 64 races. Putting the recovery bonus back scores 1.362.

Caveat: the 2026 Monte Carlo figures use weights calibrated through R16, so they flatter the engine slightly. The 2024-2025 figures do not have that problem.

### Changes at R18, then a freeze

Each change below was tested on 65 races (17 from 2026, 48 rebuilt from 2024-2025) and kept only if log loss improved on both sets.

| Change | 2026 | 2024-2025 | Kept |
| --- | :---: | :---: | :---: |
| Adaptability (years in F1) off | 1.036 to 0.947 | 1.445 to 1.435 | Yes |
| Race pace from the driver's own qualifying gap, not the team's best lap | 0.947 to 0.944 | 1.435 to 1.428 | Yes |
| Neutral values for hand-set team constants | 0.947 to 1.019 | 1.435 to 1.484 | No |
| Teammate gap capped at 0.3s (tested before R19) | worse for Monte Carlo on both sets; XGBoost better on 2026, worse on 2022-2025 | | No |

Practice pace at R18 used each driver's best lap across FP1-FP3, because FastF1 returned no data. Earlier races used FP1 only.

From R19 to the end of the season both models are frozen. No setting changes, so the last six races are a clean out-of-sample test.

The first R17 prediction (commit 3517610) used the old settings: Monte Carlo Verstappen 48.22%, XGBoost Leclerc 36.0%. It was re-run on the same grid before the race and stays in git history.

## Race Weekend Workflow

```bash
# Saturday, after qualifying
python fetch_race_data.py 17_singapore --round 17
#   grid penalties:   --penalty "Driver:places"   (repeatable)
#   pit lane start:   --pitlane "Driver"
#   no quali time:    --absent "Driver:Team"
# set CIRCUIT, TYRE_COMPOUNDS, WEATHER in races/<folder>/data.py
python engine.py 17_singapore
git add races/17_singapore && git commit -m "R17 predictions" && git push

# Sunday, after the race
python fetch_race_data.py 17_singapore --round 17 --result --score
python xgb_model.py --train
```

`fetch_race_data.py` pulls qualifying, sprint and results from the timing API at api.jolpi.ca, and FP1 times from FastF1. Hand-set fields carry over between re-fetches.

## Backtests

```bash
python xgb_model.py --backtest     # XGBoost v1 vs v2, walk-forward over 2026
python mc_backtest.py              # Monte Carlo settings sweep, 2026 and 2024-2025
python history_data.py --index     # circuit overtaking index
```

The first run downloads history into `.backtest_cache/` in a few minutes. Reruns take seconds.

## Tests

```bash
python -m pytest test_pipeline.py test_probscore.py
```

21 tests. `test_pipeline.py` covers the data bugs that reached production: grid penalties, name mismatches across sources, missing drivers, fastest-lap selection, and published predictions never being edited. `test_probscore.py` covers log loss and Brier score, the pole baseline, the overtaking index never reading the race it scores, XGBoost v2 monotonicity, and the simulation never favouring a back-of-grid start.

## Project Structure

```
F1-predictions/
├── engine.py              Monte Carlo: features, simulation, calibration
├── xgb_model.py           XGBoost v2: win classifier, 2022-2026
├── xgb_model.json         Saved v2 model, refit after each race
├── xgb_model_meta.json    v2 training info, circuit index table, 2026 schedule
├── fetch_race_data.py     Timing API pipeline: grid, sprint, penalties, results, scoring
├── probscore.py           Log loss, Brier score, pole baseline
├── grid_prior.json        P(win | grid slot), 2022-2025
├── history_data.py        Race history loader and circuit overtaking index
├── mc_backtest.py         Monte Carlo log loss sweep
├── dashboard.py           The dashboard: race picks, podiums, charts, season log loss, model weights
├── app_public.py          Public entry point, read-only, deployed to Streamlit Cloud
├── app.py                 Local entry point, same view plus a Run Prediction panel
├── config.json            Monte Carlo weights, temperatures, accuracy history
├── test_pipeline.py       Data pipeline tests
├── test_probscore.py      Scoring and model tests
├── docs/season-notes.md   Race-by-race analyses, the R14 audit, data bugs, 2024-25 backtest
└── races/
    └── 01_australia/ ... 17_singapore/
        ├── data.py          Race inputs
        ├── prediction.json  Committed before the race
        └── result.json      Actual outcome
```

## Tech Stack

Python 3.12, NumPy, XGBoost, scikit-learn, FastF1, Streamlit, Plotly. Race data from the Ergast-compatible API at api.jolpi.ca. Deployed on Streamlit Community Cloud. Every push to main rebuilds the dashboard.

## Roadmap

- **Long-run race pace from FP2 and FP3**: fuel-corrected stint averages via FastF1. The strongest candidate for information the grid does not hold. Test it as a seventh XGBoost input.
- **Cut Monte Carlo's dead features**: fifteen of eighteen changed nothing over 48 races.
- **Reliability feature**: it reads the previous race finish, so an eleventh place on pace counts as unreliable. Split driver-caused from mechanical retirements.
- **Practice-pace fallback**: a driver missing from FP1 scores 0.3, so a rookie-run absence reads as slowness. Use the median instead.
- **Mean position error**: still counts a retirement from the front as a 17-place miss.
- **Ensemble**: XGBoost picks the podium set, Monte Carlo orders it.
- **Monte Carlo overconfidence**: the halved temperature leaves little probability for drivers behind the favourite (Antonelli 0.3% at Singapore, finished P3). Test blending with the pole prior on data not used to choose settings.
- **Practice pace for XGBoost**: Leclerc topped FP1 and FP2 at Austin and finished P2 with 3.6% from XGBoost, which reads no practice data.
- **Refresh hand-set team constants** from measured pace. Neutral values failed at R18, so the constants carry signal but are stale.
- **Teammate gap**: rewards beating a teammate who had a bad session (Norris P3 from P6 at Mexico). A cap failed at R19. Revisit with the full season.

Detailed race analyses, the R14 audit and the data bugs: [docs/season-notes.md](docs/season-notes.md).

---

Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).
