# F1 2026 Race Predictor

Predicts F1 race winners from timing data, commits every prediction to GitHub before the race, and measures the result against a naive baseline.

**The headline finding is a negative one.** Backtested over 48 races from 2024 and 2025, the model picks the pole sitter in 45 of them. It ties "always pick pole" exactly, 28 winners each, because 94% of the time they are the same prediction. Fifteen of its eighteen features change nothing. The XGBoost component loses to predicting that every driver finishes where they started.

That is the point of the repo. The interesting work is not the model, it is the measurement that showed the model adds nothing, and the diagnosis of why: every feature is either derived from qualifying pace or fixed per team, so nothing in it can disagree with the grid.

R15 at Baku is the finding playing out live. The model made its most confident call of the season, 90.25% on the pole sitter, and he won. The baseline scored the same race. Podium overlap was 1 of 3 and the mean position error across the predicted top three was 4.0, the worst of the season. A correct winner from a prediction the baseline also makes is not evidence the model works.

**Live dashboard: [f1-predictions-bb.streamlit.app](https://f1-predictions-bb.streamlit.app/)**

## What It Does

Three models run side by side on every race:

- **Monte Carlo**: 100,000 simulations across 18 weighted features, with weights adjusted after each race by gradient descent.
- **XGBoost**: trains on past race features and finishing positions, re-fitted from scratch each race.
- **Logit** (from R17): four inputs, weights fit once on 91 races from 2022-2025. Grid slot, gap to pole, and both adjusted for how hard the circuit is to pass on. See What Changed in October 2026.

Every prediction is committed before lights out and never regenerated, so the track record cannot be edited after the fact. When a bug is found, the fix applies going forward and the old prediction stands.

`backtest.py` replays 2024 and 2025 through the same feature model with leave-one-race-out scoring, so any change can be tested against 48 races in minutes rather than one data point per fortnight.

The public dashboard shows every model's prediction, the actual result, a Correct or Miss badge per model, running season accuracy, and the pole baseline beside it.

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
 
**After 16 scored races**: Monte Carlo 9/16 winners correct (56%). XGBoost 5/13 since debut (38%). Average podium drivers hit: 1.81 of 3.

**The baseline it has to beat**: always picking the pole sitter gets 11/16 (69%). The model is 12.5 points behind. The pole sitter has now won the last two races, so both the model and the baseline scored both, and the gap has not moved. At 15 races a two-race gap sits inside noise, so neither figure supports a claim yet, but the comparison is the bar and it is published on the dashboard rather than left for a reader to compute.

**These numbers were wrong until R14.** Results for R1-R9 were hand-entered and the midfield was 4-7 places out, up to 16 in places. Winners were right throughout, so the headline accuracy never moved, but two podiums were scored 3/3 that were really 2/3, and several mean position errors were badly understated: Canada was recorded as 0.45 and is 11.0, Britain as 0.36 and is 5.33. Every result is now pulled from the official timing API and verified against it. See Four Silent Data Bugs below.

Four races this season were decided by mechanical failure, not pace: Russell's power unit at Canada, Antonelli's engine at Barcelona, Antonelli's wheel shield at Britain, Russell's retirement at Belgium. No model predicts a part breaking from qualifying data.

## Scoring Probabilities, Not Picks

Winner hit rate cannot separate a good model from the grid. Both pick pole most weekends, and 16 races carries a standard error near 12 points. So from R16 every round is also scored on **log loss** (minus the log of the probability given to the actual winner) and **Brier score**. Both are lower-is-better and both punish a confident miss far harder than a hedged one. `probscore.py` computes them, `score_round` now writes `mc_log_loss`, `xgb_log_loss` and `pole_log_loss` into `config.json`, and R1-R16 are backfilled.

The pole baseline becomes a probability too: P(win | starting slot) from 91 races, 2022-2025. Pole wins 55%, P2 20%, P3 11%. Stored in `grid_prior.json`.

`prob_backtest.py` then tests a deliberately small model against it: a conditional logit (softmax across the field) on four inputs. Log of grid slot, qualifying gap to pole in seconds, and each of those scaled by a circuit overtaking index. The index is the mean grid-to-finish rank correlation at that circuit over earlier races from 2014, so a race never informs its own index.

**Backtest, 91 races, leave-one-race-out:**

| Model | Log loss | Brier | Winners | Avg P(winner) |
| --- | :---: | :---: | :---: | :---: |
| Grid prior (pole baseline) | 1.611 | 0.642 | 51/91 | 35.9% |
| Logit, grid only | 1.474 | 0.638 | 51/91 | 37.9% |
| Logit, grid + pole gap | 1.332 | 0.630 | 49/91 | 38.6% |
| Logit, all four inputs | 1.329 | 0.628 | 50/91 | 38.7% |

Half the gain over the table is smoothing: a back-row winner gets a small probability instead of near zero. The other half is the **pole gap**, worth 0.142 log loss with a 90% interval of 0.013 to 0.269. A 0.8s pole and a 0.02s pole do not carry the same odds, and the grid alone cannot tell them apart. The **overtaking index** adds 0.008. Consistent, but small. Winner count does not move at all, which is the point of scoring probabilities: the gain is in how much weight lands on the right driver.

The logit is calibrated. Across 1,818 driver-races, its 50-75% calls won 57% of the time and its 10-25% calls won 23%.

**Live 2026, out of sample.** The logit and the prior were fit on 2022-2025 only, so every 2026 race is new to them:

| Model | Log loss | Brier | Winners | Avg P(winner) |
| --- | :---: | :---: | :---: | :---: |
| Monte Carlo (published) | 1.470 | 0.657 | 9/16 | 29.9% |
| Grid prior | 1.326 | 0.518 | 11/16 | 39.4% |
| Logit | 0.958 | 0.432 | 11/16 | 49.3% |

The logit beats the grid by 0.368 log loss, 90% interval 0.152 to 0.719. Monte Carlo trails the grid by 0.144, inside noise. XGBoost scores 1.459 over its 13 rounds, level with Monte Carlo. Four inputs and no hand-set constants outperform eighteen features and 100,000 simulations, because the four carry the information and the eighteen mostly restate the grid.

The engine's own softmax, scored on 2024-2025 without the Monte Carlo layer, puts 12.5% on the eventual winner on average against the grid's 37.1%. The simulation sharpens that distribution in live use (Baku went from 33% to 90%), so this understates the published model, but the starting point is too flat.

```
python prob_backtest.py                 # backtest 2022-2025
python prob_backtest.py --live          # score 2026 rounds out of sample
python prob_backtest.py --index         # circuit overtaking index
python prob_backtest.py --write-prior   # regenerate grid_prior.json
python probscore.py --backfill          # add log loss to config.json history
```

**From R17 the logit runs live.** `engine.py` writes it into `prediction.json` under `logit`, beside Monte Carlo and XGBoost, and the dashboard shows it as a third card. It is committed before lights out like the other two, and `score_round` logs `logit_log_loss` and `logit_winner_correct`. Weights and the circuit table live in `logit_model.json`, fit on 2022-2025 and never refit mid-season, so every 2026 call stays out of sample.

```
python logit_model.py --train           # refit logit_model.json (once per season)
python logit_model.py 17_<race>         # logit prediction on its own
```
 
## What Changed in October 2026

A summary of the R17 update in plain terms. The sections above hold the full numbers.

### The problem

Winner hit rate could not show whether the model added anything. Always picking pole scored 11 of 16 this season. Monte Carlo scored 9. With 16 races, a two-race gap is noise. And hit rate ignores confidence: a 90% call and a 30% call on the same winner score the same.

### Change 1: every round is scored on probability

Two new numbers per round, both lower-is-better:

- **Log loss**: minus the log of the probability a model gave the actual winner. 50% on the winner scores 0.69. 10% scores 2.30. A confident miss costs far more than a hedged one.
- **Brier score**: the squared error across every driver's probability.

The pole baseline is scored the same way, as P(win | grid slot) from 91 past races. Pole wins 55%, P2 20%, P3 11%. That gives every model one bar on one scale.

### Change 2: a new model, the logit

A logit is logistic regression, one of the simplest models in statistics. This version is a conditional logit: it compares the drivers within one race.

1. Each driver gets a score from four inputs: log of grid slot, qualifying gap to pole in seconds, and each of those scaled by the circuit overtaking index.
2. A softmax turns the scores into win probabilities summing to 100%.
3. The four weights come from 91 races, 2022-2025, chosen to put the most probability on the drivers who actually won.

Worked example, R16 Sepang. Sepang sits near the average overtaking index, so the circuit terms are close to zero and two weights do the work: minus 1.02 per unit of log grid slot, minus 2.63 per second off pole.

| Driver | Grid | Gap | Score | Win probability |
| --- | :---: | :---: | :---: | :---: |
| Verstappen | P1 | 0.000s | 0.00 | 66% |
| Hamilton | P2 | 0.298s | -1.49 | 15% |

A score 1.49 lower means about 0.23 times the probability. The gap term is the information the grid lacks: Russell's 0.837s pole at Baku gets 88%, Gasly's 0.06s pole at Monza gets 48%.

### Change 3: the circuit overtaking index

For each circuit, the rank correlation between grid and finish across earlier races from 2014. Near 0.87 means the race finishes close to grid order (Monaco). Near 0.5 means positions change a lot (Las Vegas). A race never informs its own index. In the backtest it adds little, 0.008 log loss, so the pole gap carries the model.

### Results

| | Backtest, 91 races | Live 2026, 16 races |
| --- | :---: | :---: |
| Pole baseline | 1.611 | 1.326 |
| Monte Carlo | not run | 1.470 |
| Logit | 1.329 | 0.958 |

Log loss, lower is better. The 2026 column is fully out of sample: the logit never saw a 2026 race in training.

### Why four inputs beat eighteen

The backtest showed 15 of the 18 engine features change nothing, because they restate qualifying pace. The logit keeps the two pieces of information that matter, grid slot and pace margin, and learns their weights from results instead of setting them by hand. Four weights from 91 races leaves little room to overfit.

### Files added or changed

| File | What it does |
| --- | --- |
| `probscore.py` | Log loss, Brier score, the grid-slot prior, per-round scoring and the R1-R16 backfill |
| `prob_backtest.py` | 91-race backtest of grid vs logit vs engine, ablation, circuit index, live 2026 scoring |
| `logit_model.py` | Trains the logit once and predicts live races from `data.py` |
| `logit_model.json` | Saved weights, circuit index table, 2026 schedule |
| `grid_prior.json` | P(win \| grid slot) from 2022-2025 |
| `test_probscore.py` | Ten tests for the metrics, the index and the live model |
| `engine.py` | Writes the logit into `prediction.json` under `logit` |
| `fetch_race_data.py` | `score_round` logs log loss for Monte Carlo, XGBoost, logit and pole |
| `app_public.py` | Third dashboard card, LOGIT WINNER, from R17 |
| `config.json` | R1-R16 history backfilled with log loss and Brier fields |

### Race weekend workflow

Unchanged. The usual commands now include the logit:

```
python fetch_race_data.py 17_<race> --round 17          # build the race file
python engine.py 17_<race>                              # Monte Carlo + XGBoost + logit
python fetch_race_data.py 17_<race> --round 17 --result --score
```

Retrain the logit once before 2027 with `python logit_model.py --train`. Not mid-season, so the 2026 record stays out of sample.

### Limits

- The logit was trained on 2022-2025 cars. 2026 brought new regulations, so 16 live races is early evidence.
- It cannot see race pace, strategy, safety cars or reliability. Long-run practice pace is the next input to test.
- Winner hit rate does not improve. The gain is in how much probability lands on the right driver.
 
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

### Engine Fixes (R15)

Two bugs found in the post-R14 audit, both unambiguous, both fixed:

**DNF counted twice.** The simulation already sets a retired driver's
performance to -1 and excludes them from the finishers, so they cannot win that
run. Win probability was then discounted again by the same DNF rate. Teams with
higher hand-set rates, Red Bull, Audi, Aston Martin and Cadillac, paid for
retirement twice.

**Pole sitter excluded from a random boost.** A mid-race boost loop ran
`range(1, n)`. Index 0 is the pole sitter, so the one driver in clean air was
the only one who could never receive it. Off-by-one.

Predictions before R15 are not regenerated. `config.json` records the round
where the engine changed so the accuracy history is not silently two models.

 
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

### R16 Sepang: The Models Disagree, Which Makes This Race Worth Something

Written before the race, so the claims below are falsifiable rather than
retrofitted.

Verstappen took Red Bull's first pole of 2026 by 0.298s, after the team ran a
Ferrari-derived bargeboard at Sepang. Mercedes brought a major upgrade the same
weekend and had their worst qualifying of the season, P3 and P7 after penalties.

The two models split on the winner for the first time in several races:

| | Monte Carlo | XGBoost |
| --- | --- | --- |
| P1 | Max Verstappen 40.97% | Isack Hadjar, predicted position 2.71 |
| P2 | Lewis Hamilton 19.52% | Lewis Hamilton 3.24 |
| P3 | George Russell 5.15% | Max Verstappen 3.65 |

That is a clean test. If Hadjar finishes near the front, weighting measured pace
above grid position was right here. If he spends the race in traffic, the grid
slot was the better signal. Baku offered no such test, because both models and
the naive baseline made the same call.

**Result: Verstappen won. Hadjar finished fifth from eighth.** Monte Carlo
correct, XGBoost wrong. The recovery was real but modest, three places, and
nowhere near a win.

Both models hit two of three podium drivers. XGBoost named Hadjar, Hamilton and
Verstappen against an actual podium of Verstappen, Antonelli and Hamilton, so it
identified the right drivers and ordered them badly, which is the pattern already
recorded below.

**Monte Carlo's mean position error was 6.0, the worst of its season, and the
figure is an artefact.** The arithmetic is Verstappen 0, Hamilton 1, Russell 17,
divided by three. Russell was predicted third and is recorded at twentieth, but he
did not finish twentieth on pace. He retired after 49 laps of a rain-delayed race.
Across the two predicted drivers who actually finished, the error is 0.5, the best
of the season.

So the same defect has two symptoms in one race, and both are published. Scoring a
retirement as a finishing position teaches XGBoost that slow cars finish well, and
it reports Monte Carlo's best race as its worst. The section below traces it to its
source.

### Why XGBoost Picked Hadjar, and Why the Obvious Answer Is Wrong

The explanation written here before the race was that XGBoost read a fast car
starting eighth and predicted a recovery drive. That was wrong, and the model
itself says so.

Hadjar's feature vector is worse than Verstappen's on every feature that differs
between them, and identical on the rest. His `quali_pace` is 0.893 against
Verstappen's 1.0. Nothing in the input makes him look faster.

Swapping Verstappen's features to Hadjar's values one at a time, and measuring
the change in predicted finishing position:

| Feature | Change | Effect on predicted finish |
| --- | --- | :---: |
| `track_history` | 0.5 to 0.0, less history | 0.96 places better |
| `grid_win_rate` | 0.45 to 0.003, further back | 0.43 places better |
| `practice_pace` | 1.0 to 0.739, slower | 0.23 places better |
| `adaptability` | 0.75 to 0.25, less | 0.22 places better |
| `quali_pace` | 1.0 to 0.893, slower | 0.12 places better |
| `teammate_gap` | 0.714 to 0.286, worse | 3.84 places worse |

Five of six have inverted sign. The model has learned that a slower driver
starting further back finishes better. Hadjar was ranked first because he is
worse, not despite it.

**The cause is the target variable.** `build_training_data` uses the classified
finishing position as the label, and 172 of 330 training rows, 52.1%, belong to
drivers who did not finish. A retirement is recorded at its classified position,
so Verstappen's R12 retirement enters training as finishing position 22 with Red
Bull's pole-adjacent features attached. Round 1 alone contributes Piastri at 21
and Hulkenberg at 22, both retirements.

In the region of feature space where `quali_pace` is near 1.0 and `grid_win_rate`
is 0.45, the training set therefore holds both winners labelled 1 and retirements
labelled 16 to 22. At `max_depth` 3 on 330 rows the tree shades that bucket toward
its mixed mean, which lands worse than the bucket just behind it. The global
relationship is still correct, `quali_pace` against finishing position correlates
at -0.628, but the local behaviour at the front of the grid is inverted.

The label is answering two questions at once: where did you finish, and did you
finish. Over half the rows answer the second. Fixing that is on the roadmap and
goes through the 48-race backtest rather than a hand-tune.

The same substitution corrupts the scoring, not only the training. Mean position
error treats a classified position as a race outcome, so a retirement from the
front enters as a 17-place miss. Four races this season were decided by mechanical
failure, and every one of them inflated the error of whichever model had the
retiring driver highest. Any fix has to be applied to `score_round` as well as to
`build_training_data`, or the metric will keep punishing the model for parts
breaking.

**Confidence tracked the margin, not a label.** Baku produced 90.25% off a 0.837s
pole gap on a circuit tagged `street`, softmax temperature 0.07. Sepang produces
40.97% off 0.298s on `balanced` at 0.12. The write-up below argues the Baku number
was partly an artefact of a hand-typed label. This race is the counter-example
worth recording: when the gap is genuinely smaller, the distribution genuinely
flattens.

**A prediction that did not come true.** Before running this, the expectation
written down was that stale hand-set team priors would suppress Verstappen, the
way they held Gasly seventh off a shock pole at Monza. They did not.
`ENERGY_READINESS` still reads Red Bull 0.78 against Mercedes 0.88, and measured
`quali_pace` at 1.0 with a team pace deficit of 0.0 overcame it without trouble.
The Monza failure needed a team whose season-long race pace was also weak. Red
Bull sits fourth in the constructors' championship with five podiums in six
finishes, so its priors were never that far from the truth. No value was
hand-edited before this prediction.

**Two audit fixes fired on this one race.** Colapinto's fifteen-place drop targets
slot 30 and Lindblad's back-of-grid penalty targets slot 46, both past the end of
a 22-car grid. The R15 overflow fix places them at the back in penalised order,
and the rebuilt grid matches the official FIA classification on all 22 slots.
Separately, Gasly and Bortoleto were both slower in Q3 than in Q2. The `best_lap`
fix records their quicker Q2 times; the original code would have logged them
0.800s and 0.859s slow and inflated the Alpine and Audi team pace deficits by
close to a second each.

### R15 Baku: Right Winner, Wrong Everything Else

Russell took pole by 0.837s, the largest qualifying margin of 2026. Both models
called him: Monte Carlo 90.25%, XGBoost a predicted finishing position of 1.63
and a win probability of 78.2%. No other prediction this season went above 60%.

He won. The most confident call of the season landed.

Read the rest of the distribution before treating that as a win for the model.

| | Predicted | Finished |
| --- | --- | --- |
| Monte Carlo P1 | George Russell 90.25% | 1st |
| Monte Carlo P2 | Charles Leclerc 2.18% | 4th |
| Monte Carlo P3 | Oscar Piastri 1.3% | 13th |
| Actual P2 | Max Verstappen, ranked 4th at 1.03% | 2nd from P8 on the grid |
| Actual P3 | Isack Hadjar | 3rd from P4 |

Podium overlap 1 of 3. Mean position error across the predicted top three: 4.0,
the worst of the season. Antonelli started P16 after his Q1 crash and finished
5th. Verstappen gained six places. Piastri started 3rd and finished 13th.

So the model got the one name a coin-flip on the pole sitter would also have
got, and missed both remaining podium slots by wide margins. The always-pole
baseline scored this race too. A 90.25% call that agrees with the baseline adds
no information, whatever the outcome.

The confidence number itself is partly an artefact of a label typed by hand.

Softmax temperature is set per circuit type: `street` 0.07, `high_speed` 0.10,
`balanced` 0.12, `wet` 0.18. Baku is tagged `street`, the sharpest setting on the
list, which concentrates probability on whoever leads the score ordering. The
same driver, the same 0.837s, the same feature values under `balanced` produce a
materially flatter distribution, because temperature divides the score gaps
before the softmax. One word in `data.py` moves the headline number more than any
feature weight in the model does.

Baku deserves a low temperature for a defensible reason: passing is hard and
qualifying carries. That is not the problem. The problem is that four hand-typed
labels carry more influence over the published probability than the eighteen
features the project is built around, and none of the four has been validated
against a single race. The backtest harness can settle it by refitting
temperature per track type over 48 races instead of accepting the values someone
picked once. That is now on the roadmap.

One more thing worth recording. The penalty set here broke the grid builder:
Sainz 5 places, Perez 3 places, both Aston Martins to the back for power unit
components. The rebuild returned 21 cars for a 22-car grid and raised no error.
See Four Silent Data Bugs. The grid that shipped was verified against the
official classification after the race: all 22 slots match.

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

### Four Silent Data Bugs

None of these threw an exception. Each was found by reading a published timing sheet against the generated file, and each changed the prediction.

**A grid penalty deleted a driver.** Found at Baku, R15. Penalties apply to the qualifying position, so a drop can target a slot past the end of the grid. Perez qualified 20th and took three places, aiming at 23rd, with both Aston Martins already sent to the pit lane. The rebuild capped targets at the number of grid slots, so 23 fell outside the fill loop and Perez was written out of the grid entirely. The script returned 21 cars, printed no warning, and the engine ran on 21 drivers as though that were the entry list. Fixed by routing any overshooting penalty to the back of the non-pit-lane runners in penalised order, which is what the FIA does, and by a hard exit if the rebuilt grid does not contain every driver it started with. The exit guard matters more than the fix: the class of error is losing a driver, not this specific arithmetic.

**Driver name mismatch across sources.** FastF1 writes "Kimi Antonelli", the timing API writes "Andrea Kimi Antonelli". Same for Oliver against Ollie Bearman and Alexander against Alex Albon. The engine looks FP1 up by grid name, so three drivers fell to the `practice_pace` fallback of 0.3. Antonelli had topped the session at Zandvoort and was scored as if slowest: 4.57% instead of 9.42%, off the predicted podium. He finished second. Fixed by reconciling names against the grid by surname.

**Drivers who race without qualifying.** At Madrid the API returned 20 drivers, not 22. Bearman never left the garage after an FP3 crash and Stroll set no time, both cleared to race at the stewards' discretion. The qualifying classification only lists drivers who set a time, so both vanished from the grid entirely. Fixed with `--absent "Driver:Team"`, which places them behind the classified runners with `q_time` of None, which the engine already reads as neutral.

**Fastest lap read as latest session.** The original `best_lap` took Q3, then Q2, then Q1, assuming later sessions are quicker. Albon set 1:35.307 in Q1 and a slower 1:35.532 in Q2, so he was recorded two tenths off his real pace. Harmless at P16. The damaging case is a front-runner who banks a good Q2 lap and has Q3 ruined by a red flag, who would then be scored on the ruined lap. Fixed by taking the minimum across all three sessions.

The pattern matters more than any single bug. All four produced plausible numbers, none produced an error, and all four were caught by eye rather than by anything automated. That is the strongest argument in this repo for the backtest: 60-plus races surface distortions that 13 races hide.

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

## Backtest: 48 Races, 2024 and 2025

`backtest.py` replays both prior seasons through the same feature model, scored
leave-one-race-out so nothing is measured on data it trained on. It exists
because 14 races cannot tell you whether a change helped: a two-race swing is
noise, so every tuning decision up to R14 was a guess.

Run it with `python backtest.py`. First run takes a few minutes and caches to
`.backtest_cache/`; reruns are instant.

### The result that reframes the project

**The model picks the pole sitter in 45 of 48 races.** It disagrees three
times, and of those it is right once and wrong once.

| | winners |
| --- | :---: |
| Weighted-score model | 28/48 (58%) |
| Always pick pole | 28/48 (58%) |

Level, because 94% of the time they are the same prediction. Every feature in
the model is either derived from qualifying pace or fixed per team, so nothing
in it is capable of disagreeing with the grid. That is why a season of weight
calibration produced no improvement: there was no independent prediction to
improve.

### Feature ablation

Dropping each feature and re-measuring winner accuracy across all 48 races:

| Dropped | Winners | Change |
| --- | :---: | :---: |
| `fuel_quality` | 29/48 | +1 |
| `sprint_score` | 29/48 | +1 |
| `tyre_management` | 29/48 | +1 |
| 12 other features | 28/48 | 0 |
| `quali_pace` | 27/48 | -1 |
| `race_pace` | 27/48 | -1 |
| `grid_win_rate` | 27/48 | -1 |

Fifteen of eighteen features change nothing. Three are worth one race each.
Three make it marginally worse by being included. Dropping every hand-set team
constant at once gains a race and leaves MAE unchanged.

### XGBoost over 48 races

| | held-out MAE |
| --- | :---: |
| XGBoost | 3.26 positions |
| Predict finish = grid slot | 3.09 positions |

The same result as on the 2026 data, so it is not small-sample noise. A
gradient-boosted model on 18 features loses to a one-line heuristic.

### What this changes

Accuracy does not come from tuning. It comes from a feature that is independent
of qualifying pace, and the model currently has none. Candidates worth testing:
long-run practice pace, tyre strategy divergence, circuit-specific overtaking
rates, pit-lane time loss. Each can now be measured against 48 races in about
four minutes instead of one data point per fortnight.

## Tests

`python test_pipeline.py` or `python -m pytest test_pipeline.py -v`

Ten tests covering the bug classes that reached production this season. Each of those bugs produced plausible numbers, threw no exception, and was caught by eye weeks later.

| Test | Bug it encodes |
| --- | --- |
| `test_penalty_reordering_matches_published_grid` | Grid penalties applied by hand, verified against the FIA's Hungary grid |
| `test_penalty_overflow_keeps_every_driver` | A penalty targeting a slot past the end of the grid deleting a driver, Baku R15 |
| `test_penalty_on_unknown_driver_fails_loudly` | A misspelled driver name silently doing nothing |
| `test_no_grid_driver_is_shadowed_by_a_name_variant` | FastF1 and the API spelling three drivers differently, dropping them to the `practice_pace` fallback |
| `test_name_reconciliation_maps_known_variants` | The mapping itself |
| `test_name_reconciliation_rejects_non_starters` | Reserve drivers in FP1 must not match a grid entry |
| `test_race_files_are_structurally_sound` | Duplicate grid slots, missing pace deficits, absent `r1_finish` |
| `test_results_match_grid_names` | Spelling drift silently shrinking the XGBoost training set |
| `test_fastest_lap_is_the_fastest_not_the_latest` | Taking the latest session's lap rather than the quickest |
| `test_predictions_are_never_regenerated` | The project's core claim, that published predictions are never edited |

Each data test was validated by reintroducing the original bug and confirming it fires.

`python test_probscore.py` adds ten more for the probability work: log loss and Brier against hand-computed values, the zero-probability floor, the grid prior, short-name matching for early-season files, the overtaking index never reading the race it scores, and the saved logit model placing R16 at Sepang.

Files written before the API pipeline carry known gaps, such as missing `r1_finish`. Those predictions are published and are not regenerated, so the structural checks apply from `10_belgium` onward. That boundary is a constant at the top of the file.

## Project Structure
 
```
F1-predictions/
├── engine.py              Monte Carlo + XGBoost + self-calibration
├── app.py                 Local dashboard, runs predictions
├── app_public.py          Public read-only dashboard, deployed to Streamlit Cloud
├── fetch_race_data.py     Timing API pipeline: grid, sprint, penalties, results, scoring
├── backtest.py            Replays 2024-2025, leave-one-race-out scoring and feature ablation
├── prob_backtest.py       Probability backtest 2022-2025, logit, overtaking index, live scoring
├── probscore.py           Log loss, Brier score, grid-slot prior, round scoring
├── logit_model.py         Live logit: train once, predict each race
├── logit_model.json       Logit weights, circuit index table, 2026 schedule
├── grid_prior.json        P(win | grid slot), 2022-2025
├── test_pipeline.py       Ten tests covering the bug classes that reached production
├── test_probscore.py      Ten tests for the probability scoring and the logit
├── config.json            Feature weights, accuracy history, regulation params
├── requirements.txt
└── races/
    ├── 01_australia/ ... 16_malaysia/
    │   ├── data.py         Race inputs
    │   ├── prediction.json Locked before the race
    │   └── result.json     Actual outcome
```
 
## Tech Stack
 
Python 3.12, NumPy, XGBoost, scikit-learn, FastF1, Streamlit, Plotly. Race data from the Ergast-compatible API at api.jolpi.ca.
 
Deployed free on Streamlit Community Cloud. Every push to main rebuilds the live dashboard automatically.
 
## Roadmap
 
- **Fix the engine bugs from the audit**: double-counted DNF, the recovery bonus that rewards starting further back, the pole sitter excluded from the random boost, and the reliability feature. These change future predictions only; published predictions are never regenerated
- **Extend test coverage to the engine**. `test_pipeline.py` covers the data layer. The simulation itself has none, and both R15 engine fixes were bugs a test would have caught
- **Long-run race pace from FP2 and FP3**: fuel-corrected stint averages via FastF1, added as a fifth logit input and measured against the 91-race backtest. The strongest candidate for information the grid does not hold
- **Find a feature independent of qualifying pace**. This is now the whole problem. The backtest shows the model reproduces the grid in 94% of races because every feature is either derived from qualifying or fixed per team. Candidates: long-run practice pace, tyre strategy divergence, circuit overtaking rates, pit-lane time loss. Each is testable against 48 races in minutes
- **Cut the dead features**. Fifteen of eighteen change nothing across 48 races, and three make results marginally worse. Removing them costs no accuracy and makes the remainder interpretable
- **Settle the recovery term and the reliability feature with the harness**. Both are known-wrong but their replacements are design choices, and the backtest can now measure which version is better rather than leaving it to opinion
- **Stop training XGBoost on retirements as if they were finishing positions**: the highest-value item on this list, and the only one with a measured cost. 172 of 330 training rows, 52.1%, are drivers who did not finish, labelled at their classified position. That inverts the sign on five of six features at the front of the grid, which is why XGBoost picked a driver whose every feature was worse than the pole sitter's at R16. Three candidate fixes to test over the 48 backtest races: drop non-finishers, which halves the data; train position-given-finish and multiply by a separate reliability model; or treat retirements as right-censored rather than as positions. Whichever wins must also be applied to `score_round`, because mean position error has the same flaw: Russell's R16 retirement entered as a 17-place miss and turned Monte Carlo's best race of the season into its worst
- **DNF cause split**: separate driver-caused DNFs from mechanical failures so pace scores are not penalized for parts breaking
- **Track-dependent grid weighting**: `grid_win_rate` carries the same 0.0717 weight at Monaco and Monza. At R13 this let a P22 start outrank the pole sitter. Circuits where overtaking is rare should weight starting position far higher, the way softmax temperature already varies by track type
- **Ensemble layer**: across recent races XGBoost identifies podium drivers while ordering them wrong, and Monte Carlo orders better than it selects. Let XGBoost pick the podium set and Monte Carlo rank it
- **Refresh hand-set team constants**: `ENERGY_READINESS`, `START_PROCEDURE`, `tyre_management` and `circuit_fit` are set by hand and rarely revisited. At R13 they held Alpine down while measured pace put the car on pole. Priors should decay toward measured performance as the season provides evidence
- **Practice-pace fallback**: a driver missing from `FP1_TIMES` scores 0.3, a low value, so sitting out a session for a rookie run reads as slowness. The median of drivers who did run would treat absence as no information instead of bad information
- **Fit softmax temperature instead of typing it**: the four per-track-type values, street 0.07 through wet 0.18, were set by hand and never tested. At Baku the label alone drives the headline confidence more than any feature weight does. Refit them over the 48 backtest races and report the held-out log loss for each candidate
- **Dead XGBoost features**: at R12 the model assigned `race_pace` and `tyre_management` zero importance, while `tyre_compound_fit` and `energy_score` together carried 51%. With 235 rows at max_depth 3, that concentration needs investigating before more features are added
R16 Bahrain GP in Malaysia scored. Every prediction is committed before lights out and unedited. Backtest covers 2024 and 2025, 48 races.
 
---
Built by [Brinda Bhanderi](https://www.linkedin.com/in/brindabhanderi/). Inspired by [Mariana Antaya](https://www.linkedin.com/in/marianaantaya/).