# Season Notes, 2026

Detailed write-ups moved out of the README at R17. Code these notes mention, `backtest.py` and XGBoost v1 (`xgboost_predict` in `engine.py`), was removed at R17 and remains in git history. The numbers below are as published at the time.

One correction applies throughout: XGBoost v1 retirement rows were 71 of 330 (21.5%), not 172 (52.1%). The first count included 101 lapped finishers.

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

**A suspected cause was the target variable.** `build_training_data` uses the classified
finishing position as the label, and 71 of 330 training rows, 21.5%, belong to
drivers who retired. (This section first said 172 rows, 52.1%. That count included
101 lapped drivers, who finish the race with a valid position. Corrected R17.)
A retirement is recorded at its classified position,
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
finish. One row in five answers the second.

**Dropping retirements does not fix it.** Tested at R17, leave-one-race-out over
rounds 4-16: removing the 71 retirement rows moved XGBoost's winner log loss from
1.951 to 2.010, slightly worse. Cutting to 100 trees at depth 2 gave 1.684. Keeping
only `quali_pace` and `grid_win_rate` gave 1.384. The inversion is mainly
overfitting: 300 trees on 352 rows learn noise in features that barely vary at the
front of the grid. At R17 the sprint win counted against Verstappen by 0.88
places, and a `tyre_compound_fit` of 0.945 hurt him while 0.940 helped Leclerc.
His predicted finish from pole was P10.7.

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
 

## Data Pipeline Audit (R14)

A full pass over the repo after R14 found fourteen issues. The four that changed published numbers:

**Hand-entered results.** R1-R9 were typed by hand and the midfield was 4-7 places out. Winners were correct throughout, so headline accuracy never moved, but XGBoost had been training on scrambled labels for nine of fourteen races, and `r1_finish` for each following race read from them. All results now come from the API and are verified against it.

**In-sample MAE published as if it were prediction error.** See XGBoost Performance above.

**DNF counted twice.** The simulation already prevents a retired driver from winning, then win probability is discounted again by the same DNF rate. Teams with higher hand-set rates are penalised twice over.

**Reliability is not reliability.** The feature reads the previous race finishing position: top ten scores 0.95, anything lower 0.80, missing 0.50. Finishing eleventh on pace counts as unreliable, a crash that was not the driver's fault counts as unreliable, and a driver absent from the previous race takes the heaviest penalty of all.

Also found: the pole sitter is excluded from a random boost every other driver can receive, because the loop starts at index 1; `practice_pace` defaults to 0.3 for a missing FP1 time, so skipping a session reads as being slow; DNFs are labelled two different ways in the training set; and the hand-set team constants have never been revisited, which is why Alpine was held down at Monza while measured pace put them on pole.

## Four Silent Data Bugs

None of these threw an exception. Each was found by reading a published timing sheet against the generated file, and each changed the prediction.

**A grid penalty deleted a driver.** Found at Baku, R15. Penalties apply to the qualifying position, so a drop can target a slot past the end of the grid. Perez qualified 20th and took three places, aiming at 23rd, with both Aston Martins already sent to the pit lane. The rebuild capped targets at the number of grid slots, so 23 fell outside the fill loop and Perez was written out of the grid entirely. The script returned 21 cars, printed no warning, and the engine ran on 21 drivers as though that were the entry list. Fixed by routing any overshooting penalty to the back of the non-pit-lane runners in penalised order, which is what the FIA does, and by a hard exit if the rebuilt grid does not contain every driver it started with. The exit guard matters more than the fix: the class of error is losing a driver, not this specific arithmetic.

**Driver name mismatch across sources.** FastF1 writes "Kimi Antonelli", the timing API writes "Andrea Kimi Antonelli". Same for Oliver against Ollie Bearman and Alexander against Alex Albon. The engine looks FP1 up by grid name, so three drivers fell to the `practice_pace` fallback of 0.3. Antonelli had topped the session at Zandvoort and was scored as if slowest: 4.57% instead of 9.42%, off the predicted podium. He finished second. Fixed by reconciling names against the grid by surname.

**Drivers who race without qualifying.** At Madrid the API returned 20 drivers, not 22. Bearman never left the garage after an FP3 crash and Stroll set no time, both cleared to race at the stewards' discretion. The qualifying classification only lists drivers who set a time, so both vanished from the grid entirely. Fixed with `--absent "Driver:Team"`, which places them behind the classified runners with `q_time` of None, which the engine already reads as neutral.

**Fastest lap read as latest session.** The original `best_lap` took Q3, then Q2, then Q1, assuming later sessions are quicker. Albon set 1:35.307 in Q1 and a slower 1:35.532 in Q2, so he was recorded two tenths off his real pace. Harmless at P16. The damaging case is a front-runner who banks a good Q2 lap and has Q3 ruined by a red flag, who would then be scored on the ruined lap. Fixed by taking the minimum across all three sessions.

The pattern matters more than any single bug. All four produced plausible numbers, none produced an error, and all four were caught by eye rather than by anything automated. That is the strongest argument in this repo for the backtest: 60-plus races surface distortions that 13 races hide.

The deeper issue lives in the engine, not the fetcher: `practice_pace` defaults to 0.3, a low value, so a driver who did not run reads as a driver who was slow. Those are different things. Moving the default to the median of drivers who did run is on the roadmap.

Hand-edited per race: weather forecast, circuit type, tyre compounds, circuit history. These carry over from the previous `data.py`, so a re-fetch no longer wipes tuning.


## XGBoost v1 Performance, R4-R16

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

