# F1 2026 Race Winner Prediction

I built a model that predicts who wins each Formula 1 race. It uses qualifying data, practice times, sprint results, tyre strategy, and 100,000 Monte Carlo simulations. After each race, it compares what it predicted vs what happened and adjusts its own weights. It started with hand-tuned guesses for Race 1. By mid-season it should be running on learned data.

The whole thing runs in one command: `streamlit run app.py`

## Does it work?

So far, yes.

| Race | Predicted | Win% | Actual | Result |
|------|-----------|------|--------|--------|
| R1 Australia | George Russell | 59.1% | George Russell won by 2.9s | Correct |
| R2 China | Lewis Hamilton | 36.2% | Race is today (Mar 15) | TBD |

Russell won from pole, led a Mercedes 1-2, and the model had him as a clear favorite. 2 out of 3 podium finishers were in the top 5 predictions. 27% of the grid DNS/DNF'd, which matched the elevated failure rates I'd built in for new engine partnerships.

Not everything was right. Verstappen went from P20 to P6 and the model gave him 0.56%. Ferrari's VSC strategy mistake cost Leclerc the win, and the model had no way to simulate that. Both of those gaps led to the v5 update.

## What the model actually does

For each race, it goes through three steps.

**Features.** It builds 18 numbers for every driver on the grid. Each one is between 0 and 1. They cover car speed (qualifying gaps, practice times, race pace), driver skill (teammate delta, sprint result, experience), race-day factors (start performance, reliability, energy management), tyre and pit strategy (tyre degradation handling, pit crew speed, compound suitability), and 2026-specific stuff (sustainable fuel quality, dirty air following, circuit type match, track temperature sensitivity).

**Scoring.** Multiply each feature by its weight, add them up, run softmax to get a win probability. The temperature parameter (currently 0.08) controls how confident the model is. Lower = more decisive. Higher = more spread out.

**Simulation.** Run the race 100,000 times. Each simulation randomly throws in events that happen in real F1: safety cars, rain, mechanical failures, bad pit stops, tyre degradation, energy management mistakes, lap 1 crashes, driver errors. After 100K runs, count how many times each driver won. That's the win percentage.

## Why these 18 features

I started with 10 for Australia. After the race exposed gaps (no tyre strategy, no pit execution, no circuit matching), I added 8 more for China.

| Feature | Weight | Why it matters |
|---------|--------|---------------|
| quali_pace | 14% | Gap to pole in seconds. The single strongest predictor of race pace. |
| race_pace | 10% | Team deficit from practice long runs. Tells you who has the faster car over a stint. |
| energy_score | 10% | The defining challenge of 2026. 350kW MGU-K means half the car's power is electric. Mismanage the battery and you're a sitting duck on straights. FIA is already reviewing the rules because it dominates too much. |
| tyre_management | 8% | How well the team preserves tyre life. Research shows tyre degradation is more predictive than qualifying for race outcomes. |
| sprint_score | 7% | Sprint finishing position. It's a real mini-race with real data, not a simulation. |
| grid_win_rate | 5% | Historical win rate from each grid slot. Adjusted down for 2026 because active aero makes overtaking easier. |
| practice_pace | 5% | FP1 times. Noisy signal (teams run different programs) but still data. |
| reliability | 5% | Did they finish last race? New regs = new failure modes. If something broke in Australia, it might not be fixed. |
| start_score | 4% | 2026 start procedure is completely new. Ferrari nailed it in testing. Leclerc jumped P4 to P1 at the start in Melbourne. |
| teammate_gap | 4% | Qualifying delta to teammate. Isolates driver skill from car performance. |
| circuit_fit | 4% | Some cars suit some tracks. Mercedes is strong on high-speed circuits. Ferrari is better on street tracks. |
| fuel_quality | 4% | Sustainable fuel is mandatory for the first time. Performance varies by supplier. Mercedes/Petronas is ahead. Cadillac is behind. |
| pit_execution | 4% | Pit crew speed. McLaren stops in 2.2s. Cadillac takes 3.1s. That gap costs positions. |
| track_history | 4% | Hamilton has 6 wins at Shanghai. That's not a coincidence. |
| adaptability | 3% | How many regulation changes the driver has survived. Hamilton has been through 2009, 2014, 2017, 2022, and now 2026. |
| dirty_air | 3% | Following another car used to cost 30% downforce. In 2026 it's only 10%. Huge change. |
| tyre_compound_fit | 3% | Harder compounds favor teams with better tyre management. Softer compounds favor raw speed. |
| track_temp | 3% | Hot track = more tyre degradation = teams with good tyre skills gain. Cool track = less deg = speed wins. |

## 2026 regulations baked into the model

2026 is the biggest rule change in F1 history. The model accounts for all of it.

Active aero replaced DRS. Every driver gets low-drag mode on straights now, not just cars within 1 second. I dropped the pole win rate from 60% to 45% because overtaking is way easier. Australia confirmed this: Leclerc led from P4 for multiple laps.

The power unit is completely different. MGU-H is gone. MGU-K went from 120kW to 350kW. Power is now split roughly 50/50 between petrol and electric. Verstappen called it "Formula E on steroids." Russell said his battery had "nothing in the tank" at the Melbourne start. I model this as energy noise (0.06 per driver per sim) with extra variance for new teams.

Cars are 76kg lighter and smaller. Wheelbase down 200mm, width down 100mm. The "Nimble Car Concept." Lighter cars are more sensitive to setup and fuel load, so I added weight variance to the simulation.

Sustainable fuel is mandatory. Different suppliers have different performance. I gave each team a fuel quality score based on pre-season testing reports.

Tyres are smaller (front -25mm, rear -30mm). Different degradation profiles. I added tyre management as an 8% weighted feature and simulate degradation penalties in every Monte Carlo run.

New engine partnerships (Cadillac, Audi, Aston Martin with Honda, Red Bull with Ford) have higher DNF rates. 27% of the grid DNS/DNF'd in Australia. The model's elevated failure rates were right.

## What changed after Australia

Australia was the first race of the 2026 era. The model (v3) got the winner right but missed several things.

Ferrari didn't pit under the VSC. That cost Leclerc the win. The model had no pit strategy simulation. Fixed in v5: every Monte Carlo run now simulates 1-stop vs 2-stop decisions, pit crew speed differences, and bad pit stop probability.

Verstappen recovered from P20 to P6. The model gave him 0.56% win probability. Active aero helped him gain 14 positions. The overtake boost was too conservative. Adjusted in v5.

Tyre degradation determined strategy. The model treated it as random noise. Now tyre_management is an 8% weighted feature and the simulation applies team-specific degradation penalties every run.

Track temperature affected tyre behavior. Not modeled before. Now track_temp is a feature and the simulation adjusts variance based on surface temperature.

Feature count went from 13 to 18. Softmax temperature went from 0.14 to 0.08 (more decisive predictions).

## How self-calibration works

After each race, I enter the top 10 finishing order. The model compares predicted rankings vs actual positions.

If it ranked a driver too high (predicted P2, finished P8), it looks at which features scored high for that driver and decreases their weights. If it ranked someone too low, it increases the weights.

The learning rate starts at 0.05 and decays by 30% after each race. This means early races cause bigger weight shifts. By Race 5, adjustments are smaller and more precise. By Race 10, the weights are stable and data-driven.

It's gradient descent on 18 feature weights, with one training example per race. Not a lot of data, but enough to correct the worst hand-tuning mistakes.

## China GP prediction

Hamilton to win from P3 with 36.2% probability.

He has 6 wins at Shanghai. Ferrari's start advantage was confirmed in Australia. He finished P3 in the sprint. 18 seasons of experience in rain and changing conditions. Shanghai's 1km+ back straight suits aero-efficient cars, but Hamilton's track history overrides the circuit mismatch.

Russell (P2, 18.6%) and Antonelli (P1, 11.9%) are the main threats. Antonelli is on pole but only has one season of F1 experience and got a 10-second penalty in the sprint. The model penalizes him for that.

## Project structure

```
F1-predictions/
    app.py              Dashboard (one page, dark theme)
    engine.py           Prediction engine (18 features, 100K sims)
    config.json         Weights + accuracy history
    races/
        01_australia/   Data + prediction + result
        02_china/       Data + prediction
    archive/            Old model versions (v1-v3)
    .streamlit/         Theme config
    requirements.txt
```

## Running it

```
git clone https://github.com/brinda0301/F1-predictions.git
cd F1-predictions
python -m venv venv
.\venv\Scripts\Activate.ps1       # Windows
pip install -r requirements.txt
streamlit run app.py
```

Or from the terminal:

```
python engine.py 02_china
```

## Adding a new race

Create `races/03_japan/data.py` with qualifying grid, practice times, sprint results, tyre compounds, circuit info, and weather. Run the prediction. After the race, submit the result. Model calibrates itself.

## Tech stack

Python, NumPy, Streamlit, Plotly, FastF1

## License

MIT
