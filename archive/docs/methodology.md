# Methodology

## Problem Statement

Predict the most likely winner of the 2026 F1 Australian Grand Prix at Albert Park, Melbourne.

## Approach

The model uses a weighted ensemble scoring system combined with Monte Carlo simulation.

### Feature Engineering

Ten features are extracted for each of the 22 drivers on the grid:

**Grid Position Score (25%)**
Qualifying position is the strongest single predictor of race outcome at Albert Park. The pole sitter has won approximately 40% of Australian GPs at this circuit since 1996. Score is linearly scaled from 1.0 (pole) to 0.0 (P22).

**Market Odds Signal (20%)**
Betting markets aggregate information from thousands of informed participants. Post-qualifying implied probabilities from major sportsbooks (Caesars, DraftKings, BetMGM) are used directly as a feature. This captures signals the model might miss.

**Team Strength (15%)**
Derived from pre-season testing long-run pace, qualifying simulation times, and practice session performance. Normalized to a 0-1 scale where 1.0 represents the strongest package.

**Practice Pace (10%)**
Average ranking across FP1, FP2, and FP3 sessions, normalized. Practice times correlate with race pace, though teams run different fuel loads and tire compounds.

**Experience (7%)**
Career wins, podiums, poles, and seasons in F1 contribute to a composite experience score. Experienced drivers handle pressure, strategy calls, and variable conditions better.

**Reliability (7%)**
Penalizes drivers who had mechanical issues or crashes during the weekend. A driver who failed to set a qualifying time receives a significant penalty (0.3 vs 1.0 for a clean weekend).

**Track Knowledge (5%)**
Historical performance at Albert Park specifically. Drivers who have won or stood on the podium at this circuit receive a boost.

**Consistency (5%)**
Low variance across practice sessions indicates a well-understood car setup. Measured as 1 minus the normalized standard deviation of FP1/FP2/FP3 rankings.

**Teammate Differential (3%)**
How much better a driver qualified vs their teammate. Isolates driver skill from car performance.

**Qualifying Improvement (3%)**
The delta between average practice position and qualifying position. Drivers who improve in qualifying show adaptability.

### Model Scoring

Features are combined via weighted sum to produce a raw score for each driver. Raw scores pass through a softmax function (temperature = 0.15) to generate win probabilities. The low temperature creates a peaked distribution that reflects the reality that pole position at Albert Park carries a major advantage.

### Monte Carlo Simulation

50,000 race simulations are run. Each simulation applies random perturbations:

**Safety Car (60% per race)**
At Albert Park, safety car deployments are frequent due to narrow escape roads and walls close to the racing line. Safety cars compress the field and reduce front-runner advantage.

**Rain (15%)**
Melbourne's autumn weather is variable. Rain amplifies the importance of driver skill, giving experienced drivers an edge.

**DNF (8% per driver)**
Mechanical failures, collisions, and driver errors. Each driver has an independent probability of retirement.

**First-Lap Chaos (25%)**
The tight Turn 1 at Albert Park frequently produces incidents on the opening lap. A random driver in the top 6 may lose significant positions.

### Output

For each driver:
- **Win%**: Percentage of simulations where the driver finishes P1
- **Podium%**: Percentage finishing in top 3
- **Points%**: Percentage finishing in top 10

## Limitations

1. 2026 regulations are entirely new. Historical patterns may not transfer cleanly.
2. Practice session data reflects different fuel loads and tire strategies.
3. The model does not account for pit stop strategy or tire degradation rates.
4. Team strength estimates from testing carry uncertainty.
5. Betting odds already encode much of the other information, creating partial feature correlation.

## Future Improvements

- Add telemetry data (sector times, speed trap data) as features
- Train on full historical dataset using gradient boosting (XGBoost/LightGBM)
- Incorporate tire strategy simulation
- Add weather forecast granularity (wind, temperature, humidity)
- Build a live updating pipeline for race-day predictions
