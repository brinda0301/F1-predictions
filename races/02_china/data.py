"""
2026 Chinese Grand Prix - Race Data
Shanghai International Circuit
Race: Sunday March 15, 2026 | 56 laps | 305km

Sprint weekend format: FP1 only (no FP2/FP3)
Sprint race result used as additional race pace data.

Data sources:
- Qualifying: formula1.com, the-race.com (March 14 2026)
- FP1: formula1.com, crash.net (March 13 2026)
- Sprint race: racingnews365.com (March 14 2026)
- Sprint qualifying: racingnews365.com (March 13 2026)
"""

RACE_INFO = {
    "round": 2,
    "name": "Chinese Grand Prix",
    "circuit": "Shanghai International Circuit",
    "city": "Shanghai",
    "date": "2026-03-15",
    "laps": 56,
    "distance_km": 305,
    "format": "sprint_weekend",
    "longest_straight_km": 1.2,
}

# Qualifying results (March 14 2026)
GRID = [
    {"pos": 1,  "driver": "Kimi Antonelli",      "team": "Mercedes",      "q_time": 92.064},
    {"pos": 2,  "driver": "George Russell",       "team": "Mercedes",      "q_time": 92.286},
    {"pos": 3,  "driver": "Lewis Hamilton",       "team": "Ferrari",       "q_time": 92.415},
    {"pos": 4,  "driver": "Charles Leclerc",      "team": "Ferrari",       "q_time": 92.428},
    {"pos": 5,  "driver": "Oscar Piastri",        "team": "McLaren",       "q_time": 92.550},
    {"pos": 6,  "driver": "Lando Norris",         "team": "McLaren",       "q_time": 92.608},
    {"pos": 7,  "driver": "Pierre Gasly",         "team": "Alpine",        "q_time": 92.873},
    {"pos": 8,  "driver": "Max Verstappen",       "team": "Red Bull",      "q_time": 93.002},
    {"pos": 9,  "driver": "Isack Hadjar",         "team": "Red Bull",      "q_time": 93.121},
    {"pos": 10, "driver": "Oliver Bearman",       "team": "Haas",          "q_time": 93.292},
    {"pos": 11, "driver": "Nico Hulkenberg",      "team": "Audi",          "q_time": 93.350},
    {"pos": 12, "driver": "Franco Colapinto",     "team": "Alpine",        "q_time": 93.355},
    {"pos": 13, "driver": "Esteban Ocon",         "team": "Haas",          "q_time": 93.550},
    {"pos": 14, "driver": "Liam Lawson",          "team": "Racing Bulls",  "q_time": 93.780},
    {"pos": 15, "driver": "Arvid Lindblad",       "team": "Racing Bulls",  "q_time": 93.800},
    {"pos": 16, "driver": "Gabriel Bortoleto",    "team": "Audi",          "q_time": 93.980},
    {"pos": 17, "driver": "Carlos Sainz",         "team": "Williams",      "q_time": 94.200},
    {"pos": 18, "driver": "Alex Albon",           "team": "Williams",      "q_time": 94.700},
    {"pos": 19, "driver": "Fernando Alonso",      "team": "Aston Martin",  "q_time": 95.130},
    {"pos": 20, "driver": "Valtteri Bottas",      "team": "Cadillac",      "q_time": 95.360},
    {"pos": 21, "driver": "Lance Stroll",         "team": "Aston Martin",  "q_time": 95.920},
    {"pos": 22, "driver": "Sergio Perez",         "team": "Cadillac",      "q_time": 96.830},
]

# FP1 results (only practice session, sprint weekend)
FP1_TIMES = {
    "George Russell":      92.741,
    "Kimi Antonelli":      92.861,
    "Lando Norris":        93.296,
    "Oscar Piastri":       93.350,
    "Charles Leclerc":     93.400,
    "Lewis Hamilton":      93.500,
    "Oliver Bearman":      93.600,
    "Max Verstappen":      93.700,
    "Nico Hulkenberg":     93.750,
    "Pierre Gasly":        93.800,
    "Liam Lawson":         93.900,
    "Gabriel Bortoleto":   93.950,
    "Isack Hadjar":        94.000,
    "Esteban Ocon":        94.100,
    "Franco Colapinto":    94.200,
    "Alex Albon":          94.300,
    "Carlos Sainz":        94.500,
    "Fernando Alonso":     94.800,
    "Sergio Perez":        95.000,
    "Valtteri Bottas":     95.200,
    "Lance Stroll":        95.500,
    "Arvid Lindblad":      None,  # Retired early, smoke from car
}

# Sprint race result (used as race pace indicator)
SPRINT_RESULT = [
    {"pos": 1,  "driver": "George Russell",   "team": "Mercedes"},
    {"pos": 2,  "driver": "Charles Leclerc",  "team": "Ferrari"},
    {"pos": 3,  "driver": "Lewis Hamilton",   "team": "Ferrari"},
    {"pos": 4,  "driver": "Lando Norris",     "team": "McLaren"},
    {"pos": 5,  "driver": "Kimi Antonelli",   "team": "Mercedes"},  # Had 10s penalty, recovered
    {"pos": 6,  "driver": "Oscar Piastri",    "team": "McLaren"},
    {"pos": 7,  "driver": "Pierre Gasly",     "team": "Alpine"},
    {"pos": 8,  "driver": "Oliver Bearman",   "team": "Haas"},
    {"pos": 9,  "driver": "Max Verstappen",   "team": "Red Bull"},  # Fell back then recovered
    {"pos": 10, "driver": "Isack Hadjar",     "team": "Red Bull"},
]

# Driver experience (carried from Australia, updated with R1 results)
DRIVER_EXPERIENCE = {
    "George Russell":     {"f1_seasons": 7,  "career_poles": 5,  "r1_finish": 1},
    "Kimi Antonelli":     {"f1_seasons": 1,  "career_poles": 1,  "r1_finish": 2},
    "Isack Hadjar":       {"f1_seasons": 0,  "career_poles": 0,  "r1_finish": None},  # DNF
    "Charles Leclerc":    {"f1_seasons": 7,  "career_poles": 26, "r1_finish": 3},
    "Oscar Piastri":      {"f1_seasons": 3,  "career_poles": 2,  "r1_finish": None},  # DNS
    "Lando Norris":       {"f1_seasons": 6,  "career_poles": 8,  "r1_finish": 5},
    "Lewis Hamilton":     {"f1_seasons": 18, "career_poles": 104, "r1_finish": 4},
    "Liam Lawson":        {"f1_seasons": 1,  "career_poles": 0,  "r1_finish": 13},
    "Arvid Lindblad":     {"f1_seasons": 0,  "career_poles": 0,  "r1_finish": 8},
    "Gabriel Bortoleto":  {"f1_seasons": 0,  "career_poles": 0,  "r1_finish": 9},
    "Nico Hulkenberg":    {"f1_seasons": 14, "career_poles": 1,  "r1_finish": None},  # DNS
    "Oliver Bearman":     {"f1_seasons": 1,  "career_poles": 0,  "r1_finish": 7},
    "Esteban Ocon":       {"f1_seasons": 8,  "career_poles": 0,  "r1_finish": 11},
    "Pierre Gasly":       {"f1_seasons": 8,  "career_poles": 0,  "r1_finish": 10},
    "Alex Albon":         {"f1_seasons": 5,  "career_poles": 0,  "r1_finish": 12},
    "Franco Colapinto":   {"f1_seasons": 1,  "career_poles": 0,  "r1_finish": 14},
    "Fernando Alonso":    {"f1_seasons": 22, "career_poles": 22, "r1_finish": 16},
    "Sergio Perez":       {"f1_seasons": 14, "career_poles": 3,  "r1_finish": 15},
    "Valtteri Bottas":    {"f1_seasons": 13, "career_poles": 20, "r1_finish": None},  # DNF
    "Max Verstappen":     {"f1_seasons": 10, "career_poles": 40, "r1_finish": 6},
    "Carlos Sainz":       {"f1_seasons": 10, "career_poles": 6,  "r1_finish": None},  # Not classified
    "Lance Stroll":       {"f1_seasons": 8,  "career_poles": 1,  "r1_finish": None},  # Not classified
}

# Shanghai-specific data
# 1.2km back straight = active aero matters a LOT here
# Hamilton has 6 wins at Shanghai (most of any driver)
CIRCUIT_HISTORY = {
    "Lewis Hamilton":     {"wins": 6, "podiums": 8},
    "Fernando Alonso":    {"wins": 2, "podiums": 3},
    "Max Verstappen":     {"wins": 1, "podiums": 2},
    "Oscar Piastri":      {"wins": 1, "podiums": 1},
    "Charles Leclerc":    {"wins": 0, "podiums": 1},
    "Valtteri Bottas":    {"wins": 0, "podiums": 3},
    "George Russell":     {"wins": 0, "podiums": 1},
}

# Team pace from FP1 + Sprint (gap to fastest in seconds)
TEAM_PACE_DEFICIT = {
    "Mercedes":      0.000,
    "Ferrari":       0.300,
    "McLaren":       0.400,
    "Red Bull":      0.700,
    "Alpine":        0.800,
    "Haas":          0.900,
    "Audi":          1.000,
    "Racing Bulls":  1.100,
    "Williams":      1.500,
    "Aston Martin":  2.000,
    "Cadillac":      2.500,
}

# Start procedure performance from R1 (Australia)
# Ferrari gained at the start, Mercedes lost
START_PROCEDURE = {
    "Ferrari":       0.4,
    "Mercedes":     -0.1,
    "McLaren":       0.1,
    "Red Bull":     -0.1,
    "Alpine":        0.0,
    "Haas":          0.0,
    "Audi":         -0.1,
    "Racing Bulls":  0.0,
    "Williams":     -0.1,
    "Aston Martin": -0.2,
    "Cadillac":     -0.3,
}

# Energy readiness (updated after R1)
ENERGY_READINESS = {
    "Mercedes":      0.95,
    "Ferrari":       0.90,
    "McLaren":       0.85,
    "Red Bull":      0.70,
    "Alpine":        0.65,
    "Haas":          0.60,
    "Audi":          0.55,
    "Racing Bulls":  0.55,
    "Williams":      0.50,
    "Aston Martin":  0.40,
    "Cadillac":      0.30,
}
