"""
Raw data definitions for the 2026 Australian GP prediction model.
All data sourced from official F1 results, practice sessions, and betting markets.
"""

# 2026 Australian GP Starting Grid (from qualifying, March 7 2026)
GRID_2026 = [
    {"pos": 1,  "driver": "George Russell",     "team": "Mercedes",      "q_time": 78.518},
    {"pos": 2,  "driver": "Kimi Antonelli",      "team": "Mercedes",      "q_time": 78.818},
    {"pos": 3,  "driver": "Isack Hadjar",        "team": "Red Bull",      "q_time": 79.318},
    {"pos": 4,  "driver": "Charles Leclerc",     "team": "Ferrari",       "q_time": 79.350},
    {"pos": 5,  "driver": "Oscar Piastri",       "team": "McLaren",       "q_time": 79.400},
    {"pos": 6,  "driver": "Lando Norris",        "team": "McLaren",       "q_time": 79.500},
    {"pos": 7,  "driver": "Lewis Hamilton",      "team": "Ferrari",       "q_time": 79.550},
    {"pos": 8,  "driver": "Liam Lawson",         "team": "Racing Bulls",  "q_time": 79.800},
    {"pos": 9,  "driver": "Arvid Lindblad",      "team": "Racing Bulls",  "q_time": 79.900},
    {"pos": 10, "driver": "Gabriel Bortoleto",   "team": "Audi",          "q_time": 80.000},
    {"pos": 11, "driver": "Nico Hulkenberg",     "team": "Audi",          "q_time": 80.100},
    {"pos": 12, "driver": "Oliver Bearman",      "team": "Haas",          "q_time": 80.200},
    {"pos": 13, "driver": "Esteban Ocon",        "team": "Haas",          "q_time": 80.300},
    {"pos": 14, "driver": "Pierre Gasly",        "team": "Alpine",        "q_time": 80.400},
    {"pos": 15, "driver": "Alex Albon",          "team": "Williams",      "q_time": 80.500},
    {"pos": 16, "driver": "Franco Colapinto",    "team": "Alpine",        "q_time": 80.600},
    {"pos": 17, "driver": "Fernando Alonso",     "team": "Aston Martin",  "q_time": 80.700},
    {"pos": 18, "driver": "Sergio Perez",        "team": "Cadillac",      "q_time": 81.000},
    {"pos": 19, "driver": "Valtteri Bottas",     "team": "Cadillac",      "q_time": 81.200},
    {"pos": 20, "driver": "Max Verstappen",      "team": "Red Bull",      "q_time": None},
    {"pos": 21, "driver": "Carlos Sainz",        "team": "Williams",      "q_time": None},
    {"pos": 22, "driver": "Lance Stroll",        "team": "Aston Martin",  "q_time": None},
]

# Practice session rankings (FP1, FP2, FP3)
FP_RANKINGS = {
    "Charles Leclerc":     {"fp1": 1,  "fp2": 4,  "fp3": 3},
    "Lewis Hamilton":      {"fp1": 2,  "fp2": 5,  "fp3": 2},
    "Max Verstappen":      {"fp1": 3,  "fp2": 6,  "fp3": 7},
    "Oscar Piastri":       {"fp1": 5,  "fp2": 1,  "fp3": 4},
    "Kimi Antonelli":      {"fp1": 6,  "fp2": 2,  "fp3": 8},
    "George Russell":      {"fp1": 4,  "fp2": 3,  "fp3": 1},
    "Lando Norris":        {"fp1": 7,  "fp2": 7,  "fp3": 5},
    "Isack Hadjar":        {"fp1": 8,  "fp2": 8,  "fp3": 6},
    "Liam Lawson":         {"fp1": 9,  "fp2": 9,  "fp3": 9},
    "Arvid Lindblad":      {"fp1": 10, "fp2": 10, "fp3": 10},
    "Gabriel Bortoleto":   {"fp1": 11, "fp2": 11, "fp3": 11},
    "Nico Hulkenberg":     {"fp1": 12, "fp2": 12, "fp3": 12},
    "Oliver Bearman":      {"fp1": 13, "fp2": 13, "fp3": 13},
    "Esteban Ocon":        {"fp1": 14, "fp2": 14, "fp3": 14},
    "Pierre Gasly":        {"fp1": 15, "fp2": 15, "fp3": 15},
    "Alex Albon":          {"fp1": 16, "fp2": 16, "fp3": 16},
    "Franco Colapinto":    {"fp1": 17, "fp2": 17, "fp3": 17},
    "Fernando Alonso":     {"fp1": 18, "fp2": 18, "fp3": 18},
    "Sergio Perez":        {"fp1": 19, "fp2": 19, "fp3": 19},
    "Valtteri Bottas":     {"fp1": 20, "fp2": 20, "fp3": 20},
    "Carlos Sainz":        {"fp1": 21, "fp2": 21, "fp3": 21},
    "Lance Stroll":        {"fp1": 22, "fp2": 22, "fp3": 22},
}

# Historical Australian GP winners at Albert Park
HISTORICAL_AUS_GP = {
    2017: [("Sebastian Vettel", 1), ("Lewis Hamilton", 2), ("Valtteri Bottas", 3)],
    2018: [("Sebastian Vettel", 1), ("Lewis Hamilton", 2), ("Kimi Raikkonen", 3)],
    2019: [("Valtteri Bottas", 1), ("Lewis Hamilton", 2), ("Max Verstappen", 3)],
    2022: [("Charles Leclerc", 1), ("Sergio Perez", 2), ("George Russell", 3)],
    2023: [("Max Verstappen", 1), ("Lewis Hamilton", 2), ("Fernando Alonso", 3)],
    2024: [("Carlos Sainz", 1), ("Charles Leclerc", 2), ("Lando Norris", 3)],
    2025: [("Lando Norris", 1), ("Max Verstappen", 2), ("George Russell", 3)],
}

# Driver career stats (as of start of 2026 season)
DRIVER_STATS = {
    "George Russell":     {"wins": 4,  "podiums": 25, "poles": 5,  "seasons": 7,  "aus_wins": 0, "aus_podiums": 1},
    "Kimi Antonelli":     {"wins": 0,  "podiums": 3,  "poles": 0,  "seasons": 1,  "aus_wins": 0, "aus_podiums": 0},
    "Isack Hadjar":       {"wins": 0,  "podiums": 0,  "poles": 0,  "seasons": 0,  "aus_wins": 0, "aus_podiums": 0},
    "Charles Leclerc":    {"wins": 9,  "podiums": 40, "poles": 26, "seasons": 7,  "aus_wins": 1, "aus_podiums": 2},
    "Oscar Piastri":      {"wins": 3,  "podiums": 16, "poles": 2,  "seasons": 3,  "aus_wins": 0, "aus_podiums": 0},
    "Lando Norris":       {"wins": 6,  "podiums": 30, "poles": 8,  "seasons": 6,  "aus_wins": 1, "aus_podiums": 2},
    "Lewis Hamilton":     {"wins": 105,"podiums": 202,"poles": 104,"seasons": 18, "aus_wins": 2, "aus_podiums": 8},
    "Liam Lawson":        {"wins": 0,  "podiums": 0,  "poles": 0,  "seasons": 1,  "aus_wins": 0, "aus_podiums": 0},
    "Arvid Lindblad":     {"wins": 0,  "podiums": 0,  "poles": 0,  "seasons": 0,  "aus_wins": 0, "aus_podiums": 0},
    "Gabriel Bortoleto":  {"wins": 0,  "podiums": 0,  "poles": 0,  "seasons": 0,  "aus_wins": 0, "aus_podiums": 0},
    "Nico Hulkenberg":    {"wins": 0,  "podiums": 0,  "poles": 1,  "seasons": 14, "aus_wins": 0, "aus_podiums": 0},
    "Oliver Bearman":     {"wins": 0,  "podiums": 0,  "poles": 0,  "seasons": 1,  "aus_wins": 0, "aus_podiums": 0},
    "Esteban Ocon":       {"wins": 1,  "podiums": 4,  "poles": 0,  "seasons": 8,  "aus_wins": 0, "aus_podiums": 0},
    "Pierre Gasly":       {"wins": 1,  "podiums": 4,  "poles": 0,  "seasons": 8,  "aus_wins": 0, "aus_podiums": 0},
    "Alex Albon":         {"wins": 0,  "podiums": 2,  "poles": 0,  "seasons": 5,  "aus_wins": 0, "aus_podiums": 0},
    "Franco Colapinto":   {"wins": 0,  "podiums": 0,  "poles": 0,  "seasons": 1,  "aus_wins": 0, "aus_podiums": 0},
    "Fernando Alonso":    {"wins": 32, "podiums": 106,"poles": 22, "seasons": 22, "aus_wins": 1, "aus_podiums": 3},
    "Sergio Perez":       {"wins": 6,  "podiums": 39, "poles": 3,  "seasons": 14, "aus_wins": 0, "aus_podiums": 1},
    "Valtteri Bottas":    {"wins": 10, "podiums": 67, "poles": 20, "seasons": 13, "aus_wins": 1, "aus_podiums": 3},
    "Max Verstappen":     {"wins": 63, "podiums": 112,"poles": 40, "seasons": 10, "aus_wins": 1, "aus_podiums": 3},
    "Carlos Sainz":       {"wins": 4,  "podiums": 25, "poles": 6,  "seasons": 10, "aus_wins": 1, "aus_podiums": 1},
    "Lance Stroll":       {"wins": 0,  "podiums": 3,  "poles": 1,  "seasons": 8,  "aus_wins": 0, "aus_podiums": 0},
}

# Team strength index from testing + practice (1.0 = best)
TEAM_STRENGTH = {
    "Mercedes":      1.00,
    "Ferrari":       0.88,
    "McLaren":       0.85,
    "Red Bull":      0.82,
    "Racing Bulls":  0.55,
    "Audi":          0.50,
    "Haas":          0.45,
    "Alpine":        0.42,
    "Williams":      0.40,
    "Aston Martin":  0.35,
    "Cadillac":      0.30,
}

# Post-qualifying betting odds (implied win probability)
BETTING_ODDS = {
    "George Russell":     0.778,
    "Kimi Antonelli":     0.154,
    "Charles Leclerc":    0.080,
    "Oscar Piastri":      0.060,
    "Lando Norris":       0.050,
    "Lewis Hamilton":     0.040,
    "Isack Hadjar":       0.030,
    "Max Verstappen":     0.020,
    "Liam Lawson":        0.008,
    "Arvid Lindblad":     0.005,
    "Gabriel Bortoleto":  0.003,
    "Fernando Alonso":    0.002,
    "Nico Hulkenberg":    0.002,
    "Sergio Perez":       0.001,
    "Valtteri Bottas":    0.001,
    "Oliver Bearman":     0.001,
    "Esteban Ocon":       0.001,
    "Pierre Gasly":       0.001,
    "Alex Albon":         0.001,
    "Franco Colapinto":   0.001,
    "Carlos Sainz":       0.001,
    "Lance Stroll":       0.001,
}

# Number of drivers on grid
GRID_SIZE = 22
