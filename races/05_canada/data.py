"""
R5: Canadian Grand Prix 2026
Circuit Gilles Villeneuve, Montreal.
Sprint weekend, first time Canada hosts a sprint.
Russell ended Antonelli's three-pole streak by 0.068s. Russell also won the sprint.
Race-day forecast: 60% rain probability, cool track temps (~24°C).
Race moved to May from June for the calendar reshuffle.
"""

RACE_INFO = {
    "name": "Canadian Grand Prix",
    "circuit": "Circuit Gilles Villeneuve",
    "date": "2026-05-24",
    "round": 5,
    "laps": 70,
    "is_sprint_weekend": True,
}

# Grid by quali Q3 time. Russell on pole, Antonelli 0.068s back.
# Sprint result also feeds into the model. No grid penalties announced.
GRID = [
    {"driver": "George Russell",        "team": "Mercedes",     "pos": 1,  "q_time": 72.578},
    {"driver": "Andrea Kimi Antonelli", "team": "Mercedes",     "pos": 2,  "q_time": 72.646},
    {"driver": "Lando Norris",          "team": "McLaren",      "pos": 3,  "q_time": 72.729},
    {"driver": "Oscar Piastri",         "team": "McLaren",      "pos": 4,  "q_time": 72.781},
    {"driver": "Lewis Hamilton",        "team": "Ferrari",      "pos": 5,  "q_time": 72.868},
    {"driver": "Max Verstappen",        "team": "Red Bull",     "pos": 6,  "q_time": 72.907},
    {"driver": "Isack Hadjar",          "team": "Red Bull",     "pos": 7,  "q_time": 72.935},
    {"driver": "Charles Leclerc",       "team": "Ferrari",      "pos": 8,  "q_time": 72.976},
    {"driver": "Arvid Lindblad",        "team": "Racing Bulls", "pos": 9,  "q_time": 73.280},
    {"driver": "Franco Colapinto",      "team": "Alpine",       "pos": 10, "q_time": 73.688},
    {"driver": "Nico Hulkenberg",       "team": "Audi",         "pos": 11, "q_time": 73.309},
    {"driver": "Liam Lawson",           "team": "Racing Bulls", "pos": 12, "q_time": 73.320},
    {"driver": "Gabriel Bortoleto",     "team": "Audi",         "pos": 13, "q_time": 73.494},
    {"driver": "Pierre Gasly",          "team": "Alpine",       "pos": 14, "q_time": 73.610},
    {"driver": "Carlos Sainz",          "team": "Williams",     "pos": 15, "q_time": 73.696},
    {"driver": "Ollie Bearman",         "team": "Haas",         "pos": 16, "q_time": 73.839},
    {"driver": "Esteban Ocon",          "team": "Haas",         "pos": 17, "q_time": 73.909},
    {"driver": "Alex Albon",            "team": "Williams",     "pos": 18, "q_time": 73.915},
    {"driver": "Fernando Alonso",       "team": "Aston Martin", "pos": 19, "q_time": 74.260},
    {"driver": "Sergio Perez",          "team": "Cadillac",     "pos": 20, "q_time": 74.493},
    {"driver": "Lance Stroll",          "team": "Aston Martin", "pos": 21, "q_time": 75.259},
    {"driver": "Valtteri Bottas",       "team": "Cadillac",     "pos": 22, "q_time": 75.336},
]

# Only one practice session at a sprint weekend. Antonelli topped FP1.
# Times are estimates based on session positions and typical Montreal pace gaps.
FP1_TIMES = {
    "Andrea Kimi Antonelli": 73.412,
    "George Russell":        73.521,
    "Lando Norris":          73.687,
    "Oscar Piastri":         73.754,
    "Lewis Hamilton":        73.823,
    "Charles Leclerc":       73.901,
    "Max Verstappen":        74.012,
    "Isack Hadjar":          74.089,
    "Arvid Lindblad":        74.244,
    "Carlos Sainz":          74.401,
    "Franco Colapinto":      74.512,
    "Pierre Gasly":          74.598,
    "Nico Hulkenberg":       74.687,
    "Gabriel Bortoleto":     74.811,
    "Liam Lawson":           74.902,
    "Esteban Ocon":          74.978,
    "Ollie Bearman":         75.034,
    "Alex Albon":            75.156,
    "Sergio Perez":          75.487,
    "Fernando Alonso":       75.612,
    "Lance Stroll":          75.901,
    "Valtteri Bottas":       76.045,
}

# Sprint result (Saturday). Russell won. Antonelli P3 after tangling with Russell.
# Norris took P2 after Antonelli overshot Turn 1 on the last lap.
SPRINT_RESULT = [
    {"driver": "George Russell",        "pos": 1},
    {"driver": "Lando Norris",          "pos": 2},
    {"driver": "Andrea Kimi Antonelli", "pos": 3},
    {"driver": "Oscar Piastri",         "pos": 4},
    {"driver": "Charles Leclerc",       "pos": 5},
    {"driver": "Lewis Hamilton",        "pos": 6},
    {"driver": "Max Verstappen",        "pos": 7},
    {"driver": "Arvid Lindblad",        "pos": 8},
    {"driver": "Franco Colapinto",      "pos": 9},
    {"driver": "Carlos Sainz",          "pos": 10},
    {"driver": "Liam Lawson",           "pos": 11},
    {"driver": "Gabriel Bortoleto",     "pos": 12},
    {"driver": "Esteban Ocon",          "pos": 13},
    {"driver": "Sergio Perez",          "pos": 14},
    {"driver": "Nico Hulkenberg",       "pos": 15},
    {"driver": "Lance Stroll",          "pos": 16},
    {"driver": "Valtteri Bottas",       "pos": 17},
    {"driver": "Ollie Bearman",         "pos": 18},
    {"driver": "Alex Albon",            "pos": 19},
    {"driver": "Pierre Gasly",          "pos": 20},
    {"driver": "Isack Hadjar",          "pos": 21},
    {"driver": "Fernando Alonso",       "pos": 22},
]

# F1 seasons and last race (R4 Miami) finishing position.
# Miami DNFs: Hulkenberg, Lawson, Gasly, Hadjar.
DRIVER_EXPERIENCE = {
    "Andrea Kimi Antonelli": {"f1_seasons": 2,  "r1_finish": 1},
    "George Russell":        {"f1_seasons": 7,  "r1_finish": 4},
    "Lando Norris":          {"f1_seasons": 7,  "r1_finish": 2},
    "Oscar Piastri":         {"f1_seasons": 4,  "r1_finish": 3},
    "Lewis Hamilton":        {"f1_seasons": 19, "r1_finish": 7},
    "Max Verstappen":        {"f1_seasons": 11, "r1_finish": 5},
    "Isack Hadjar":          {"f1_seasons": 2,  "r1_finish": None},  # DNF Miami
    "Charles Leclerc":       {"f1_seasons": 8,  "r1_finish": 6},
    "Arvid Lindblad":        {"f1_seasons": 0,  "r1_finish": 14},
    "Franco Colapinto":      {"f1_seasons": 2,  "r1_finish": 8},
    "Nico Hulkenberg":       {"f1_seasons": 14, "r1_finish": None},  # DNF Miami
    "Liam Lawson":           {"f1_seasons": 2,  "r1_finish": None},  # DNF Miami
    "Gabriel Bortoleto":     {"f1_seasons": 2,  "r1_finish": 12},
    "Pierre Gasly":          {"f1_seasons": 9,  "r1_finish": None},  # DNF Miami
    "Carlos Sainz":          {"f1_seasons": 11, "r1_finish": 9},
    "Ollie Bearman":         {"f1_seasons": 2,  "r1_finish": 11},
    "Esteban Ocon":          {"f1_seasons": 9,  "r1_finish": 13},
    "Alex Albon":            {"f1_seasons": 6,  "r1_finish": 10},
    "Fernando Alonso":       {"f1_seasons": 23, "r1_finish": 15},
    "Sergio Perez":          {"f1_seasons": 15, "r1_finish": 16},
    "Lance Stroll":          {"f1_seasons": 9,  "r1_finish": 17},
    "Valtteri Bottas":       {"f1_seasons": 13, "r1_finish": 18},
}

# Seconds behind fastest team. Mercedes upgrades for Montreal pushed
# them further ahead. McLaren second-fastest, Ferrari third.
TEAM_PACE_DEFICIT = {
    "Mercedes":     0.00,
    "McLaren":      0.15,
    "Ferrari":      0.29,
    "Red Bull":     0.33,
    "Racing Bulls": 0.70,
    "Alpine":       1.11,
    "Audi":         0.73,   # Hulkenberg made Q2 with strong pace
    "Williams":     0.83,
    "Haas":         1.26,
    "Aston Martin": 1.68,
    "Cadillac":     1.92,
}

# Race start launch advantage. Mercedes finally had clean starts in
# the Canada sprint, so the persistent negative is softened.
START_PROCEDURE = {
    "Mercedes":     -0.02,
    "Ferrari":       0.05,
    "McLaren":       0.05,
    "Red Bull":      0.00,
    "Alpine":        0.00,
    "Racing Bulls":  0.00,
    "Audi":         -0.02,
    "Haas":          0.02,
    "Williams":     -0.02,
    "Aston Martin": -0.05,
    "Cadillac":     -0.08,
}

# Energy management capability. Same hierarchy as Miami.
ENERGY_READINESS = {
    "Mercedes":     0.88,
    "McLaren":      0.85,
    "Ferrari":      0.80,
    "Red Bull":     0.78,
    "Racing Bulls": 0.70,
    "Alpine":       0.60,
    "Williams":     0.62,
    "Haas":         0.58,
    "Audi":         0.50,
    "Aston Martin": 0.45,
    "Cadillac":     0.40,
}

# Past Canadian GP results. Russell won here in 2025.
# Hamilton has 7 Canadian GP wins all-time. Verstappen won 2022 and 2023.
CIRCUIT_HISTORY = {
    "George Russell":  {"wins": 1, "podiums": 2},
    "Lewis Hamilton":  {"wins": 7, "podiums": 11},
    "Max Verstappen":  {"wins": 2, "podiums": 4},
    "Fernando Alonso": {"wins": 1, "podiums": 3},
    "Lando Norris":    {"wins": 0, "podiums": 1},
    "Charles Leclerc": {"wins": 0, "podiums": 2},
    "Carlos Sainz":    {"wins": 0, "podiums": 1},
}

# Montreal is a semi-permanent street circuit with long straights,
# heavy braking zones, and walls close to the racing line.
# Treated as balanced. Pit loss ~16-18s, lower than Miami.
CIRCUIT = {
    "type": "balanced",
    "pit_loss_seconds": 17,
}

# Pirelli C3 Hard, C4 Medium, C5 Soft. Same compounds as Miami.
# Softest range available. Pirelli expects less graining than 2025.
# Cool track temps may push some teams toward two-stop strategies.
TYRE_COMPOUNDS = {
    "hardness": 0.4,
    "one_stop_probability": 0.55,  # Lower than Miami because cool track + rain risk
}

# Race day forecast: 60% rain probability. Track temp ~24°C (cooler than Miami).
# Last 2024 Canadian GP started on intermediates due to wet conditions.
# FIA may declare Rain Hazard. New 'low grip mode' for active aero could debut.
WEATHER = {
    "track_temp_c": 24,
    "rain_probability": 0.60,
}