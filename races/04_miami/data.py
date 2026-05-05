"""
R4: Miami Grand Prix
Sprint weekend. Antonelli pole. Hadjar DSQ from quali (2mm floor) starts pit lane.
Race start moved to 13:00 local because of thunderstorm forecast.
"""

RACE_INFO = {
    "name": "Miami Grand Prix",
    "circuit": "Miami International Autodrome",
    "date": "2026-05-04",
    "round": 4,
    "laps": 57,
    "is_sprint_weekend": True,
}

# Final race grid after Hadjar DSQ. Pos = grid position. q_time = absolute Q3 time.
# Hadjar's q_time kept here for reference. He starts pit lane (pos 22).
GRID = [
    {"driver": "Andrea Kimi Antonelli", "team": "Mercedes",     "pos": 1,  "q_time": 87.798},
    {"driver": "Max Verstappen",        "team": "Red Bull",     "pos": 2,  "q_time": 87.964},
    {"driver": "Charles Leclerc",       "team": "Ferrari",      "pos": 3,  "q_time": 88.143},
    {"driver": "Lando Norris",          "team": "McLaren",      "pos": 4,  "q_time": 88.183},
    {"driver": "George Russell",        "team": "Mercedes",     "pos": 5,  "q_time": 88.197},
    {"driver": "Lewis Hamilton",        "team": "Ferrari",      "pos": 6,  "q_time": 88.319},
    {"driver": "Oscar Piastri",         "team": "McLaren",      "pos": 7,  "q_time": 88.500},
    {"driver": "Franco Colapinto",      "team": "Alpine",       "pos": 8,  "q_time": 88.762},
    {"driver": "Pierre Gasly",          "team": "Alpine",       "pos": 9,  "q_time": 88.810},
    {"driver": "Nico Hulkenberg",       "team": "Audi",         "pos": 10, "q_time": 89.121},
    {"driver": "Liam Lawson",           "team": "Racing Bulls", "pos": 11, "q_time": 89.181},
    {"driver": "Ollie Bearman",         "team": "Haas",         "pos": 12, "q_time": 89.249},
    {"driver": "Carlos Sainz",          "team": "Williams",     "pos": 13, "q_time": 89.250},
    {"driver": "Esteban Ocon",          "team": "Haas",         "pos": 14, "q_time": 89.454},
    {"driver": "Alex Albon",            "team": "Williams",     "pos": 15, "q_time": 89.628},
    {"driver": "Arvid Lindblad",        "team": "Racing Bulls", "pos": 16, "q_time": 89.278},
    {"driver": "Fernando Alonso",       "team": "Aston Martin", "pos": 17, "q_time": 90.243},
    {"driver": "Lance Stroll",          "team": "Aston Martin", "pos": 18, "q_time": 90.309},
    {"driver": "Valtteri Bottas",       "team": "Cadillac",     "pos": 19, "q_time": 90.774},
    {"driver": "Sergio Perez",          "team": "Cadillac",     "pos": 20, "q_time": 91.112},
    {"driver": "Gabriel Bortoleto",     "team": "Audi",         "pos": 21, "q_time": 92.882},
    {"driver": "Isack Hadjar",          "team": "Red Bull",     "pos": 22, "q_time": None},  # DSQ'd from quali, pit lane start
]

# FP1 times in seconds. Sprint weekend gives only one practice session.
# Leclerc P1 in FP1, Verstappen P2.
FP1_TIMES = {
    "Charles Leclerc":       89.310,
    "Max Verstappen":        89.607,
    "Oscar Piastri":         89.758,
    "Lewis Hamilton":        89.777,
    "Andrea Kimi Antonelli": 90.079,
    "George Russell":        90.100,
    "Lando Norris":          90.208,
    "Pierre Gasly":          90.587,
    "Isack Hadjar":          90.873,
    "Carlos Sainz":          90.930,
    "Franco Colapinto":      91.015,
    "Alex Albon":            91.024,
    "Ollie Bearman":         91.091,
    "Gabriel Bortoleto":     91.111,
    "Nico Hulkenberg":       91.595,
    "Esteban Ocon":          91.635,
    "Liam Lawson":           91.648,
    "Sergio Perez":          92.047,
    "Fernando Alonso":       92.593,
    "Valtteri Bottas":       92.762,
    "Arvid Lindblad":        92.862,
    "Lance Stroll":          92.959,
}

# Sprint result after Antonelli +5s track limits and Bortoleto DSQ.
# Norris dominated by 3.8s. McLaren's first 2026 win.
SPRINT_RESULT = [
    {"driver": "Lando Norris",          "pos": 1},
    {"driver": "Oscar Piastri",         "pos": 2},
    {"driver": "Charles Leclerc",       "pos": 3},
    {"driver": "George Russell",        "pos": 4},
    {"driver": "Max Verstappen",        "pos": 5},
    {"driver": "Andrea Kimi Antonelli", "pos": 6},
    {"driver": "Lewis Hamilton",        "pos": 7},
    {"driver": "Pierre Gasly",          "pos": 8},
    {"driver": "Isack Hadjar",          "pos": 9},
    {"driver": "Franco Colapinto",      "pos": 10},
    {"driver": "Esteban Ocon",          "pos": 11},
    {"driver": "Ollie Bearman",         "pos": 12},
    {"driver": "Carlos Sainz",          "pos": 13},
    {"driver": "Liam Lawson",           "pos": 14},
    {"driver": "Fernando Alonso",       "pos": 15},
    {"driver": "Lance Stroll",          "pos": 16},
    {"driver": "Sergio Perez",          "pos": 17},
    {"driver": "Alex Albon",            "pos": 18},
    {"driver": "Valtteri Bottas",       "pos": 19},
    {"driver": "Arvid Lindblad",        "pos": 20},
    {"driver": "Nico Hulkenberg",       "pos": 21},  # DNS engine fire
    {"driver": "Gabriel Bortoleto",     "pos": 22},  # DSQ technical
]

# F1 seasons and last race (R3 Japan) finishing position.
# Used for adaptability and reliability features.
DRIVER_EXPERIENCE = {
    "Andrea Kimi Antonelli": {"f1_seasons": 2,  "r1_finish": 1},
    "Max Verstappen":        {"f1_seasons": 11, "r1_finish": 8},
    "Charles Leclerc":       {"f1_seasons": 8,  "r1_finish": 3},
    "Lando Norris":          {"f1_seasons": 7,  "r1_finish": 5},
    "George Russell":        {"f1_seasons": 7,  "r1_finish": 4},
    "Lewis Hamilton":        {"f1_seasons": 19, "r1_finish": 6},
    "Oscar Piastri":         {"f1_seasons": 4,  "r1_finish": 2},
    "Franco Colapinto":      {"f1_seasons": 2,  "r1_finish": 15},
    "Pierre Gasly":          {"f1_seasons": 9,  "r1_finish": 7},
    "Isack Hadjar":          {"f1_seasons": 2,  "r1_finish": 12},
    "Nico Hulkenberg":       {"f1_seasons": 14, "r1_finish": 11},
    "Liam Lawson":           {"f1_seasons": 2,  "r1_finish": 9},
    "Ollie Bearman":         {"f1_seasons": 2,  "r1_finish": None},
    "Carlos Sainz":          {"f1_seasons": 11, "r1_finish": 14},
    "Esteban Ocon":          {"f1_seasons": 9,  "r1_finish": 10},
    "Alex Albon":            {"f1_seasons": 6,  "r1_finish": 13},
    "Arvid Lindblad":        {"f1_seasons": 0,  "r1_finish": 17},
    "Fernando Alonso":       {"f1_seasons": 23, "r1_finish": 18},
    "Lance Stroll":          {"f1_seasons": 9,  "r1_finish": None},
    "Valtteri Bottas":       {"f1_seasons": 13, "r1_finish": 20},
    "Sergio Perez":          {"f1_seasons": 15, "r1_finish": 19},
    "Gabriel Bortoleto":     {"f1_seasons": 2,  "r1_finish": 16},
}

# Seconds behind fastest team (based on Q3 pace at Miami).
# Mercedes fastest with Antonelli pole.
TEAM_PACE_DEFICIT = {
    "Mercedes":     0.00,
    "Red Bull":     0.17,
    "Ferrari":      0.35,
    "McLaren":      0.39,  # Major upgrade package landed Friday
    "Alpine":       0.96,
    "Audi":         1.32,
    "Racing Bulls": 1.38,
    "Haas":         1.45,
    "Williams":     1.45,
    "Aston Martin": 2.45,
    "Cadillac":     3.00,
}

# Race start launch advantage. Positive = strong getaway.
# Antonelli has lost positions at every 2026 start. Mercedes negative.
# Ferrari and McLaren have shown better starts.
START_PROCEDURE = {
    "Mercedes":     -0.05,
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

# Energy management capability (0 to 1). 350kW MGU-K means battery
# discipline is half the race. Mercedes and McLaren strongest.
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

# Past Miami GP results. Norris won 2024. Verstappen won 2023.
CIRCUIT_HISTORY = {
    "Lando Norris":          {"wins": 1, "podiums": 2},
    "Max Verstappen":        {"wins": 1, "podiums": 3},
    "Charles Leclerc":       {"wins": 0, "podiums": 2},
    "Oscar Piastri":         {"wins": 1, "podiums": 1},  # Won 2025
    "Carlos Sainz":          {"wins": 0, "podiums": 1},
    "George Russell":        {"wins": 0, "podiums": 0},
    "Lewis Hamilton":        {"wins": 0, "podiums": 1},
    "Fernando Alonso":       {"wins": 0, "podiums": 1},
}

# Miami is a permanent street hybrid with three long straights.
# Treated as balanced. Pit loss ~22 seconds (typical for street circuits).
CIRCUIT = {
    "type": "balanced",
    "pit_loss_seconds": 22,
}

# Pirelli brought C3 hard, C4 medium, C5 soft (softest range available).
# Sprint burned new soft sets. Race expected to run medium-to-hard.
TYRE_COMPOUNDS = {
    "hardness": 0.4,            # Soft side (C3 to C5)
    "one_stop_probability": 0.7, # Medium-to-hard plan dominant
}

# Original 16:00 forecast was 88% rain plus thunderstorms. Race moved to
# 13:00 local specifically to beat that storm window. For the actual race
# window of 13:00 to 15:00, rain probability is much lower.
WEATHER = {
    "track_temp_c": 32,
    "rain_probability": 0.10,
}