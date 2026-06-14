"""
R7: Barcelona-Catalunya Grand Prix 2026
Circuit de Barcelona-Catalunya, Montmelo, Spain.
Note: Officially called "Barcelona-Catalunya Grand Prix" not Spanish GP,
because Madrid has the Spanish GP slot in September 2026.

Russell took pole, ending Antonelli's pole streak after Monaco.
Hamilton P2 split the Mercedes duo. Antonelli P3.
Leclerc crashed in Q3 at Turn 4, will start P10 with no representative lap.
Dry forecast Sunday. Mercedes traditionally strong here.
"""

RACE_INFO = {
    "name": "Barcelona-Catalunya Grand Prix",
    "circuit": "Circuit de Barcelona-Catalunya",
    "date": "2026-06-14",
    "round": 7,
    "laps": 66,
    "is_sprint_weekend": False,
}

# Grid from quali. Russell pole, Hamilton P2, Antonelli P3.
# Leclerc crashed in Q3 with no Q3 time, qualified P10.
GRID = [
    {"driver": "George Russell",        "team": "Mercedes",     "pos": 1,  "q_time": 75.234},
    {"driver": "Lewis Hamilton",        "team": "Ferrari",      "pos": 2,  "q_time": 75.298},
    {"driver": "Andrea Kimi Antonelli", "team": "Mercedes",     "pos": 3,  "q_time": 75.367},
    {"driver": "Lando Norris",          "team": "McLaren",      "pos": 4,  "q_time": 75.456},
    {"driver": "Max Verstappen",        "team": "Red Bull",     "pos": 5,  "q_time": 75.523},
    {"driver": "Isack Hadjar",          "team": "Red Bull",     "pos": 6,  "q_time": 75.612},
    {"driver": "Oscar Piastri",         "team": "McLaren",      "pos": 7,  "q_time": 75.698},
    {"driver": "Liam Lawson",           "team": "Racing Bulls", "pos": 8,  "q_time": 75.789},
    {"driver": "Nico Hulkenberg",       "team": "Audi",         "pos": 9,  "q_time": 75.867},
    {"driver": "Charles Leclerc",       "team": "Ferrari",      "pos": 10, "q_time": None},  # Crashed Q3 no time
    {"driver": "Pierre Gasly",          "team": "Alpine",       "pos": 11, "q_time": 76.123},
    {"driver": "Franco Colapinto",      "team": "Alpine",       "pos": 12, "q_time": 76.234},
    {"driver": "Arvid Lindblad",        "team": "Racing Bulls", "pos": 13, "q_time": 76.345},
    {"driver": "Carlos Sainz",          "team": "Williams",     "pos": 14, "q_time": 76.456},
    {"driver": "Gabriel Bortoleto",     "team": "Audi",         "pos": 15, "q_time": 76.567},
    {"driver": "Alex Albon",            "team": "Williams",     "pos": 16, "q_time": 76.689},
    {"driver": "Esteban Ocon",          "team": "Haas",         "pos": 17, "q_time": 76.801},
    {"driver": "Ollie Bearman",         "team": "Haas",         "pos": 18, "q_time": 76.912},
    {"driver": "Sergio Perez",          "team": "Cadillac",     "pos": 19, "q_time": 77.034},
    {"driver": "Valtteri Bottas",       "team": "Cadillac",     "pos": 20, "q_time": 77.156},
    {"driver": "Lance Stroll",          "team": "Aston Martin", "pos": 21, "q_time": 77.289},
    {"driver": "Fernando Alonso",       "team": "Aston Martin", "pos": 22, "q_time": 77.412},
]

# FP1 from Friday. Russell was fast all weekend.
FP1_TIMES = {
    "George Russell":        76.123,
    "Andrea Kimi Antonelli": 76.234,
    "Lando Norris":          76.345,
    "Lewis Hamilton":        76.412,
    "Oscar Piastri":         76.489,
    "Max Verstappen":        76.567,
    "Charles Leclerc":       76.645,
    "Isack Hadjar":          76.723,
    "Liam Lawson":           76.890,
    "Carlos Sainz":          76.978,
    "Pierre Gasly":          77.067,
    "Nico Hulkenberg":       77.156,
    "Franco Colapinto":      77.234,
    "Gabriel Bortoleto":     77.345,
    "Alex Albon":            77.456,
    "Arvid Lindblad":        77.567,
    "Esteban Ocon":          77.678,
    "Ollie Bearman":         77.789,
    "Sergio Perez":          78.012,
    "Valtteri Bottas":       78.145,
    "Fernando Alonso":       78.267,
    "Lance Stroll":          78.398,
}

SPRINT_RESULT = []

# Monaco finishes feed in as r1_finish (last race result).
DRIVER_EXPERIENCE = {
    "George Russell":        {"f1_seasons": 7,  "r1_finish": 12},  # P12 Monaco
    "Lewis Hamilton":        {"f1_seasons": 19, "r1_finish": 2},   # P2 Monaco
    "Andrea Kimi Antonelli": {"f1_seasons": 2,  "r1_finish": 1},   # Won Monaco
    "Lando Norris":          {"f1_seasons": 7,  "r1_finish": None}, # DNF Monaco
    "Max Verstappen":        {"f1_seasons": 11, "r1_finish": None}, # DNF Monaco
    "Isack Hadjar":          {"f1_seasons": 2,  "r1_finish": 4},
    "Oscar Piastri":         {"f1_seasons": 4,  "r1_finish": 5},
    "Liam Lawson":           {"f1_seasons": 2,  "r1_finish": 6},
    "Nico Hulkenberg":       {"f1_seasons": 14, "r1_finish": 13},
    "Charles Leclerc":       {"f1_seasons": 8,  "r1_finish": None}, # DNF Monaco
    "Pierre Gasly":          {"f1_seasons": 9,  "r1_finish": 3},   # P3 Monaco
    "Franco Colapinto":      {"f1_seasons": 2,  "r1_finish": 14},
    "Arvid Lindblad":        {"f1_seasons": 0,  "r1_finish": 7},
    "Carlos Sainz":          {"f1_seasons": 11, "r1_finish": None}, # DNF Monaco
    "Gabriel Bortoleto":     {"f1_seasons": 2,  "r1_finish": 11},
    "Alex Albon":            {"f1_seasons": 6,  "r1_finish": 8},
    "Esteban Ocon":          {"f1_seasons": 9,  "r1_finish": 9},
    "Ollie Bearman":         {"f1_seasons": 2,  "r1_finish": None}, # DNF Monaco
    "Sergio Perez":          {"f1_seasons": 15, "r1_finish": 15},
    "Valtteri Bottas":       {"f1_seasons": 13, "r1_finish": None}, # DNF Monaco
    "Lance Stroll":          {"f1_seasons": 9,  "r1_finish": None}, # DNF Monaco
    "Fernando Alonso":       {"f1_seasons": 23, "r1_finish": 10},
}

# Computed from quali Q3 times. Russell on pole, Hamilton 0.064s back.
TEAM_PACE_DEFICIT = {
    "Mercedes":     0.00,
    "Ferrari":      0.06,
    "McLaren":      0.22,
    "Red Bull":     0.29,
    "Racing Bulls": 0.56,
    "Audi":         0.63,
    "Alpine":       0.89,
    "Williams":     1.22,
    "Haas":         1.57,
    "Cadillac":     1.80,
    "Aston Martin": 2.05,
}

# Defaults work for Barcelona. Mercedes neutral. Cadillac penalised by Perez false start in Monaco.
START_PROCEDURE = {
    "Mercedes":      0.00,
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

# Mercedes dominance + new rear wing keeps energy management top.
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

# Hamilton has 6 Spanish/Barcelona GP wins, most of any active driver.
# Verstappen has 3. Alonso 2 (home race veteran).
CIRCUIT_HISTORY = {
    "Lewis Hamilton":  {"wins": 6, "podiums": 11},
    "Max Verstappen":  {"wins": 3, "podiums": 5},
    "Fernando Alonso": {"wins": 2, "podiums": 4},
    "Charles Leclerc": {"wins": 0, "podiums": 1},
    "Lando Norris":    {"wins": 0, "podiums": 1},
    "Carlos Sainz":    {"wins": 0, "podiums": 1},
    "George Russell":  {"wins": 0, "podiums": 1},
}

# Barcelona is a balanced circuit. Long medium-speed corners.
# Pit loss moderate (~21s).
CIRCUIT = {
    "type": "balanced",
    "pit_loss_seconds": 21,
}

# Pirelli C2-C3-C4. Harder than Monaco. Two-stop common.
TYRE_COMPOUNDS = {
    "hardness": 0.60,
    "one_stop_probability": 0.55,
}

# Sunday June 14 forecast: warm and dry. Track ~42C.
WEATHER = {
    "track_temp_c": 42,
    "rain_probability": 0.05,
}