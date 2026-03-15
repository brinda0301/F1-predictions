"""
2026 Australian Grand Prix - Race Data
Albert Park Circuit, Melbourne
Race: Sunday March 8, 2026 | 58 laps | 306km
"""

RACE_INFO = {
    "round": 1,
    "name": "Australian Grand Prix",
    "circuit": "Albert Park, Melbourne",
    "date": "2026-03-08",
    "laps": 58,
    "distance_km": 306,
    "format": "standard",
}

GRID = [
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

FP1_TIMES = {
    "George Russell": 80.2, "Kimi Antonelli": 80.5, "Charles Leclerc": 79.9,
    "Lewis Hamilton": 80.0, "Max Verstappen": 80.3, "Oscar Piastri": 80.4,
    "Lando Norris": 80.6, "Isack Hadjar": 80.7, "Liam Lawson": 80.9,
    "Arvid Lindblad": 81.0, "Gabriel Bortoleto": 81.1, "Nico Hulkenberg": 81.2,
    "Oliver Bearman": 81.3, "Esteban Ocon": 81.4, "Pierre Gasly": 81.5,
    "Alex Albon": 81.6, "Franco Colapinto": 81.7, "Fernando Alonso": 81.8,
    "Sergio Perez": 82.0, "Valtteri Bottas": 82.2, "Carlos Sainz": 82.5,
    "Lance Stroll": 83.0,
}

SPRINT_RESULT = []

DRIVER_EXPERIENCE = {
    "George Russell":     {"f1_seasons": 7,  "career_poles": 5},
    "Kimi Antonelli":     {"f1_seasons": 1,  "career_poles": 0},
    "Isack Hadjar":       {"f1_seasons": 0,  "career_poles": 0},
    "Charles Leclerc":    {"f1_seasons": 7,  "career_poles": 26},
    "Oscar Piastri":      {"f1_seasons": 3,  "career_poles": 2},
    "Lando Norris":       {"f1_seasons": 6,  "career_poles": 8},
    "Lewis Hamilton":     {"f1_seasons": 18, "career_poles": 104},
    "Liam Lawson":        {"f1_seasons": 1,  "career_poles": 0},
    "Arvid Lindblad":     {"f1_seasons": 0,  "career_poles": 0},
    "Gabriel Bortoleto":  {"f1_seasons": 0,  "career_poles": 0},
    "Nico Hulkenberg":    {"f1_seasons": 14, "career_poles": 1},
    "Oliver Bearman":     {"f1_seasons": 1,  "career_poles": 0},
    "Esteban Ocon":       {"f1_seasons": 8,  "career_poles": 0},
    "Pierre Gasly":       {"f1_seasons": 8,  "career_poles": 0},
    "Alex Albon":         {"f1_seasons": 5,  "career_poles": 0},
    "Franco Colapinto":   {"f1_seasons": 1,  "career_poles": 0},
    "Fernando Alonso":    {"f1_seasons": 22, "career_poles": 22},
    "Sergio Perez":       {"f1_seasons": 14, "career_poles": 3},
    "Valtteri Bottas":    {"f1_seasons": 13, "career_poles": 20},
    "Max Verstappen":     {"f1_seasons": 10, "career_poles": 40},
    "Carlos Sainz":       {"f1_seasons": 10, "career_poles": 6},
    "Lance Stroll":       {"f1_seasons": 8,  "career_poles": 1},
}

TEAM_PACE_DEFICIT = {
    "Mercedes": 0.0, "Ferrari": 0.15, "McLaren": 0.20, "Red Bull": 0.25,
    "Racing Bulls": 0.60, "Audi": 0.70, "Haas": 0.80, "Alpine": 0.85,
    "Williams": 0.90, "Aston Martin": 1.0, "Cadillac": 1.2,
}

START_PROCEDURE = {
    "Ferrari": 0.3, "Mercedes": 0.0, "McLaren": 0.0, "Red Bull": -0.1,
    "Racing Bulls": 0.0, "Audi": -0.1, "Haas": 0.0, "Alpine": 0.0,
    "Williams": -0.1, "Aston Martin": -0.2, "Cadillac": -0.2,
}

ENERGY_READINESS = {
    "Mercedes": 0.90, "Ferrari": 0.85, "McLaren": 0.80, "Red Bull": 0.75,
    "Racing Bulls": 0.65, "Audi": 0.50, "Haas": 0.60, "Alpine": 0.55,
    "Williams": 0.50, "Aston Martin": 0.40, "Cadillac": 0.35,
}

CIRCUIT_HISTORY = {
    "Lewis Hamilton": {"wins": 2, "podiums": 8},
    "Charles Leclerc": {"wins": 1, "podiums": 2},
    "Max Verstappen": {"wins": 1, "podiums": 3},
    "Lando Norris": {"wins": 1, "podiums": 2},
    "Carlos Sainz": {"wins": 1, "podiums": 1},
    "Valtteri Bottas": {"wins": 1, "podiums": 3},
    "Fernando Alonso": {"wins": 1, "podiums": 3},
    "George Russell": {"wins": 0, "podiums": 1},
}

# -- new v2 fields --

CIRCUIT = {
    "type": "balanced",         # mix of fast straights and slow chicanes
    "pit_loss_seconds": 21,     # time lost entering/exiting pits
    "laps": 58,
    "sc_probability": 0.75,     # Albert Park hands out safety cars
}

TYRE_COMPOUNDS = {
    "hard": "C3",
    "medium": "C4",
    "soft": "C5",
    "hardness": 0.3,            # softer range (0=softest, 1=hardest)
    "one_stop_probability": 0.60,
    "graining_risk": 0.4,       # front tyre graining was present
}

WEATHER = {
    "track_temp_c": 35,
    "air_temp_c": 22,
    "rain_probability": 0.05,   # dry race
    "wind_kph": 15,
}
