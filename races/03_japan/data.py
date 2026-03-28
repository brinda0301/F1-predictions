"""
2026 Japanese Grand Prix - Race Data
Suzuka Circuit
Race: Sunday March 29, 2026 | 53 laps | 307km

Circuit notes:
- 5.807km, 18 corners, figure-eight layout
- One of the toughest circuits for tyres on the calendar
- High-speed S-curves (T3-T7) + Degner curves + 130R demand aero confidence
- Resurfaced across almost the entire lap (sectors 2+3 done after 2025)
- New asphalt = low grip, expect graining on softer compounds
- Pirelli: C1 (Hard), C2 (Medium), C3 (Soft) - HARDEST range, C1 debut
- Two Straight Mode zones, one Overtake Mode activation point
- Antonelli on pole for the third straight race
- Verstappen out in Q2, called his car "undrivable"
"""

RACE_INFO = {
    "round": 3,
    "name": "Japanese Grand Prix",
    "circuit": "Suzuka Circuit",
    "date": "2026-03-29",
    "laps": 53,
    "distance_km": 307,
    "format": "standard",
}

GRID = [
    {"pos": 1,  "driver": "Kimi Antonelli",      "team": "Mercedes",      "q_time": 88.778},
    {"pos": 2,  "driver": "George Russell",       "team": "Mercedes",      "q_time": 89.076},
    {"pos": 3,  "driver": "Oscar Piastri",        "team": "McLaren",       "q_time": 89.132},
    {"pos": 4,  "driver": "Charles Leclerc",      "team": "Ferrari",       "q_time": 89.405},
    {"pos": 5,  "driver": "Lando Norris",         "team": "McLaren",       "q_time": 89.409},
    {"pos": 6,  "driver": "Lewis Hamilton",       "team": "Ferrari",       "q_time": 89.567},
    {"pos": 7,  "driver": "Pierre Gasly",         "team": "Alpine",        "q_time": 89.691},
    {"pos": 8,  "driver": "Isack Hadjar",         "team": "Red Bull",      "q_time": 89.978},
    {"pos": 9,  "driver": "Gabriel Bortoleto",    "team": "Audi",          "q_time": 90.274},
    {"pos": 10, "driver": "Arvid Lindblad",       "team": "Racing Bulls",  "q_time": 90.319},
    {"pos": 11, "driver": "Max Verstappen",       "team": "Red Bull",      "q_time": 90.406},
    {"pos": 12, "driver": "Esteban Ocon",         "team": "Haas",          "q_time": 90.453},
    {"pos": 13, "driver": "Nico Hulkenberg",      "team": "Audi",          "q_time": 90.522},
    {"pos": 14, "driver": "Liam Lawson",          "team": "Racing Bulls",  "q_time": 90.639},
    {"pos": 15, "driver": "Franco Colapinto",     "team": "Alpine",        "q_time": 90.771},
    {"pos": 16, "driver": "Carlos Sainz",         "team": "Williams",      "q_time": 91.177},
    {"pos": 17, "driver": "Alex Albon",           "team": "Williams",      "q_time": 90.365},
    {"pos": 18, "driver": "Oliver Bearman",       "team": "Haas",          "q_time": 90.367},
    {"pos": 19, "driver": "Sergio Perez",         "team": "Cadillac",      "q_time": 91.483},
    {"pos": 20, "driver": "Valtteri Bottas",      "team": "Cadillac",      "q_time": 91.607},
    {"pos": 21, "driver": "Fernando Alonso",      "team": "Aston Martin",  "q_time": 91.923},
    {"pos": 22, "driver": "Lance Stroll",         "team": "Aston Martin",  "q_time": 92.197},
]

FP1_TIMES = {
    "George Russell": 90.1, "Kimi Antonelli": 89.9, "Lewis Hamilton": 90.4,
    "Charles Leclerc": 90.3, "Oscar Piastri": 90.0, "Lando Norris": 90.5,
    "Max Verstappen": 90.8, "Isack Hadjar": 91.0, "Pierre Gasly": 90.7,
    "Liam Lawson": 91.2, "Arvid Lindblad": 91.1, "Alex Albon": 91.5,
    "Oliver Bearman": 91.3, "Esteban Ocon": 91.4, "Gabriel Bortoleto": 91.0,
    "Franco Colapinto": 91.6, "Carlos Sainz": 91.8, "Nico Hulkenberg": 91.3,
    "Fernando Alonso": 92.2, "Sergio Perez": 92.0, "Lance Stroll": 92.5,
    "Valtteri Bottas": 92.3,
}

SPRINT_RESULT = []

DRIVER_EXPERIENCE = {
    "Kimi Antonelli":     {"f1_seasons": 1,  "career_poles": 2,  "r1_finish": 2,  "r2_finish": 1},
    "George Russell":     {"f1_seasons": 7,  "career_poles": 5,  "r1_finish": 1,  "r2_finish": 2},
    "Oscar Piastri":      {"f1_seasons": 3,  "career_poles": 2},
    "Charles Leclerc":    {"f1_seasons": 7,  "career_poles": 26, "r1_finish": 3,  "r2_finish": 4},
    "Lando Norris":       {"f1_seasons": 6,  "career_poles": 8,  "r1_finish": 5},
    "Lewis Hamilton":     {"f1_seasons": 18, "career_poles": 104, "r1_finish": 4, "r2_finish": 3},
    "Pierre Gasly":       {"f1_seasons": 8,  "career_poles": 0,  "r1_finish": 10, "r2_finish": 6},
    "Isack Hadjar":       {"f1_seasons": 0,  "career_poles": 0,  "r2_finish": 8},
    "Gabriel Bortoleto":  {"f1_seasons": 0,  "career_poles": 0,  "r1_finish": 9},
    "Arvid Lindblad":     {"f1_seasons": 0,  "career_poles": 0,  "r1_finish": 8,  "r2_finish": 12},
    "Max Verstappen":     {"f1_seasons": 10, "career_poles": 40, "r1_finish": 6},
    "Esteban Ocon":       {"f1_seasons": 8,  "career_poles": 0,  "r1_finish": 11, "r2_finish": 14},
    "Nico Hulkenberg":    {"f1_seasons": 14, "career_poles": 1,  "r2_finish": 11},
    "Liam Lawson":        {"f1_seasons": 1,  "career_poles": 0,  "r1_finish": 13, "r2_finish": 7},
    "Franco Colapinto":   {"f1_seasons": 1,  "career_poles": 0,  "r1_finish": 14, "r2_finish": 10},
    "Carlos Sainz":       {"f1_seasons": 10, "career_poles": 6,  "r2_finish": 9},
    "Alex Albon":         {"f1_seasons": 5,  "career_poles": 0,  "r1_finish": 12},
    "Oliver Bearman":     {"f1_seasons": 1,  "career_poles": 0,  "r1_finish": 7,  "r2_finish": 5},
    "Sergio Perez":       {"f1_seasons": 14, "career_poles": 3,  "r1_finish": 15, "r2_finish": 15},
    "Valtteri Bottas":    {"f1_seasons": 13, "career_poles": 20, "r2_finish": 13},
    "Fernando Alonso":    {"f1_seasons": 22, "career_poles": 22, "r1_finish": 16},
    "Lance Stroll":       {"f1_seasons": 8,  "career_poles": 1},
}

TEAM_PACE_DEFICIT = {
    "Mercedes": 0.0,
    "Ferrari": 0.15,
    "McLaren": 0.10,
    "Red Bull": 0.35,
    "Alpine": 0.50,
    "Racing Bulls": 0.55,
    "Audi": 0.60,
    "Haas": 0.65,
    "Williams": 0.85,
    "Aston Martin": 1.05,
    "Cadillac": 1.10,
}

START_PROCEDURE = {
    "Ferrari": 0.20,
    "Mercedes": 0.10,
    "McLaren": 0.05,
    "Red Bull": -0.05,
    "Racing Bulls": 0.0,
    "Alpine": -0.05,
    "Audi": -0.1,
    "Haas": 0.0,
    "Williams": -0.1,
    "Aston Martin": -0.15,
    "Cadillac": -0.2,
}

ENERGY_READINESS = {
    "Mercedes": 0.92,
    "Ferrari": 0.85,
    "McLaren": 0.82,
    "Red Bull": 0.68,
    "Alpine": 0.62,
    "Racing Bulls": 0.60,
    "Audi": 0.52,
    "Haas": 0.58,
    "Williams": 0.50,
    "Aston Martin": 0.40,
    "Cadillac": 0.38,
}

CIRCUIT_HISTORY = {
    "Lewis Hamilton": {"wins": 5, "podiums": 9},
    "Max Verstappen": {"wins": 4, "podiums": 5},
    "Valtteri Bottas": {"wins": 1, "podiums": 5},
    "Charles Leclerc": {"wins": 0, "podiums": 2},
    "George Russell": {"wins": 0, "podiums": 1},
    "Lando Norris": {"wins": 0, "podiums": 1},
    "Oscar Piastri": {"wins": 0, "podiums": 0},
    "Fernando Alonso": {"wins": 0, "podiums": 3},
    "Sergio Perez": {"wins": 0, "podiums": 2},
}

CIRCUIT = {
    "type": "high_speed",
    "pit_loss_seconds": 22,
    "laps": 53,
    "sc_probability": 0.40,
    "heavy_braking_zones": 3,
}

TYRE_COMPOUNDS = {
    "hard": "C1",
    "medium": "C2",
    "soft": "C3",
    "hardness": 0.9,
    "one_stop_probability": 0.75,
    "graining_risk": 0.5,
}

WEATHER = {
    "track_temp_c": 25,
    "air_temp_c": 19,
    "rain_probability": 0.15,
    "wind_kph": 8,
}