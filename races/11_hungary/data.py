"""
R11: Hungary 2026
Grid, FP1 and pace built by hand: FastF1 had no Hungary session data at run time.
FINAL GRID: post-penalty, FIA confirmed.
Hamilton -3 (impeding Piastri), Antonelli -3 (yellow flags).
Perez starts from the pit lane (parc ferme breach), modelled here as P22.
"""

RACE_INFO = {"name": "Hungarian Grand Prix", "circuit": "Hungary", "date": "2026-07-26", "round": 11, "is_sprint_weekend": False}

GRID = [
    {"driver": "Lando Norris", "team": "McLaren", "pos": 1, "q_time": 77.207},
    {"driver": "Lewis Hamilton", "team": "Ferrari", "pos": 5, "q_time": 77.219},
    {"driver": "Charles Leclerc", "team": "Ferrari", "pos": 2, "q_time": 77.445},
    {"driver": "Andrea Kimi Antonelli", "team": "Mercedes", "pos": 7, "q_time": 77.479},
    {"driver": "Oscar Piastri", "team": "McLaren", "pos": 3, "q_time": 77.684},
    {"driver": "Max Verstappen", "team": "Red Bull", "pos": 4, "q_time": 77.725},
    {"driver": "George Russell", "team": "Mercedes", "pos": 6, "q_time": 77.760},
    {"driver": "Isack Hadjar", "team": "Red Bull", "pos": 8, "q_time": 77.856},
    {"driver": "Arvid Lindblad", "team": "Racing Bulls", "pos": 9, "q_time": 78.281},
    {"driver": "Nico Hulkenberg", "team": "Audi", "pos": 10, "q_time": 78.686},
    {"driver": "Liam Lawson", "team": "Racing Bulls", "pos": 11, "q_time": 78.765},
    {"driver": "Pierre Gasly", "team": "Alpine", "pos": 12, "q_time": 78.844},
    {"driver": "Franco Colapinto", "team": "Alpine", "pos": 13, "q_time": 79.027},
    {"driver": "Gabriel Bortoleto", "team": "Audi", "pos": 14, "q_time": 79.105},
    {"driver": "Esteban Ocon", "team": "Haas", "pos": 15, "q_time": 79.734},
    {"driver": "Fernando Alonso", "team": "Aston Martin", "pos": 16, "q_time": 79.808},
    {"driver": "Ollie Bearman", "team": "Haas", "pos": 17, "q_time": 80.233},
    {"driver": "Carlos Sainz", "team": "Williams", "pos": 18, "q_time": 80.621},
    {"driver": "Alex Albon", "team": "Williams", "pos": 19, "q_time": 80.658},
    {"driver": "Lance Stroll", "team": "Aston Martin", "pos": 20, "q_time": 80.659},
    {"driver": "Valtteri Bottas", "team": "Cadillac", "pos": 21, "q_time": 80.886},
    {"driver": "Sergio Perez", "team": "Cadillac", "pos": 22, "q_time": 81.322},
]

# Antonelli, Piastri, Bearman, Colapinto and Bottas sat out FP1 for rookie runs.
FP1_TIMES = {
    "Charles Leclerc": 79.075,
    "Max Verstappen": 79.559,
    "Lewis Hamilton": 79.618,
    "Isack Hadjar": 79.997,
    "George Russell": 80.066,
    "Gabriel Bortoleto": 80.360,
    "Nico Hulkenberg": 80.623,
    "Arvid Lindblad": 80.760,
    "Liam Lawson": 80.866,
    "Lando Norris": 81.024,
    "Esteban Ocon": 81.051,
    "Fernando Alonso": 81.550,
    "Pierre Gasly": 81.704,
    "Alex Albon": 81.819,
    "Sergio Perez": 82.089,
    "Lance Stroll": 83.471,
    "Carlos Sainz": 83.734,
}

SPRINT_RESULT = []

# r1_finish = finishing position at Belgium (R10).
DRIVER_EXPERIENCE = {
    "Andrea Kimi Antonelli": {"f1_seasons": 2, "r1_finish": 1},
    "Charles Leclerc": {"f1_seasons": 8, "r1_finish": 2},
    "Max Verstappen": {"f1_seasons": 11, "r1_finish": 3},
    "Lewis Hamilton": {"f1_seasons": 19, "r1_finish": 4},
    "Oscar Piastri": {"f1_seasons": 4, "r1_finish": 5},
    "Isack Hadjar": {"f1_seasons": 2, "r1_finish": 6},
    "Lando Norris": {"f1_seasons": 7, "r1_finish": 7},
    "Gabriel Bortoleto": {"f1_seasons": 2, "r1_finish": 8},
    "Arvid Lindblad": {"f1_seasons": 0, "r1_finish": 9},
    "Franco Colapinto": {"f1_seasons": 2, "r1_finish": 10},
    "Pierre Gasly": {"f1_seasons": 9, "r1_finish": 11},
    "Liam Lawson": {"f1_seasons": 2, "r1_finish": 12},
    "Nico Hulkenberg": {"f1_seasons": 14, "r1_finish": 13},
    "Ollie Bearman": {"f1_seasons": 2, "r1_finish": 14},
    "Alex Albon": {"f1_seasons": 6, "r1_finish": 15},
    "Carlos Sainz": {"f1_seasons": 11, "r1_finish": 16},
    "Esteban Ocon": {"f1_seasons": 9, "r1_finish": 17},
    "Valtteri Bottas": {"f1_seasons": 13, "r1_finish": 18},
    "Fernando Alonso": {"f1_seasons": 23, "r1_finish": 19},
    "Lance Stroll": {"f1_seasons": 9, "r1_finish": 20},
    "Sergio Perez": {"f1_seasons": 15, "r1_finish": 21},
    "George Russell": {"f1_seasons": 7, "r1_finish": 22},
}

TEAM_PACE_DEFICIT = {
    "McLaren": 0.0,
    "Ferrari": 0.012,
    "Mercedes": 0.272,
    "Red Bull": 0.518,
    "Racing Bulls": 1.074,
    "Audi": 1.479,
    "Alpine": 1.637,
    "Haas": 2.527,
    "Aston Martin": 2.601,
    "Williams": 3.414,
    "Cadillac": 3.679,
}

START_PROCEDURE = {"Mercedes": 0.0, "Ferrari": 0.05, "McLaren": 0.05, "Red Bull": 0.0, "Alpine": 0.0, "Racing Bulls": 0.0, "Audi": -0.02, "Haas": 0.02, "Williams": -0.02, "Aston Martin": -0.05, "Cadillac": -0.08}

ENERGY_READINESS = {"Mercedes": 0.88, "McLaren": 0.85, "Ferrari": 0.8, "Red Bull": 0.78, "Racing Bulls": 0.7, "Alpine": 0.6, "Williams": 0.62, "Haas": 0.58, "Audi": 0.5, "Aston Martin": 0.45, "Cadillac": 0.4}

CIRCUIT = {"type": "balanced", "pit_loss_seconds": 20}
TYRE_COMPOUNDS = {"hardness": 0.35, "one_stop_probability": 0.55}
WEATHER = {"track_temp_c": 51, "rain_probability": 0.15}

CIRCUIT_HISTORY = {
    "Lewis Hamilton":  {"wins": 8, "podiums": 11},
    "Max Verstappen":  {"wins": 1, "podiums": 5},
    "Fernando Alonso": {"wins": 1, "podiums": 4},
    "Lando Norris":    {"wins": 1, "podiums": 3},
    "Oscar Piastri":   {"wins": 1, "podiums": 2},
    "Charles Leclerc": {"wins": 0, "podiums": 2},
    "George Russell":  {"wins": 0, "podiums": 1},
}