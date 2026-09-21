from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAIN = RESULTS / "main"
STAFFING = RESULTS / "staffing"
SIMULATION = RESULTS / "simulation"
STATE = RESULTS / "state"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"

for _path in (MAIN, STAFFING, SIMULATION, STATE, TABLES, FIGURES):
    _path.mkdir(parents=True, exist_ok=True)
