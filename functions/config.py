from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LUTS_DIR = PROJECT_ROOT / "luts"
LUTS_DIR.mkdir(parents=True, exist_ok=True)

INSTANCES_DIR = PROJECT_ROOT / "instances"
INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tol = 1e-9