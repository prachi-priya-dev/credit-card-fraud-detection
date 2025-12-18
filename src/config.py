from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# src/config.py
RANDOM_STATE = 42
TEST_SIZE = 0.2

# optional: how many thresholds to evaluate
THRESHOLDS_TO_CHECK = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
