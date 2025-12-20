from pathlib import Path
import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
THRESH_PATH = PROJECT_ROOT / "models" / "best_threshold.txt"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train and save model first."
        )
    return joblib.load(MODEL_PATH)


def load_threshold(default=0.5):
    if THRESH_PATH.exists():
        return float(THRESH_PATH.read_text().strip())
    return float(default)


MODEL = load_model()
THRESHOLD = load_threshold()


def predict_one(features: list[float]) -> tuple[int, float]:
    # model expects shape (1, 30)
    X = np.array(features, dtype=float).reshape(1, -1)

    if hasattr(MODEL, "predict_proba"):
        prob = float(MODEL.predict_proba(X)[0, 1])
    else:
        # if model gives scores, convert to pseudo-probability
        score = float(MODEL.decision_function(X)[0])
        prob = float(1 / (1 + np.exp(-score)))

    fraud = 1 if prob >= THRESHOLD else 0
    return fraud, prob
