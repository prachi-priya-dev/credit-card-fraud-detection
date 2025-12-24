from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
THRESH_PATH = PROJECT_ROOT / "models" / "best_threshold.txt"

# ✅ fixed feature order your model expects
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train and save model first.")
    return joblib.load(MODEL_PATH)


def load_threshold(default=0.5):
    if THRESH_PATH.exists():
        return float(THRESH_PATH.read_text().strip())
    return float(default)


MODEL = load_model()
THRESHOLD = load_threshold()


def features_dict_to_list(features: dict) -> list[float]:
    missing = [k for k in FEATURE_ORDER if k not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return [float(features[k]) for k in FEATURE_ORDER]


def predict_one(features: dict, threshold: float | None = None) -> tuple[int, float, float]:
    used_threshold = float(threshold) if threshold is not None else float(THRESHOLD)

    feature_list = features_dict_to_list(features)
    X = pd.DataFrame([feature_list], columns=FEATURE_ORDER)

    if hasattr(MODEL, "predict_proba"):
        prob = float(MODEL.predict_proba(X)[0, 1])
    else:
        score = float(MODEL.decision_function(X)[0])
        prob = float(1 / (1 + np.exp(-score)))

    # keep within [0, 1]
    prob = max(min(prob, 1.0), 0.0)

    fraud = 1 if prob >= used_threshold else 0
    return fraud, prob, used_threshold
