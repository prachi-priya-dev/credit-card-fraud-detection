from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# ✅ Change this to the model you want to serve
SOURCE_MODEL = MODELS_DIR / "rf_balanced.joblib"

TARGET_MODEL = MODELS_DIR / "best_model.joblib"
TARGET_THRESH = MODELS_DIR / "best_threshold.txt"

BEST_THRESHOLD = 0.5  # change later if you tune

def main():
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(f"Source model not found: {SOURCE_MODEL}")

    model = joblib.load(SOURCE_MODEL)
    joblib.dump(model, TARGET_MODEL)

    TARGET_THRESH.write_text(str(BEST_THRESHOLD))

    print(f"✅ Exported: {SOURCE_MODEL.name} -> {TARGET_MODEL.name}")
    print(f"✅ Saved threshold: {TARGET_THRESH} = {BEST_THRESHOLD}")

if __name__ == "__main__":
    main()
