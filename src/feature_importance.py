# src/feature_importance.py
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from .data_loader import load_data


FIG_DIR = Path("reports") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = Path("models")


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def main():
    # Load data only to get feature names
    df = load_data()
    feature_names = df.drop("Class", axis=1).columns.tolist()

    model_path = MODELS_DIR / "rf_balanced.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run: python -m src.train_models"
        )

    model = joblib.load(model_path)

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Loaded model does not expose feature_importances_")

    importances = model.feature_importances_
    fi = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    # Save table
    out_csv = REPORTS_DIR / "feature_importance_rf.csv"
    fi.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot top 15
    top_n = 15
    top = fi.head(top_n).iloc[::-1]  # reverse for nicer bar plot

    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    savefig(FIG_DIR / "feature_importance_rf_top15.png")

    print("\nTop features:")
    print(fi.head(10).to_string(index=False))

    print("\n✅ Feature importance complete.")


if __name__ == "__main__":
    main()
