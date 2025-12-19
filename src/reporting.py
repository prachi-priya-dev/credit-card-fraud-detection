# src/reporting.py
import json
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

from .data_loader import load_data
from .config import TEST_SIZE, RANDOM_STATE


REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path("models")
REPORTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved: {path}")


def metrics_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm, title, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Normal", "Fraud"])
    plt.yticks([0, 1], ["Normal", "Fraud"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    savefig(out_path)


def main():
    # 1) Load data
    df = load_data()
    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 2) Load best model
    model_path = MODELS_DIR / "rf_balanced.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run: python -m src.train_models"
        )

    model = joblib.load(model_path)

    # 3) Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # 4) PR Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision)
    plt.title(f"Precision–Recall Curve (RF Balanced) | PR-AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    savefig(FIG_DIR / "pr_curve_rf_balanced.png")

    # 5) Threshold sweep
    grid = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for thr in grid:
        m = metrics_at_threshold(y_test.values, y_prob, thr)
        precs.append(m["precision"])
        recs.append(m["recall"])
        f1s.append(m["f1"])

    plt.figure(figsize=(7, 5))
    plt.plot(grid, precs, label="Precision")
    plt.plot(grid, recs, label="Recall")
    plt.plot(grid, f1s, label="F1")
    plt.title("Precision / Recall / F1 vs Threshold (RF Balanced)")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    savefig(FIG_DIR / "threshold_tradeoff_rf_balanced.png")

    # 6) Choose best threshold by F1 (for reporting)
    best_idx = int(np.argmax(f1s))
    best_thr = float(grid[best_idx])
    best_metrics = metrics_at_threshold(y_test.values, y_prob, best_thr)

    # Also include threshold=0.5 metrics for comparison
    default_metrics = metrics_at_threshold(y_test.values, y_prob, 0.5)

    # 7) Confusion matrices
    plot_confusion_matrix(
        np.array(default_metrics["confusion_matrix"]),
        "Confusion Matrix (Threshold=0.50)",
        FIG_DIR / "confusion_matrix_thr_0_50.png",
    )

    plot_confusion_matrix(
        np.array(best_metrics["confusion_matrix"]),
        f"Confusion Matrix (Best F1 Threshold={best_thr:.2f})",
        FIG_DIR / "confusion_matrix_best_f1.png",
    )

    # 8) Save report JSON
    report = {
        "model": "rf_balanced",
        "pr_auc": float(pr_auc),
        "default_threshold": default_metrics,
        "best_f1_threshold": best_metrics,
    }

    out_json = REPORTS_DIR / "phase5_reporting_rf_balanced.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Saved: {out_json}")

    print("\n✅ Reporting complete.")
    print("Default threshold metrics:", default_metrics)
    print("Best-F1 threshold metrics:", best_metrics)


if __name__ == "__main__":
    main()
