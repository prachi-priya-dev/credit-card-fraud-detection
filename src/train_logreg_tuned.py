import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_split_data
from src.utils import save_json


def eval_at_threshold(y_true, probs, thr):
    y_pred = (probs >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    pr_auc = float(average_precision_score(y_true, probs))

    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "pr_auc": pr_auc,
    }


def main():
    X_train, X_test, y_train, y_test = load_split_data()

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    results = [eval_at_threshold(y_test, probs, t) for t in thresholds]

    best = max(results, key=lambda r: r["f1"])

    output = {
        "model": "logreg_scaled_balanced",
        "results": results,
        "best_by_f1": best,
    }

    Path("reports").mkdir(exist_ok=True)
    save_json(output, "reports/phase4_logreg_tuned.json")

    print("Saved: reports/phase4_logreg_tuned.json ✅")
    print("Best:", best)


if __name__ == "__main__":
    main()
