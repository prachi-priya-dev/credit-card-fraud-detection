import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score

from .data_loader import load_data
from .metrics import compute_metrics
from .config import REPORT_DIR


def evaluate_at_threshold(y_true, y_prob, threshold: float):
    y_pred = (y_prob >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_prob)


def main():
    df = load_data()
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load the model you trained in phase 1
    import joblib
    from .config import MODEL_DIR
    model = joblib.load(MODEL_DIR / "baseline_logreg.joblib")

    y_prob = model.predict_proba(X_test)[:, 1]

    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"PR-AUC: {pr_auc:.6f}")

    # Plot PR curve
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    plt.show()

    # Evaluate at multiple thresholds
    candidate_thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

    results = {}
    for t in candidate_thresholds:
        m = evaluate_at_threshold(y_test.to_numpy(), y_prob, t)
        results[str(t)] = m
        print(f"\nThreshold = {t}")
        print(m)

    # Pick threshold that maximizes F1 (simple rule for now)
    best_t = None
    best_f1 = -1
    for t, m in results.items():
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)

    print(f"\n✅ Best threshold by F1 = {best_t} (F1={best_f1:.4f})")

    # Save results
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    with open(REPORT_DIR / "threshold_results.json", "w") as f:
        json.dump(
            {
                "pr_auc": pr_auc,
                "best_threshold_by_f1": best_t,
                "results": results,
            },
            f,
            indent=2,
        )

    print("Saved: reports/threshold_results.json ✅")


if __name__ == "__main__":
    main()
