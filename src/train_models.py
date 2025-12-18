# src/train_models.py
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from .data_loader import load_data
from sklearn.model_selection import train_test_split
from .config import TEST_SIZE, RANDOM_STATE
from .config import RANDOM_STATE

REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


def evaluate_probs(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    pr_auc = average_precision_score(y_true, y_prob)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "pr_auc": float(pr_auc),
    }


def main():
    df = load_data()
    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {}

    # 1) Logistic Regression with class_weight balanced
    models["logreg_balanced"] = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=None,
    )

    # 2) RandomForest with class_weight balanced_subsample
    models["rf_balanced"] = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    # 3) Logistic Regression + undersampling (simple + strong baseline)
    models["logreg_undersample"] = ImbPipeline(
        steps=[
            ("under", RandomUnderSampler(random_state=RANDOM_STATE)),
            (
                "model",
                LogisticRegression(
                    max_iter=2000, random_state=RANDOM_STATE
                ),
            ),
        ]
    )

    results = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        # probability estimates
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # fallback: decision function -> sigmoid-ish scaling
            scores = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-scores))

        metrics = evaluate_probs(y_test, y_prob, threshold=0.5)
        results[name] = metrics
        print(metrics)

    out_path = REPORTS_DIR / "phase3_model_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path} ✅")


if __name__ == "__main__":
    main()
