import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .data_loader import load_data
from .metrics import compute_metrics
from .config import MODEL_DIR, REPORT_DIR

def main():
    df = load_data()

    print("Shape:", df.shape)
    fraud_ratio = df["Class"].mean()
    print(f"Fraud ratio: {fraud_ratio:.4f} ({fraud_ratio*100:.2f}%)")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print("Baseline metrics:", metrics)

    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    joblib.dump(pipeline, MODEL_DIR / "baseline_logreg.joblib")
    with open(REPORT_DIR / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model + metrics ✅")

if __name__ == "__main__":
    main()
