# 💳 Credit Card Fraud Detection (Machine Learning + FastAPI)

An end-to-end machine learning project to detect **fraudulent credit card transactions** on a highly imbalanced dataset (**fraud ≈ 0.17%**).
Built to be **interview-ready**: baseline → threshold tuning → model comparison → EDA → explainability → FastAPI deployment (+ UI).

---

## 🔗 Live Demo

- **API Base URL:** https://credit-card-fraud-detection-xw8b.onrender.com
- **Interactive API Docs (Swagger):** https://credit-card-fraud-detection-xw8b.onrender.com/docs
- **UI:** https://credit-card-fraud-detection-xw8b.onrender.com/ui
- **Health Check:** https://credit-card-fraud-detection-xw8b.onrender.com/health

---

## 🚀 Highlights

- ✅ Handles **extreme class imbalance**
- ✅ Uses **PR-AUC** (more meaningful than accuracy for fraud)
- ✅ **Threshold tuning** (does not assume 0.5)
- ✅ Model comparison:
  - Logistic Regression (baseline / balanced)
  - Random Forest (balanced_subsample)
  - Logistic Regression + undersampling
- ✅ EDA + reporting plots
- ✅ Explainability via **feature importance**
- ✅ Deployed with **FastAPI + Docker** + simple UI

---

## 📂 Project Structure

```text
credit-card-fraud-detection/
  src/
    config.py
    data_loader.py
    train_baseline.py
    threshold_tuning.py
    train_models.py
    eda_report.py
    reporting.py
    api/
      app.py
      predict.py
      schemas.py
  reports/
    figures/
      class_distribution.png
      amount_distribution_log.png
      correlation_heatmap.png
      pr_curve_rf_balanced.png
      threshold_tradeoff_rf_balanced.png
      confusion_matrix_thr_0_50.png
      confusion_matrix_best_f1.png
      feature_importance_rf_top15.png
  models/
    best_model.joblib
    best_threshold.txt
  requirements.txt
  Dockerfile
  README.md
```
---

## 📊 Key Outputs

This project generates production-quality artifacts:

Class distribution visualization

Amount distribution comparison (log scale)

Correlation heatmap

Precision–Recall curve

Threshold tradeoff plot

Confusion matrices

Feature importance (Random Forest)

--- 

*Plots*
Class Distribution
![Class Distribution](reports/figures/class_distribution.png)

Amount Distribution (Log Scale)
![Amount Distribution](reports/figures/amount_distribution_log.png)

Correlation Heatmap
![Correlation Heatmap](reports/figures/correlation_heatmap.png)

Precision–Recall Curve (RF Balanced)
![Precision–Recall Curve](reports/figures/pr_curve_rf_balanced.png)

Threshold Tradeoff
![Threshold Tradeoff](reports/figures/threshold_tradeoff_rf_balanced.png)

Confusion Matrix (thr=0.50)
![Confusion Matrix](reports/figures/confusion_matrix_thr_0_50.png)

Confusion Matrix (Best F1 threshold)
![Confusion Matrix](reports/figures/confusion_matrix_best_f1.png)

Feature Importance (Random Forest)
![Feature Importance](reports/figures/feature_importance_rf_top15.png)

---

## 🧪 How to Run

### 1) Setup environment

python -m venv .venv

*source .venv/bin/activate*                  # macOS/Linux

*.venv\Scripts\activate*                     # Windows

pip install -r requirements.txt

### 2) Add dataset

Place the dataset file at:

data/raw/creditcard.csv

### 3) Run pipeline

python -m src.train_baseline

python -m src.threshold_tuning

python -m src.train_models

python -m src.eda_report

python -m src.reporting

---
## 🧠 API Usage (Deployed)

✅ Health check
```text
curl "https://credit-card-fraud-detection-xw8b.onrender.com/health"
```

✅ Predict (single transaction)

```text
Note: /predict is POST only. Opening it in the browser does a GET and returns 405 Method Not Allowed (expected).

curl -X POST "https://credit-card-fraud-detection-xw8b.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Time": 0,
      "V1": -1.359807,
      "V2": -0.072781,
      "V3": 2.536347,
      "V4": 1.378155,
      "V5": -0.338321,
      "V6": 0.462388,
      "V7": 0.239599,
      "V8": 0.098698,
      "V9": 0.363787,
      "V10": 0.090794,
      "V11": -0.5516,
      "V12": -0.6178,
      "V13": -0.9913,
      "V14": -0.3112,
      "V15": 1.4682,
      "V16": -0.4704,
      "V17": 0.2079,
      "V18": 0.0258,
      "V19": 0.4039,
      "V20": 0.2514,
      "V21": -0.0183,
      "V22": 0.2778,
      "V23": -0.1104,
      "V24": 0.0669,
      "V25": 0.1285,
      "V26": -0.1891,
      "V27": 0.1335,
      "V28": -0.0211,
      "Amount": 149.62
    }
  }'
```

``` text 
Response

{
  "fraud": 0,
  "confidence": 0.02,
  "threshold": 0.5
}
```
(confidence value depends on the model output)

---

## 📌 Notes on Evaluation

Fraud datasets are imbalanced, so:

Accuracy can be misleading

Precision/Recall tradeoff is critical

We use PR-AUC and threshold tuning to control false positives vs missed fraud


## 🐳 Docker (Optional Local Run)

```text 
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

Then open:

*http://127.0.0.1:8000/ui*

*http://127.0.0.1:8000/docs*

---