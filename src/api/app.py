from fastapi import FastAPI
from .schemas import (
    PredictRequest,
    PredictResponse,
    PredictBatchRequest,
    PredictBatchResponse,
)
from .predict import predict_one, THRESHOLD

app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="Predict fraud probability from credit card transaction features.",
)
@app.get("/")
def root():
    return {
        "message": "Credit Card Fraud Detection API is running ✅",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "predict_batch": "/predict_batch",
    }



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    fraud, prob = predict_one(req.features)
    return PredictResponse(fraud=fraud, confidence=prob, threshold=THRESHOLD)


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    results = []
    for item in req.items:
        fraud, prob = predict_one(item.features)
        results.append(
            PredictResponse(fraud=fraud, confidence=prob, threshold=THRESHOLD)
        )
    return PredictBatchResponse(results=results)
