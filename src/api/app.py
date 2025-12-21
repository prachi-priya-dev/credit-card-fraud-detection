from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev; later you can restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],   # IMPORTANT: includes OPTIONS
    allow_headers=["*"],
)


# ✅ project root = credit-card-fraud-detection/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "static"

# ✅ Mount UI only if folder exists (prevents RuntimeError)
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

    @app.get("/")
    def root(): # type: ignore
        return RedirectResponse(url="/ui")
else:
    @app.get("/")
    def root():
        return {"ok": True, "message": "API running. (UI folder not found)"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    fraud, prob, used_threshold = predict_one(req.features, threshold=req.threshold)
    return PredictResponse(fraud=fraud, confidence=prob, threshold=used_threshold)


@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    results = []
    for item in req.items:
        fraud, prob, used_threshold = predict_one(item.features, threshold=item.threshold)
        results.append(PredictResponse(fraud=fraud, confidence=prob, threshold=used_threshold))
    return PredictBatchResponse(results=results)

