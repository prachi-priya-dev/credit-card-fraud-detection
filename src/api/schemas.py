from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class PredictRequest(BaseModel):
    # expects 30 feature values (Time, V1..V28, Amount) in correct order
    features: List[float] = Field(..., min_length=30, max_length=30)


class PredictResponse(BaseModel):
    fraud: int
    confidence: float
    threshold: float


class PredictBatchRequest(BaseModel):
    items: List[PredictRequest] = Field(..., min_length=1)


class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]
