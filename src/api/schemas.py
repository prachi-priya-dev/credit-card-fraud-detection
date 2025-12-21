from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictRequest(BaseModel):
    # frontend sends a dict of named features
    features: Dict[str, float] = Field(..., description="Feature map: Time, V1..V28, Amount")
    # allow UI to override threshold
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    fraud: int
    confidence: float
    threshold: float


class PredictBatchRequest(BaseModel):
    items: List[PredictRequest] = Field(..., min_length=1)


class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]
