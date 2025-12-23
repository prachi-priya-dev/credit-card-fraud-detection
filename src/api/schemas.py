from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

# Model expects these exact 30 keys
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
FEATURE_SET = set(FEATURE_ORDER)


class PredictRequest(BaseModel):
    features: Dict[str, float]
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_features(self):
        keys = set(self.features.keys())

        missing = sorted(list(FEATURE_SET - keys))
        extra = sorted(list(keys - FEATURE_SET))

        if missing:
            raise ValueError(f"Missing required features: {missing}")
        if extra:
            raise ValueError(f"Unexpected feature keys: {extra}")

        return self


class PredictResponse(BaseModel):
    fraud: int
    confidence: float
    threshold: float


class PredictBatchRequest(BaseModel):
    items: List[PredictRequest] = Field(min_length=1, max_length=1000)


class PredictBatchResponse(BaseModel):
    results: List[PredictResponse]
