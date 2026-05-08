from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .recommendation_features import (
    DEFAULT_GEMINI_MODEL,
    ExtractedRecommendationFeatures,
    GeminiFeatureExtractor,
)

app = FastAPI(title="Almaty Apartment Recommender API")


class FeatureExtractionRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)


class FeatureExtractionResponse(BaseModel):
    model: str
    features: ExtractedRecommendationFeatures


@lru_cache
def get_feature_extractor() -> GeminiFeatureExtractor:
    return GeminiFeatureExtractor(
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommendations/features", response_model=FeatureExtractionResponse)
def extract_recommendation_features(
    request: FeatureExtractionRequest,
    extractor: GeminiFeatureExtractor = Depends(get_feature_extractor),
) -> FeatureExtractionResponse:
    try:
        features = extractor.extract(request.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Gemini returned invalid feature JSON: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini feature extraction failed: {exc}") from exc

    return FeatureExtractionResponse(model=extractor.model, features=features)
