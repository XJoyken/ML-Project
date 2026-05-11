from __future__ import annotations

import os
from importlib import import_module
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .krisha import parse_krisha_listing_url
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


class RecommendationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)


class RecommendationResponse(BaseModel):
    model: str
    features: ExtractedRecommendationFeatures
    items: list[dict[str, Any]]


class ApartmentEvaluationRequest(BaseModel):
    url: str = Field(min_length=10, max_length=2000)


class ApartmentEvaluationResponse(BaseModel):
    source_url: str
    parsed_listing: dict[str, Any]
    evaluation: dict[str, Any]


@lru_cache
def get_feature_extractor() -> GeminiFeatureExtractor:
    return GeminiFeatureExtractor(
        model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
    )


@lru_cache
def get_recommender():
    service_module = import_module("ml_project.recommender.service")
    return service_module.KnnRecommendationService()


@lru_cache
def get_apartment_evaluator():
    service_module = import_module("ml_project.evaluation")
    return service_module.ApartmentEvaluationService()


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


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend_listings(
    request: RecommendationRequest,
    extractor: GeminiFeatureExtractor = Depends(get_feature_extractor),
    recommender=Depends(get_recommender),
) -> RecommendationResponse:
    try:
        features = extractor.extract(request.prompt)
        items = recommender.recommend(features, limit=request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Recommendation input is invalid: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Recommendation failed: {exc}") from exc

    return RecommendationResponse(
        model=extractor.model,
        features=features,
        items=items,
    )


@app.post("/apartments/evaluate", response_model=ApartmentEvaluationResponse)
def evaluate_apartment(
    request: ApartmentEvaluationRequest,
    evaluator=Depends(get_apartment_evaluator),
) -> ApartmentEvaluationResponse:
    try:
        listing = parse_krisha_listing_url(request.url)
        evaluation = evaluator.evaluate(listing)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Apartment evaluation failed: {exc}") from exc

    return ApartmentEvaluationResponse(
        source_url=request.url,
        parsed_listing=listing,
        evaluation=evaluation,
    )
