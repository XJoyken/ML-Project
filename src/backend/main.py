from __future__ import annotations




import os
from importlib import import_module
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from .krisha import parse_krisha_listing_url
from .recommendation_features import (
    DEFAULT_GEMINI_MODEL,
    ExtractedRecommendationFeatures,
    GeminiFeatureExtractor,
)



app = FastAPI(title="Almaty Apartment Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeatureExtractionRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)


class FeatureExtractionResponse(BaseModel):
    model: str
    features: ExtractedRecommendationFeatures


class RecommendationRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)
    language: Literal["ru", "en"] = Field(default="ru")
    use_llm: bool | None = Field(default=None)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    prioritize_air_quality: bool = Field(default=False)


class RecommendationResponse(BaseModel):
    model: str
    features: ExtractedRecommendationFeatures
    items: list[dict[str, Any]]
    plan: dict[str, Any]
    summary: str
    source: Literal["gemini", "fallback"]
    fallback_reason_code: str | None = None
    fallback_reason: str | None = None
    matched_candidates: int


class ApartmentEvaluationRequest(BaseModel):
    url: str = Field(min_length=10, max_length=2000)
    language: Literal["ru", "en"] = Field(default="ru")
    use_llm: bool | None = Field(default=None)


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
    return service_module.RecommendationService()


@lru_cache
def get_recommendation_narrative():
    return import_module("ml_project.recommender.narrative")


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
    narrative_module=Depends(get_recommendation_narrative),
) -> RecommendationResponse:
    try:
        features = extractor.extract(request.prompt)
        result = recommender.recommend(
            features,
            limit=request.limit,
            mmr_lambda=request.mmr_lambda,
            prioritize_air_quality=request.prioritize_air_quality,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Recommendation input is invalid: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Recommendation failed: {exc}") from exc

    narrative = narrative_module.generate(
        plan_payload=result["plan"],
        items=result["items"],
        language=request.language,
        use_llm=request.use_llm,
        prioritize_air_quality=request.prioritize_air_quality,
    )
    item_explanations = {item.listing_id: item.explanation for item in narrative.items}
    item_urls = {item.listing_id: item.url for item in narrative.items if getattr(item, "url", None)}
    enriched_items = [
        {
            **item,
            "explanation": item_explanations.get(str(item.get("listing_id")), ""),
            "url": item_urls.get(str(item.get("listing_id")), f"https://krisha.kz/a/show/{item.get('listing_id')}"),
        }
        for item in result["items"]
    ]

    return RecommendationResponse(
        model=extractor.model,
        features=features,
        items=enriched_items,
        plan=result["plan"],
        summary=narrative.summary,
        source=narrative.source,
        fallback_reason_code=narrative.fallback_reason_code,
        fallback_reason=narrative.fallback_reason,
        matched_candidates=result["matched_candidates"],
    )


@app.post("/apartments/evaluate", response_model=ApartmentEvaluationResponse)
def evaluate_apartment(
    request: ApartmentEvaluationRequest,
    evaluator=Depends(get_apartment_evaluator),
) -> ApartmentEvaluationResponse:
    try:
        listing = parse_krisha_listing_url(request.url)
        evaluation = evaluator.evaluate(
            listing,
            language=request.language,
            use_llm=request.use_llm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Apartment evaluation failed: {exc}") from exc

    return ApartmentEvaluationResponse(
        source_url=request.url,
        parsed_listing=listing,
        evaluation=evaluation,
    )
