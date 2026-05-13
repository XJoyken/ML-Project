from __future__ import annotations

import os
from importlib import import_module
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

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


class InvestmentParamsOverride(BaseModel):
    vacancy_rate: float | None = Field(default=0.08, ge=0.0, le=0.9)
    repair_cost_pct: float | None = Field(default=0.05, ge=0.0, le=0.5)
    agent_commission_months: float | None = Field(default=0.5, ge=0.0, le=3.0)
    tenant_turnover_years: float | None = Field(default=1.0, ge=0.5, le=10.0)
    maintenance_pct: float | None = Field(default=0.05, ge=0.0, le=0.3)
    property_tax_pct: float | None = Field(default=0.003, ge=0.0, le=0.05)
    discount_rate_pct: float | None = Field(default=0.153, ge=0.0, le=50.0) #Current inflation + 3%
    horizon_years: int | None = Field(default=10, ge=1, le=30)


class InvestmentRequest(BaseModel):
    url: str = Field(min_length=10, max_length=2000)
    language: Literal["ru", "en"] = Field(default="ru")
    use_llm: bool | None = Field(default=None)
    investment_params: InvestmentParamsOverride | None = Field(default=None)


class InvestmentResponse(BaseModel):
    source_url: str
    parsed_listing: dict[str, Any]
    sale_evaluation: dict[str, Any]
    rent_evaluation: dict[str, Any]
    investment: dict[str, Any]
    market_summary: dict[str, Any]
    narrative: dict[str, Any]


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


@lru_cache
def get_rent_evaluator():
    service_module = import_module("ml_project.rent.evaluation")
    return service_module.RentEvaluationService()


@lru_cache
def get_investment_narrative_module():
    return import_module("ml_project.rent.narrative")


@lru_cache
def get_macro_catalogs():
    macro = import_module("ml_project.macro")
    inflation = macro.load_inflation_catalog()
    market = macro.load_market_catalog()
    return inflation, market, macro.summarise(inflation, market)


def get_investment_module():
    return import_module("ml_project.rent.investment")


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


@app.post("/apartments/investment", response_model=InvestmentResponse)
def evaluate_investment(
    request: InvestmentRequest,
    sale_evaluator=Depends(get_apartment_evaluator),
    rent_evaluator=Depends(get_rent_evaluator),
    narrative_module=Depends(get_investment_narrative_module),
    macro_catalogs=Depends(get_macro_catalogs),
) -> InvestmentResponse:
    investment_module = get_investment_module()
    inflation, market, market_summary = macro_catalogs

    try:
        listing = parse_krisha_listing_url(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Krisha parse failed: {exc}") from exc

    try:
        sale_evaluation = sale_evaluator.evaluate(
            listing,
            language=request.language,
            use_llm=False,  # the investment narrative carries its own LLM text
        )
        rent_evaluation = rent_evaluator.evaluate(listing)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Inference failed: {exc}") from exc

    override = request.investment_params or InvestmentParamsOverride()
    params = investment_module.InvestmentParams(
        vacancy_rate=override.vacancy_rate,
        repair_cost_pct=override.repair_cost_pct,
        agent_commission_months=override.agent_commission_months,
        tenant_turnover_years=override.tenant_turnover_years,
        maintenance_pct=override.maintenance_pct,
        property_tax_pct=override.property_tax_pct,
        discount_rate_pct=override.discount_rate_pct,
        horizon_years=override.horizon_years,
    )
    investment = investment_module.compute_investment_metrics(
        sale_price_kzt=float(sale_evaluation["listing_price_kzt"]),
        monthly_rent_kzt=float(rent_evaluation["predicted_rent_kzt"]),
        params=params,
        inflation=inflation,
        market=market,
    ).as_dict()

    narrative = narrative_module.generate(
        investment=investment,
        sale_evaluation=sale_evaluation,
        rent_evaluation=rent_evaluation,
        market_summary=market_summary,
        language=request.language,
        use_llm=request.use_llm,
    )

    return InvestmentResponse(
        source_url=request.url,
        parsed_listing=listing,
        sale_evaluation=sale_evaluation,
        rent_evaluation=rent_evaluation,
        investment=investment,
        market_summary=market_summary,
        narrative=narrative.model_dump(),
    )
