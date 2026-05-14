"""Offline Precision@K evaluation for the recommender.

For each query we have an explicit relevance rule (exact rooms / price cap /
POI distance cap / district / floor flags). We run the recommender end-to-end
(Gemini feature extraction + scoring + MMR) and check what fraction of the
top-K results actually satisfy the rule. The output is aggregated as
mean/median Precision@K across queries plus the top failures.

Why offline: query→relevance pairs are hand-curated, so the metric isolates
the recommender pipeline from real-world user-feedback noise. This is the same
shape as a typical RecSys benchmark (NDCG/MAP/HitRate are easy follow-ons).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Polite delay between Gemini calls to avoid rate limits.
INTER_QUERY_SLEEP_S = 1.5
# Retry transient 503/UNAVAILABLE/quota errors before giving up on a query.
MAX_RETRIES = 3
RETRY_BACKOFF_S = 6.0


def _extract_with_retries(extractor: Any, prompt: str) -> Any:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return extractor.extract(prompt)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            transient = ("503" in msg or "unavailable" in msg or "overloaded" in msg
                         or "rate" in msg or "quota" in msg or "429" in msg)
            if not transient or attempt == MAX_RETRIES - 1:
                raise
            print(f"  retry {attempt + 1}/{MAX_RETRIES} after {RETRY_BACKOFF_S}s: {exc}", file=sys.stderr)
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError("extract() failed without raising")


@dataclass(slots=True)
class QueryResult:
    query_id: str
    prompt: str
    k: int
    relevance_rule: dict[str, Any]
    returned_ids: list[str]
    relevant_ids: list[str]
    precision: float
    hit: bool
    candidates_passing_rule: int           # how many listings in the whole index satisfy the rule
    unmapped_preferences: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AggregateResult:
    k: int
    n_queries: int
    mean_precision_at_k: float
    median_precision_at_k: float
    hit_rate_at_k: float
    per_query: list[QueryResult]


def _matches_rule(row: pd.Series, rule: dict[str, Any]) -> bool:
    """Apply the relevance rule to a single feature-frame row."""
    for key, target in rule.items():
        if key.endswith("_max"):
            feature = key[:-4]
            value = row.get(feature)
            if value is None or pd.isna(value) or float(value) > float(target):
                return False
        elif key.endswith("_min"):
            feature = key[:-4]
            value = row.get(feature)
            if value is None or pd.isna(value) or float(value) < float(target):
                return False
        elif key == "district":
            actual = row.get("district")
            if actual is None or pd.isna(actual):
                return False
            if str(actual).strip().lower() != str(target).strip().lower():
                return False
        else:
            value = row.get(key)
            if value is None or pd.isna(value):
                return False
            # Booleans / ints: compare rounded
            if round(float(value)) != round(float(target)):
                return False
    return True


def _all_listing_ids_passing(features: pd.DataFrame, listings: pd.DataFrame, rule: dict[str, Any]) -> list[str]:
    mask = features.apply(lambda r: _matches_rule(r, rule), axis=1)
    return listings.loc[mask.values, "listing_id"].astype(str).tolist()


def evaluate_queries(
    queries: list[dict[str, Any]],
    *,
    k: int = 10,
    extractor: Any | None = None,
    recommender: Any | None = None,
    pre_extracted: dict[str, Any] | None = None,
) -> AggregateResult:
    """Run each query through the full recommender stack and score it.

    `extractor` and `recommender` are injected so tests can stub them.
    `pre_extracted` lets the caller skip the Gemini extraction step when the
    same prompts will be reused across multiple K values — provide a dict
    keyed by `query_id` returning the already-extracted features object.
    """
    if extractor is None or recommender is None:
        # Import lazily — these load heavy models/keys.
        from backend.recommendation_features import GeminiFeatureExtractor
        from ml_project.recommender.service import RecommendationService

        extractor = extractor or GeminiFeatureExtractor()
        recommender = recommender or RecommendationService()

    features = recommender.features
    listings = recommender.listings

    per_query: list[QueryResult] = []
    features_by_id_idx = listings["listing_id"].astype(str).to_list()

    for q_idx, q in enumerate(queries):
        prompt = q["prompt"]
        rule = q["relevance"]
        print(f"[{q_idx + 1}/{len(queries)}] {q['id']} k={k}: {prompt}", file=sys.stderr)

        try:
            extracted = (pre_extracted or {}).get(q["id"])
            if extracted is None:
                extracted = _extract_with_retries(extractor, prompt)
                time.sleep(INTER_QUERY_SLEEP_S)
            result = recommender.recommend(extracted, limit=k, mmr_lambda=0.7, prioritize_air_quality=False)
        except Exception as exc:
            # Skip query but keep going so we still get aggregate over the survivors.
            print(f"  FAILED ({exc}); skipping", file=sys.stderr)
            per_query.append(QueryResult(
                query_id=q["id"], prompt=prompt, k=k, relevance_rule=rule,
                returned_ids=[], relevant_ids=[], precision=0.0, hit=False,
                candidates_passing_rule=0, unmapped_preferences=[f"__error__: {exc}"],
            ))
            continue

        items = result["items"]
        relevant_in_topk: list[str] = []
        returned_ids: list[str] = []
        for item in items:
            lid = str(item.get("listing_id"))
            returned_ids.append(lid)
            try:
                pos = features_by_id_idx.index(lid)
            except ValueError:
                continue
            feature_row = features.iloc[pos]
            if _matches_rule(feature_row, rule):
                relevant_in_topk.append(lid)

        precision = len(relevant_in_topk) / k if k > 0 else 0.0
        candidates_passing = _all_listing_ids_passing(features, listings, rule)

        per_query.append(QueryResult(
            query_id=q["id"],
            prompt=prompt,
            k=k,
            relevance_rule=rule,
            returned_ids=returned_ids,
            relevant_ids=relevant_in_topk,
            precision=precision,
            hit=len(relevant_in_topk) > 0,
            candidates_passing_rule=len(candidates_passing),
            unmapped_preferences=list(getattr(extracted, "unmapped_preferences", []) or []),
        ))

    precisions = [q.precision for q in per_query]
    return AggregateResult(
        k=k,
        n_queries=len(per_query),
        mean_precision_at_k=statistics.fmean(precisions) if precisions else 0.0,
        median_precision_at_k=statistics.median(precisions) if precisions else 0.0,
        hit_rate_at_k=sum(1 for q in per_query if q.hit) / len(per_query) if per_query else 0.0,
        per_query=per_query,
    )


def run_eval(
    *,
    queries_path: Path,
    output_path: Path,
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Entry point used by the CLI. Reads queries, runs eval at each K, writes JSON."""
    # Load .env so GEMINI_API_KEY is available when the CLI runs outside the FastAPI process.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = payload["queries"]

    from backend.recommendation_features import GeminiFeatureExtractor
    from ml_project.recommender.service import RecommendationService

    extractor = GeminiFeatureExtractor()
    recommender = RecommendationService()

    # Extract features once per prompt so multiple K values share the same Gemini calls.
    print(f"Extracting features for {len(queries)} prompts...", file=sys.stderr)
    pre_extracted: dict[str, Any] = {}
    for q_idx, q in enumerate(queries):
        print(f"  [{q_idx + 1}/{len(queries)}] {q['id']}", file=sys.stderr)
        try:
            pre_extracted[q["id"]] = _extract_with_retries(extractor, q["prompt"])
        except Exception as exc:
            print(f"    FAILED: {exc}", file=sys.stderr)
        time.sleep(INTER_QUERY_SLEEP_S)

    results_by_k: dict[str, Any] = {}
    for k in k_values:
        agg = evaluate_queries(
            queries, k=k, extractor=extractor, recommender=recommender,
            pre_extracted=pre_extracted,
        )
        results_by_k[f"k_{k}"] = {
            "k": agg.k,
            "n_queries": agg.n_queries,
            "mean_precision": round(agg.mean_precision_at_k, 4),
            "median_precision": round(agg.median_precision_at_k, 4),
            "hit_rate": round(agg.hit_rate_at_k, 4),
            "per_query": [
                {
                    "id": q.query_id,
                    "prompt": q.prompt,
                    "precision": round(q.precision, 4),
                    "hit": q.hit,
                    "relevant_count": len(q.relevant_ids),
                    "returned_count": len(q.returned_ids),
                    "candidates_passing_rule": q.candidates_passing_rule,
                    "unmapped_preferences": q.unmapped_preferences,
                }
                for q in agg.per_query
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_by_k, ensure_ascii=False, indent=2), encoding="utf-8")
    return results_by_k


__all__ = ["AggregateResult", "QueryResult", "evaluate_queries", "run_eval"]
