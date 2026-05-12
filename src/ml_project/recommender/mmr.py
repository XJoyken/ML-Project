from __future__ import annotations

import numpy as np


def mmr_select(
    *,
    scores: np.ndarray,
    similarity_matrix: np.ndarray,
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Greedy Maximal Marginal Relevance.

    Picks indices in [0, N) that maximize the trade-off between high `scores` and
    low similarity to already-picked items. Returns up to `k` indices.
    """
    n = len(scores)
    if n == 0:
        return []
    k = min(k, n)
    remaining = set(range(n))
    selected: list[int] = []

    first = int(np.argmax(scores))
    selected.append(first)
    remaining.discard(first)

    while remaining and len(selected) < k:
        best_idx = -1
        best_value = -float("inf")
        for idx in remaining:
            max_sim = max(similarity_matrix[idx, j] for j in selected)
            value = lambda_ * scores[idx] - (1.0 - lambda_) * max_sim
            if value > best_value:
                best_value = value
                best_idx = idx
        selected.append(best_idx)
        remaining.discard(best_idx)
    return selected


def build_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of `features` (already standardised)."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normed = features / safe_norms
    similarity = normed @ normed.T
    return np.clip(similarity, 0.0, 1.0)
