from __future__ import annotations

import numpy as np
import pytest

from ml_project.recommender.mmr import build_similarity_matrix, mmr_select
from ml_project.recommender.scoring import (
    HARD_WEIGHT_THRESHOLD,
    categorical_score,
    hard_filter_passes,
    score_numeric,
)


def test_score_numeric_at_most_perfect_when_under():
    assert score_numeric(actual=400.0, target=500.0, preference="at_most", feature="schools_nearest_dist_m") == 1.0


def test_score_numeric_at_most_drops_linearly_when_over():
    s = score_numeric(actual=750.0, target=500.0, preference="at_most", feature="schools_nearest_dist_m")
    assert 0.49 < s < 0.51


def test_score_numeric_at_most_zero_when_far_over():
    s = score_numeric(actual=1500.0, target=500.0, preference="at_most", feature="schools_nearest_dist_m")
    assert s == 0.0


def test_score_numeric_at_least_mirror():
    s_above = score_numeric(actual=2.0, target=1.0, preference="at_least", feature="rooms")
    s_below = score_numeric(actual=0.5, target=1.0, preference="at_least", feature="rooms")
    assert s_above == 1.0
    assert 0.49 < s_below < 0.51


def test_score_numeric_equal_rooms_exact_and_off_by_one():
    assert score_numeric(actual=2.0, target=2.0, preference="equal", feature="rooms") == 1.0
    assert score_numeric(actual=3.0, target=2.0, preference="equal", feature="rooms") == 0.5
    assert score_numeric(actual=4.0, target=2.0, preference="equal", feature="rooms") == 0.0


def test_score_numeric_prefer_low_acts_like_at_most():
    a = score_numeric(actual=20.0, target=30.0, preference="prefer_low", feature="air_pm25_warm_day")
    b = score_numeric(actual=20.0, target=30.0, preference="at_most", feature="air_pm25_warm_day")
    assert a == b


def test_hard_filter_passes_at_most_with_slack():
    # 10% slack
    assert hard_filter_passes(actual=550.0, target=500.0, preference="at_most", feature="schools_nearest_dist_m")
    assert not hard_filter_passes(actual=601.0, target=500.0, preference="at_most", feature="schools_nearest_dist_m")


def test_hard_filter_passes_equal_rooms():
    assert hard_filter_passes(actual=2.0, target=2.0, preference="equal", feature="rooms")
    assert not hard_filter_passes(actual=3.0, target=2.0, preference="equal", feature="rooms")


def test_categorical_score_case_insensitive():
    assert categorical_score("Бостандыкский район", "бостандыкский район") == 1.0
    assert categorical_score("Алмалинский район", "Бостандыкский район") == 0.0
    assert categorical_score(None, "X") == 0.0


def test_mmr_picks_diverse_items():
    scores = np.array([1.0, 0.95, 0.9, 0.85, 0.8])
    # First and second are identical; third differs sharply.
    similarity = np.array(
        [
            [1.0, 0.99, 0.10, 0.20, 0.30],
            [0.99, 1.0, 0.10, 0.20, 0.30],
            [0.10, 0.10, 1.0, 0.15, 0.20],
            [0.20, 0.20, 0.15, 1.0, 0.40],
            [0.30, 0.30, 0.20, 0.40, 1.0],
        ]
    )
    chosen = mmr_select(scores=scores, similarity_matrix=similarity, k=3, lambda_=0.5)
    assert chosen[0] == 0
    # The second pick must NOT be the near-duplicate index 1
    assert chosen[1] != 1


def test_build_similarity_matrix_diagonal_is_one():
    matrix = np.array([[1.0, 2.0], [-1.0, 1.0], [3.0, -2.0]])
    sim = build_similarity_matrix(matrix)
    assert sim.shape == (3, 3)
    for i in range(3):
        assert abs(sim[i, i] - 1.0) < 1e-6


def test_hard_threshold_constant():
    assert 0.5 < HARD_WEIGHT_THRESHOLD < 1.0
