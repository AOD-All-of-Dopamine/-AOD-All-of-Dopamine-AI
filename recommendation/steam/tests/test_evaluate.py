# tests/test_evaluate.py
import math

import pandas as pd

from src.evaluate import dcg, ndcg_at_k, per_anchor_metrics, precision_at_k


def test_dcg():
    assert dcg([3, 1, 0]) == 3 + 1 / math.log2(3)


def test_ndcg_perfect():
    assert ndcg_at_k([3, 2, 1, 0], k=4) == 1.0


def test_ndcg_known_value():
    # rels [0, 3]: dcg = 3/log2(3); ideal [3, 0]: 3
    expected = (3 / math.log2(3)) / 3
    assert abs(ndcg_at_k([0, 3], k=2) - expected) < 1e-9


def test_precision_at_k():
    assert precision_at_k([3, 2, 1, 0, 2], k=5, threshold=2) == 3 / 5


def test_per_anchor_metrics():
    df = pd.DataFrame({
        "anchor_steam_appid": [1] * 3 + [2] * 3,
        "rank": [1, 2, 3, 1, 2, 3],
        "relevance": [3, 1, 0, 2, 2, 2],
        "recommendation_confidence": [2, 2, 1, 3, 3, 3],
    })
    per = per_anchor_metrics(df, k=3)
    a1 = per[per["anchor_steam_appid"] == 1].iloc[0]
    a2 = per[per["anchor_steam_appid"] == 2].iloc[0]
    assert a1["precision"] == 1 / 3
    assert a2["precision"] == 1.0
    assert a2["ndcg"] == 1.0  # 이미 내림차순
    assert a1["avg_relevance"] == 4 / 3
