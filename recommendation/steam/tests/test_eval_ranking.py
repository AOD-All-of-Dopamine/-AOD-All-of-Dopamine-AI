# tests/test_eval_ranking.py
import pandas as pd
import pytest

from src.eval_ranking import KEY_COLS, _eval_variant, compute_idcgs
from src.metrics import LINEAR, assert_pool_coverage


def _pooled():
    return pd.DataFrame({
        "anchor_steam_appid": [1, 1, 1],
        "anchor_name": ["A", "A", "A"],
        "candidate_steam_appid": [10, 20, 30],
        "relevance": [3, 2, 0],
        "recommendation_confidence": [3, 2, 1],
    })


def _ranked(candidates):
    return pd.DataFrame({
        "anchor_steam_appid": [1] * len(candidates),
        "candidate_steam_appid": candidates,
        "rank": range(1, len(candidates) + 1),
    })


def test_eval_variant_reads_judgments_in_rank_order():
    pooled = _pooled()
    idcgs = compute_idcgs(pooled, [1], k=3, mode=LINEAR)
    res = _eval_variant(pooled, _ranked([10, 20, 30]), [1], idcgs, k=3, mode=LINEAR)
    row = res.iloc[0]
    assert row["gains"] == [3, 2, 0]
    assert row["NDCG"] == pytest.approx(1.0)  # 이미 최적 순서
    assert row["P@10"] == pytest.approx(2 / 3)


def test_eval_variant_penalizes_bad_order():
    pooled = _pooled()
    idcgs = compute_idcgs(pooled, [1], k=3, mode=LINEAR)
    res = _eval_variant(pooled, _ranked([30, 20, 10]), [1], idcgs, k=3, mode=LINEAR)
    assert res.iloc[0]["gains"] == [0, 2, 3]
    assert res.iloc[0]["NDCG"] < 0.8


def test_unjudged_candidate_fails_loudly_instead_of_scoring_zero():
    """예전에는 판정 없는 후보를 relevance 0 으로 채워 조용히 넘어갔다."""
    with pytest.raises(ValueError, match="pool 재생성"):
        assert_pool_coverage(_pooled(), _ranked([10, 99]), KEY_COLS, k=10, label="R2-new")
