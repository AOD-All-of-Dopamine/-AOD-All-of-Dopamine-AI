# tests/test_metrics.py
import math

import pandas as pd
import pytest

from src.metrics import (
    EXPONENTIAL,
    LINEAR,
    assert_pool_coverage,
    dcg,
    gain,
    ndcg_at_k,
    precision_at_k,
)


def test_gain_modes():
    assert gain(3, LINEAR) == 3.0
    assert gain(3, EXPONENTIAL) == 7.0
    assert gain(0, EXPONENTIAL) == 0.0


def test_gain_rejects_unknown_mode():
    with pytest.raises(ValueError, match="ndcg_gain"):
        gain(2, "quadratic")


def test_dcg_differs_by_mode():
    # 두 모드가 실제로 다른 값을 낸다 — 이게 섞여 있으면 S1/P1 비교가 성립하지 않는다
    assert dcg([3, 1, 0], LINEAR) == 3 + 1 / math.log2(3)
    assert dcg([3, 1, 0], EXPONENTIAL) == 7 + 1 / math.log2(3)


def test_ndcg_perfect_order_is_one_in_both_modes():
    for mode in (LINEAR, EXPONENTIAL):
        assert ndcg_at_k([3, 2, 1, 0], k=4, mode=mode) == 1.0


def test_ndcg_uses_external_ideal_pool():
    """pooled 평가: 분모는 랭킹이 아니라 판정 풀 전체에서 온다."""
    rels, pool = [1, 1], [3, 3, 1, 1]
    self_ideal = ndcg_at_k(rels, k=2, mode=EXPONENTIAL)
    pooled = ndcg_at_k(rels, k=2, ideal_pool=pool, mode=EXPONENTIAL)
    assert self_ideal == 1.0  # 자기 기준으로는 완벽해 보이지만
    assert pooled < 0.2  # 풀 전체 기준으로는 나쁜 랭킹이다


def test_ndcg_zero_pool_is_zero_not_error():
    assert ndcg_at_k([0, 0], k=2, ideal_pool=[0, 0], mode=EXPONENTIAL) == 0.0


def test_precision_at_k():
    assert precision_at_k([3, 2, 1, 0, 2], k=5, threshold=2) == 3 / 5
    assert precision_at_k([], k=5) == 0.0


def _judged(pairs):
    return pd.DataFrame(pairs, columns=["anchor_steam_appid", "candidate_steam_appid"])


def test_assert_pool_coverage_passes_when_complete():
    judged = _judged([(1, 10), (1, 20), (2, 30)])
    ranked = pd.DataFrame({
        "anchor_steam_appid": [1, 1, 2],
        "candidate_steam_appid": [10, 20, 30],
        "rank": [1, 2, 1],
    })
    n = assert_pool_coverage(
        judged, ranked, ["anchor_steam_appid", "candidate_steam_appid"], k=10
    )
    assert n == 3


def test_assert_pool_coverage_fails_on_missing_pair():
    judged = _judged([(1, 10)])
    ranked = pd.DataFrame({
        "anchor_steam_appid": [1, 1],
        "candidate_steam_appid": [10, 99],
        "rank": [1, 2],
    })
    with pytest.raises(ValueError, match="pool 재생성"):
        assert_pool_coverage(
            judged, ranked, ["anchor_steam_appid", "candidate_steam_appid"], k=10, label="R2"
        )


def test_assert_pool_coverage_ignores_below_k():
    """Top-k 밖의 미판정은 문제가 아니다."""
    judged = _judged([(1, 10)])
    ranked = pd.DataFrame({
        "anchor_steam_appid": [1, 1],
        "candidate_steam_appid": [10, 99],
        "rank": [1, 11],
    })
    assert assert_pool_coverage(
        judged, ranked, ["anchor_steam_appid", "candidate_steam_appid"], k=10
    ) == 1


def test_assert_pool_coverage_normalizes_int_float_keys():
    """판정 파일이 float(NaN 때문), 랭킹이 int64 여도 같은 쌍으로 인식돼야 한다."""
    judged = pd.DataFrame({
        "anchor_steam_appid": [1.0],
        "candidate_steam_appid": [10.0],
    })
    ranked = pd.DataFrame({
        "anchor_steam_appid": [1],
        "candidate_steam_appid": [10],
        "rank": [1],
    })
    assert assert_pool_coverage(
        judged, ranked, ["anchor_steam_appid", "candidate_steam_appid"], k=10
    ) == 1
