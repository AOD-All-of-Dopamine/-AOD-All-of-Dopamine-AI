"""인기도 백분위 — 결측과 0 을 섞지 않는지 지킨다.

Steam 에서 `fillna(0).rank(pct=True)` 가 미보고 87%를 한 동점으로 눕혀
인기도 보정을 이진 플래그로 만든 사례가 있다. 같은 실수를 이 도메인에서
반복하지 않도록 계약을 테스트로 고정한다.
"""
import pandas as pd
import pytest

from src.personalization.personalized_ranker import PersonalizedRanker


def _ranker(interest_counts, tmp_path, pop_boost=0.15):
    """dataset.parquet 만 있으면 되는 최소 아티팩트로 랭커를 만든다."""
    df = pd.DataFrame({
        "item_id": list(range(1, len(interest_counts) + 1)),
        "interest_count": interest_counts,
    })
    d = tmp_path / "wn_test"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "dataset.parquet", index=False)
    return PersonalizedRanker(pop_boost=pop_boost, artifacts=str(d))


def _candidates(ids):
    return pd.DataFrame({
        "item_id": ids,
        "seed_similarity": [0.5] * len(ids),
    })


def test_missing_interest_lands_at_the_bottom(tmp_path):
    """결측은 관측값 순위에 끼지 않고 0.0 을 받는다.

    예전 구현은 결측을 0으로 눕힌 뒤 함께 순위를 매겨, 결측이 동점 그룹의 평균
    순위(중간값)를 받았다. 그러면 관측된 작품끼리의 해상도가 그만큼 깎인다.
    """
    r = _ranker([None, 100, 1_000, 10_000], tmp_path)
    out = r.rank(_candidates([1, 2, 3, 4])).set_index("item_id")

    assert out.loc[1, "interest_percentile"] == 0.0          # 결측 = 바닥
    # 관측된 셋은 서로 다른 값을 갖고 단조 증가한다
    obs = [out.loc[i, "interest_percentile"] for i in (2, 3, 4)]
    assert obs == sorted(obs)
    assert len(set(obs)) == 3
    assert obs[0] > 0.0                                       # 결측보다는 위


def test_real_zero_is_not_treated_as_missing(tmp_path):
    """관심 수 0 은 관측값이다 — 결측과 달리 순위에 참여한다.

    웹소설은 실제로 관심 0 인 작품이 있다(파일럿 378건 중 103건). 그것들은
    바닥이 맞지만, '측정하지 못했다'와 '측정했고 0이다'는 다른 사실이다.
    """
    r = _ranker([0, 0, 500, 5_000], tmp_path)
    out = r.rank(_candidates([1, 2, 3, 4])).set_index("item_id")

    # 0 끼리는 동점이므로 같은 백분위를 갖고, 0 보다 크다(순위에 참여했다는 뜻)
    assert out.loc[1, "interest_percentile"] == out.loc[2, "interest_percentile"]
    assert out.loc[1, "interest_percentile"] > 0.0


def test_percentile_keeps_resolution_across_the_observed_range(tmp_path):
    """관측 구간 전체에 해상도가 남아야 한다.

    Steam 결함의 증상이 이것이었다 — 리뷰 101개(0.879)와 520만개(1.000)가
    폭 0.12 안에 들어가 사실상 구분이 사라졌다.
    """
    counts = [None] * 87 + [100 * (i + 1) for i in range(13)]   # Steam 과 같은 87:13 비율
    r = _ranker(counts, tmp_path)
    ids = list(range(88, 101))                                   # 관측된 13개
    out = r.rank(_candidates(ids))["interest_percentile"]

    assert out.min() > 0.0
    assert out.max() == pytest.approx(1.0)
    assert out.max() - out.min() > 0.5      # 결함 구현에서는 0.12 였다


def test_boost_is_multiplicative_on_similarity(tmp_path):
    """유사하지 않은 작품은 인기만으로 올라오지 못한다 — 곱셈 구조 유지."""
    r = _ranker([10, 10_000], tmp_path, pop_boost=0.15)
    cand = pd.DataFrame({"item_id": [1, 2], "seed_similarity": [0.9, 0.5]})
    out = r.rank(cand).set_index("item_id")

    # 2번이 훨씬 인기 있지만 유사도가 낮아 1번을 넘지 못한다
    assert out.loc[1, "final_score"] > out.loc[2, "final_score"]
