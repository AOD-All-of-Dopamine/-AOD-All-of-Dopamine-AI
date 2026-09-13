# tests/test_trend_integration.py
"""트렌드가 개인화 랭킹에 붙는 방식 — 곱셈 보정이라 유사도가 주항으로 남아야 한다.

`src/trend/trend_ranker.py` 에 결합 공식이 있었지만 개인화 경로는 한 번도 쓰지 않았고,
`trend_features.parquet` 도 구 코퍼스(19,476행)였다.
"""
import numpy as np
import pandas as pd
import pytest

from src.personalization.personalized_ranker import PersonalizedRanker


@pytest.fixture
def artifacts(tmp_path):
    pd.DataFrame({
        "steam_appid": [1, 2, 3],
        "name": ["유사함/무명", "덜유사함/신작", "덜유사함/구작"],
        "recommendations_total": [500, 500, 500],
    }).to_parquet(tmp_path / "dataset.parquet", index=False)
    return tmp_path


@pytest.fixture
def trend(tmp_path):
    d = tmp_path / "trend"
    d.mkdir()
    pd.DataFrame({"steam_appid": [1, 2, 3], "trend_signal": [0.0, 0.09, 0.0]}).to_parquet(
        d / "trend_features.parquet", index=False)
    return d


def _cands(sims):
    return pd.DataFrame({"steam_appid": [1, 2, 3], "seed_similarity": sims})


def test_trend_is_off_by_default(artifacts):
    r = PersonalizedRanker(rec_boost=0.0, artifacts=artifacts)
    out = r.rank(_cands([0.80, 0.79, 0.78]))
    assert out["steam_appid"].tolist() == [1, 2, 3]
    assert "trend_norm" not in out.columns


def test_trend_can_reorder_a_near_tie(artifacts, trend):
    """유사도가 거의 같으면 신작이 앞선다 — 트렌드 보정이 하는 일이다."""
    r = PersonalizedRanker(rec_boost=0.0, artifacts=artifacts, trend_weight=0.10, trend_dir=trend)
    out = r.rank(_cands([0.800, 0.799, 0.798]))
    assert out["steam_appid"].tolist()[0] == 2


def test_similarity_still_dominates(artifacts, trend):
    """개인화가 가장 중요하다 — 트렌드가 유사도 격차를 뒤집으면 안 된다."""
    r = PersonalizedRanker(rec_boost=0.0, artifacts=artifacts, trend_weight=0.10, trend_dir=trend)
    out = r.rank(_cands([0.90, 0.70, 0.60]))
    assert out["steam_appid"].tolist() == [1, 2, 3]


def test_trend_signal_is_normalised_to_unit_range(artifacts, trend):
    """원값 최대가 0.0915 라 weight 를 그대로 쓰면 기여가 0.09% 다 — 정규화해야 뜻이 통한다."""
    r = PersonalizedRanker(rec_boost=0.0, artifacts=artifacts, trend_weight=0.10, trend_dir=trend)
    out = r.rank(_cands([0.5, 0.5, 0.5])).set_index("steam_appid")
    assert out.loc[2, "trend_norm"] == pytest.approx(1.0)
    assert out.loc[2, "final_score"] / out.loc[1, "final_score"] == pytest.approx(1.10)


def test_unknown_appid_gets_zero_trend(artifacts, trend):
    r = PersonalizedRanker(rec_boost=0.0, artifacts=artifacts, trend_weight=0.10, trend_dir=trend)
    out = r.rank(pd.DataFrame({"steam_appid": [1, 99], "seed_similarity": [0.5, 0.5]}))
    assert out.set_index("steam_appid").loc[99, "trend_norm"] == 0.0


def test_missing_trend_artifact_fails_loudly(artifacts, tmp_path):
    """조용히 0 으로 돌면 '트렌드를 켰다'고 믿으면서 아무 일도 안 일어난다."""
    with pytest.raises(SystemExit, match="trend_features"):
        PersonalizedRanker(rec_boost=0.0, artifacts=artifacts,
                           trend_weight=0.1, trend_dir=tmp_path / "없음")


def test_rec_boost_and_trend_add_up(artifacts, trend):
    r = PersonalizedRanker(rec_boost=0.20, artifacts=artifacts, trend_weight=0.10, trend_dir=trend)
    out = r.rank(_cands([0.5, 0.5, 0.5])).set_index("steam_appid")
    # 세 게임의 리뷰가 같으므로 rec_percentile 도 같다 → 차이는 트렌드에서만 온다
    assert out.loc[2, "final_score"] > out.loc[1, "final_score"]
