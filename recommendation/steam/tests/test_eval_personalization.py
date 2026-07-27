# tests/test_eval_personalization.py
"""P1 평가 회귀 테스트.

가장 중요한 것은 `test_reproduces_frozen_summary_*` 다. 동결된 P1 베이스라인
(`artifacts/p1_review/p1_llm_proxy_evaluation_summary.json`)은 한동안 리포 안의 어떤
코드로도 재생성되지 않았다 — 평가 함수가 프로필 단위로 집계하지 않았기 때문이다.
이 테스트가 그 숫자를 코드에 묶어둔다.
"""
import numpy as np
import pandas as pd
import pytest

from src.eval_personalization import evaluate_from_judgments, load_judged_split
from src.metrics import EXPONENTIAL, LINEAR

# summary.json 의 공식 수치. 원본은 linear gain + 프로필별 pooled IDCG 로 계산됐다.
FROZEN = {
    "dev": {
        "profiles": 12,
        "MAX": {"NDCG": 0.8188, "P@10": 0.8083, "Conf@10": 2.0917},
        "MEAN": {"NDCG": 0.7790, "P@10": 0.7833, "Conf@10": 2.0333},
        "TOP2_MEAN": {"NDCG": 0.7728, "P@10": 0.7833, "Conf@10": 1.9917},
    },
    "val": {
        "profiles": 8,
        "MAX": {"NDCG": 0.8126, "P@10": 0.8000, "Conf@10": 2.2125},
        "MEAN": {"NDCG": 0.8507, "P@10": 0.9000, "Conf@10": 2.2875},
        "TOP2_MEAN": {"NDCG": 0.8712, "P@10": 0.8875, "Conf@10": 2.3625},
    },
}


def _row(res, strategy):
    return res[res["strategy"] == strategy].iloc[0]


@pytest.mark.parametrize("split", ["dev", "val"])
def test_reproduces_frozen_summary(split):
    res = evaluate_from_judgments(load_judged_split(split), mode=LINEAR)
    expected = FROZEN[split]
    for strategy, vals in expected.items():
        if strategy == "profiles":
            continue
        row = _row(res, strategy)
        assert row["profiles"] == expected["profiles"]
        for metric, want in vals.items():
            assert row[metric] == pytest.approx(want, abs=1e-3), f"{split}/{strategy}/{metric}"


def test_dev_val_reversal_survives_gain_change():
    """Dev/Val 승자 역전은 gain 모드 탓이 아니다 — 표본이 작아서다.

    이 성질이 깨지면 '집계 전략 결정 불가'라는 P1 결론을 다시 봐야 한다.
    """
    dev = evaluate_from_judgments(load_judged_split("dev"), mode=EXPONENTIAL)
    val = evaluate_from_judgments(load_judged_split("val"), mode=EXPONENTIAL)
    assert _row(dev, "MAX")["NDCG"] > _row(dev, "TOP2_MEAN")["NDCG"]
    assert _row(val, "TOP2_MEAN")["NDCG"] > _row(val, "MAX")["NDCG"]


def test_per_profile_grouping_changes_result():
    """프로필 groupby 없이 전체 풀 상위 10개를 뽑던 예전 방식과 값이 달라야 한다.

    예전 방식에서는 MEAN/TOP2_MEAN 이 P@10 = 1.0 으로 계산됐다.
    """
    joined = load_judged_split("dev")
    fixed = _row(evaluate_from_judgments(joined, mode=LINEAR), "MEAN")["P@10"]

    scored = joined[joined["mean_score"].notna()]
    legacy_rels = scored.nlargest(10, "mean_score")["relevance"].astype(float)
    legacy = (legacy_rels >= 2).mean()

    assert legacy == 1.0  # 옛 버그 재현
    assert fixed == pytest.approx(0.7833, abs=1e-3)


def test_ndcg_uses_per_profile_pool_not_global():
    """IDCG 가 프로필별 풀에서 나오는지 — 합성 데이터로 확인."""
    joined = pd.DataFrame({
        "profile_id": ["p1", "p1", "p2", "p2"],
        "relevance": [3.0, 0.0, 1.0, 1.0],
        "max_score": [0.9, 0.8, 0.5, 0.4],
    })
    res = evaluate_from_judgments(joined, strategies=["max"], k=2, mode=LINEAR)
    # 두 프로필 모두 자기 풀 안에서 완벽한 순서 → macro-average 1.0
    assert res.iloc[0]["NDCG"] == pytest.approx(1.0)
    assert res.iloc[0]["profiles"] == 2


def test_empty_strategy_column_is_skipped():
    joined = pd.DataFrame({
        "profile_id": ["p1"],
        "relevance": [3.0],
        "max_score": [np.nan],
    })
    assert evaluate_from_judgments(joined, strategies=["max"], k=2).empty
