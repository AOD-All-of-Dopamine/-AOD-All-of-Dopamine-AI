# tests/test_evaluate.py
import pandas as pd
import pytest

from src.evaluate import (
    assert_fully_judged,
    build_comparison,
    compute_summary,
    per_anchor_metrics,
)
from src.metrics import EXPONENTIAL, LINEAR

CFG = {
    "evaluation": {
        "baseline_experiment_id": "steam_s1_tfidf_v2",
        "treatment_experiment_id": "steam_s1_qwen_v2",
        "precision_gate": 0.70,
    }
}


def test_per_anchor_metrics():
    df = pd.DataFrame({
        "anchor_steam_appid": [1] * 3 + [2] * 3,
        "rank": [1, 2, 3, 1, 2, 3],
        "relevance": [3, 1, 0, 2, 2, 2],
        "recommendation_confidence": [2, 2, 1, 3, 3, 3],
    })
    per = per_anchor_metrics(df, k=3, mode=EXPONENTIAL)
    a1 = per[per["anchor_steam_appid"] == 1].iloc[0]
    a2 = per[per["anchor_steam_appid"] == 2].iloc[0]
    assert a1["precision"] == 1 / 3
    assert a2["precision"] == 1.0
    assert a2["ndcg"] == 1.0  # 이미 내림차순
    assert a1["avg_relevance"] == 4 / 3


def test_per_anchor_metrics_respects_gain_mode():
    df = pd.DataFrame({
        "anchor_steam_appid": [1] * 3,
        "rank": [1, 2, 3],
        "relevance": [1, 3, 0],
        "recommendation_confidence": [1, 1, 1],
    })
    lin = per_anchor_metrics(df, k=3, mode=LINEAR).iloc[0]["ndcg"]
    exp = per_anchor_metrics(df, k=3, mode=EXPONENTIAL).iloc[0]["ndcg"]
    assert lin != exp  # 모드가 실제로 결과를 바꾼다


def _summaries(base_p, base_n, treat_p, treat_n):
    return {
        "steam_s1_tfidf_v2": {"precision_at_10": base_p, "ndcg_at_10": base_n},
        "steam_s1_qwen_v2": {"precision_at_10": treat_p, "ndcg_at_10": treat_n},
    }


def test_build_comparison_gate_passes():
    c = build_comparison(_summaries(0.5, 0.5, 0.75, 0.8), CFG)
    assert c["treatment_beats_baseline"] is True
    assert c["quality_gate_passed"] is True
    assert c["precision_delta"] == pytest.approx(0.25)


def test_build_comparison_gate_fails_below_threshold():
    c = build_comparison(_summaries(0.5, 0.5, 0.65, 0.8), CFG)
    assert c["treatment_beats_baseline"] is True
    assert c["quality_gate_passed"] is False  # 이겼어도 절대 기준 미달


def test_build_comparison_raises_on_missing_experiment_id():
    """예전에는 ID 가 어긋나면 조용히 스킵돼 게이트가 한 번도 산출되지 않았다."""
    summaries = {"steam_s1_qwen_v1": {"precision_at_10": 0.8, "ndcg_at_10": 0.8}}
    with pytest.raises(SystemExit, match="실험 ID가 없습니다"):
        build_comparison(summaries, CFG)


def test_assert_fully_judged_passes_when_complete():
    df = pd.DataFrame({"experiment_id": ["a", "a"], "relevance": [1, 2]})
    assert assert_fully_judged(df, "x.xlsx") is None


def test_assert_fully_judged_reports_per_experiment_counts():
    df = pd.DataFrame({
        "experiment_id": ["a", "a", "b"],
        "relevance": [1, None, None],
    })
    with pytest.raises(SystemExit) as e:
        assert_fully_judged(df, "x.xlsx")
    msg = str(e.value)
    assert "2/3 행이 미판정" in msg
    assert "a: 1/2 판정됨" in msg
    assert "b: 0/1 판정됨" in msg


def test_compute_summary_averages_anchors():
    per = pd.DataFrame({
        "precision": [1.0, 0.0],
        "ndcg": [1.0, 0.5],
        "avg_relevance": [3.0, 1.0],
        "avg_confidence": [3.0, 1.0],
    })
    s = compute_summary(per)
    assert s["precision_at_10"] == 0.5
    assert s["ndcg_at_10"] == 0.75
