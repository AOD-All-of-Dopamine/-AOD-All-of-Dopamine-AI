# tests/test_eval_product.py
"""평가 하네스가 스스로를 못 믿게 만드는 두 가지 사고를 막는다.

1. 판정 풀이 랭킹을 못 덮으면 NDCG 가 1을 넘는다 (실측 P07 1.160).
2. 프로필 8개 × 10칸이면 1칸이 0.0125 — 신뢰구간 없이 인용하면 잡음을 개선이라 부르게 된다.
"""
import pandas as pd
import pytest

from src.eval_product import compare, evaluate_config, summarize
from src.metrics import (
    JUDGED_DENOM,
    SLOT_DENOM,
    format_ci,
    ndcg_at_k,
    paired_bootstrap_ci,
    precision_at_k,
)


# ------------------------------------------------------------ NDCG 상한 가드

def test_ndcg_raises_when_pool_does_not_cover_ranking():
    """pool 에 3점짜리가 없는데 랭킹 1위가 3점이면 NDCG 가 1을 넘는다."""
    with pytest.raises(ValueError, match="NDCG > 1"):
        ndcg_at_k([3, 3, 3], k=10, ideal_pool=[1, 1])


def test_ndcg_ok_when_pool_covers_ranking():
    # DCG = 7/1 + 3/log2(3) = 8.893 · IDCG = 7/1 + 7/log2(3) + 3/2 = 12.917
    assert ndcg_at_k([3, 2], k=10, ideal_pool=[3, 3, 2, 0]) == pytest.approx(0.6885, abs=1e-3)


def test_ndcg_one_when_ranking_is_the_ideal():
    assert ndcg_at_k([3, 2, 0], k=10, ideal_pool=[3, 2, 0]) == pytest.approx(1.0)


def test_ndcg_zero_pool_is_zero_not_error():
    assert ndcg_at_k([0, 0], k=10, ideal_pool=[0, 0]) == 0.0


# ------------------------------------------------------- 짝지은 신뢰구간

def test_paired_ci_detects_consistent_improvement():
    r = paired_bootstrap_ci([0.5] * 8, [0.7] * 8, iterations=2000)
    assert r["delta"] == pytest.approx(0.2)
    assert r["significant"]


def test_paired_ci_rejects_one_slot_noise():
    """8프로필 중 하나만 1칸 오른 것은 유의하지 않아야 한다."""
    base = [0.6] * 8
    var = [0.7] + [0.6] * 7
    r = paired_bootstrap_ci(base, var, iterations=5000)
    assert r["delta"] == pytest.approx(0.0125)
    assert not r["significant"]


def test_paired_ci_rejects_offsetting_trade():
    """P05 +0.2 / P07 -0.4 처럼 상쇄되는 트레이드는 유의하지 않다."""
    r = paired_bootstrap_ci([0.5] * 8, [0.6, 0.6, 0.6, 0.5, 0.7, 0.5, 0.1, 0.5], iterations=5000)
    assert not r["significant"]


def test_paired_ci_needs_matching_shapes():
    with pytest.raises(ValueError, match="짝이 맞지"):
        paired_bootstrap_ci([0.1, 0.2], [0.1])


def test_paired_ci_needs_at_least_two_profiles():
    with pytest.raises(ValueError, match="2개 미만"):
        paired_bootstrap_ci([0.5], [0.7])


def test_format_ci_marks_verdict():
    assert "판정 불가" in format_ci(paired_bootstrap_ci([0.5] * 4, [0.5] * 4, iterations=500))


# ------------------------------------------------- P@k 분모를 명시해야 하는 이유

def test_precision_denominator_changes_the_answer():
    """미판정 3칸을 뺀 7칸만 세면 0.71, 10칸으로 세면 0.50 — 같은 목록이다."""
    rels = [3, 2, 2, 2, 2, 1, 0]          # 10칸 중 7칸만 판정됨
    assert precision_at_k(rels, k=10, denominator=JUDGED_DENOM) == pytest.approx(5 / 7)
    assert precision_at_k(rels, k=10, denominator=SLOT_DENOM) == pytest.approx(0.5)


def test_precision_denominators_agree_when_pool_is_complete():
    rels = [3, 2, 2, 2, 2, 1, 0, 0, 1, 2]
    assert (precision_at_k(rels, k=10, denominator=JUDGED_DENOM)
            == precision_at_k(rels, k=10, denominator=SLOT_DENOM))


def test_precision_rejects_unknown_denominator():
    with pytest.raises(ValueError, match="denominator"):
        precision_at_k([3], k=10, denominator="whatever")


# ----------------------------------------------------------- evaluate_config

@pytest.fixture
def judgments():
    return {("pa", 1): 3, ("pa", 2): 2, ("pa", 3): 0, ("pb", 4): 3, ("pb", 5): 1}


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    p = pd.DataFrame({"profile_id": ["pa", "pb"], "liked_appids": [[100], [200]],
                      "profile_type": ["coherent"] * 2, "split": ["dev"] * 2})
    path = tmp_path / "profiles.parquet"
    p.to_parquet(path, index=False)
    monkeypatch.setattr("src.eval_product.PROFILES_PATH", path)
    return path


def test_evaluate_config_computes_per_profile(profiles, judgments):
    ranked = {"pa": [1, 2, 3], "pb": [4, 5]}
    per = evaluate_config(lambda pid, liked: pd.DataFrame({"steam_appid": ranked[pid]}),
                          ["pa", "pb"], judgments)
    assert per["unjudged"].sum() == 0
    assert per.set_index("profile_id").loc["pa", "p_at_k"] == pytest.approx(0.2)  # 2/10
    assert (per["ndcg_at_k"] <= 1.0).all()


def test_evaluate_config_refuses_unjudged_by_default(profiles, judgments):
    """미판정을 조용히 0점으로 두면 후보를 많이 바꾸는 설정이 자동으로 불리해진다."""
    with pytest.raises(ValueError, match="미판정"):
        evaluate_config(lambda pid, liked: pd.DataFrame({"steam_appid": [1, 999]}),
                        ["pa"], judgments)


def test_evaluate_config_can_opt_out_of_strict(profiles, judgments):
    per = evaluate_config(lambda pid, liked: pd.DataFrame({"steam_appid": [1, 999]}),
                          ["pa"], judgments, strict=False)
    assert per.iloc[0]["unjudged"] == 1


def test_summarize_reports_pool_size(profiles, judgments):
    per = evaluate_config(lambda pid, liked: pd.DataFrame({"steam_appid": [1, 2]}),
                          ["pa"], judgments)
    s = summarize(per)
    assert s["profiles"] == 1 and s["unjudged"] == 0 and s["pool_size_total"] == 3


# ------------------------------------------------------------------ compare

def test_compare_attaches_ci_to_every_variant(profiles, judgments):
    a = evaluate_config(lambda pid, l: pd.DataFrame({"steam_appid": {"pa": [1], "pb": [4]}[pid]}),
                        ["pa", "pb"], judgments)
    b = evaluate_config(lambda pid, l: pd.DataFrame({"steam_appid": {"pa": [3], "pb": [5]}[pid]}),
                        ["pa", "pb"], judgments)
    out = compare({"기준": a, "변형": b})
    assert out.loc[0, "95%CI"] == "(기준선)"
    assert out.loc[1, "유의"] in ("예", "아니오")


def test_compare_refuses_mismatched_profile_sets(profiles, judgments):
    a = evaluate_config(lambda pid, l: pd.DataFrame({"steam_appid": [1]}), ["pa"], judgments)
    b = evaluate_config(lambda pid, l: pd.DataFrame({"steam_appid": [4]}), ["pb"], judgments)
    with pytest.raises(ValueError, match="짝지을 수 없습니다"):
        compare({"a": a, "b": b})
