# tests/test_blind_judge.py
"""블라인드가 실제로 가려지는지 — 판정 514쌍 중 387쌍이 이 장치 없이 매겨졌다."""
import numpy as np
import pandas as pd
import pytest

from src.blind_judge import SCORE_COL, TAG_COL, build_sheet, leaks, merge, write


@pytest.fixture
def dataset():
    return pd.DataFrame([
        {"steam_appid": i, "name": f"게임{i}", "genres": ["액션"],
         "short_description": f"설명 {i}", "recommendations_total": i * 100}
        for i in range(1, 21)
    ])


@pytest.fixture
def profiles():
    return pd.DataFrame({
        "profile_id": ["alpha", "beta", "gamma"],
        "liked_appids": [[1, 2], [3], [4, 5, 6]],
        "profile_type": ["coherent"] * 3,
        "split": ["dev", "val", "dev"],
    })


@pytest.fixture
def built(profiles, dataset):
    pairs = [("alpha", 10), ("alpha", 11), ("beta", 12), ("gamma", 13), ("gamma", 14)]
    return build_sheet(pairs, profiles, dataset)


# ------------------------------------------------------------ 무엇이 가려지나

def test_sheet_hides_identity_columns(built):
    sheet, _ = built
    assert leaks(sheet) == []
    for banned in ("profile_id", "candidate_appid", "rank", "strategy", "final_score"):
        assert banned not in sheet.columns


def test_write_refuses_a_leaky_sheet(tmp_path, built):
    sheet, mapping = built
    sheet = sheet.assign(profile_id="alpha")
    with pytest.raises(ValueError, match="정체를 드러내는"):
        write(tmp_path, sheet, mapping)


def test_reviews_are_hidden_by_default(built):
    """리뷰 수는 min_reviews 실험의 조작 변수라, 보이면 어느 설정에서 왔는지 추론된다."""
    sheet, _ = built
    assert "리뷰수" not in sheet.columns


def test_reviews_can_be_shown_explicitly(profiles, dataset):
    sheet, _ = build_sheet([("alpha", 10)], profiles, dataset, show_reviews=True)
    assert sheet.iloc[0]["리뷰수"] == 1000


def test_profile_ids_are_anonymised(built):
    sheet, mapping = built
    assert set(sheet["프로필"]) <= {"A01", "A02", "A03"}
    assert not set(sheet["프로필"]) & set(mapping["profile_id"])


def test_anon_ids_are_not_a_function_of_definition_order(profiles, dataset):
    """A01 = 정의 첫 프로필이면 익명이 아니다 — 배정이 seed 에 의존해야 한다."""
    pairs = [(p, 10 + i) for i, p in enumerate(["alpha", "beta", "gamma"])]
    seen = set()
    for seed in range(6):
        _, mapping = build_sheet(pairs, profiles, dataset, seed=seed)
        m = mapping.set_index("profile_id")["anon_profile_id"]
        seen.add(tuple(m[p] for p in ("alpha", "beta", "gamma")))
    assert len(seen) > 1, "seed 를 바꿔도 익명 id 배정이 그대로다"


def test_rows_are_shuffled(profiles, dataset):
    """섞지 않으면 프로필별로 뭉쳐 나와 맥락이 그대로 보인다."""
    pairs = [("alpha", i) for i in range(10, 16)] + [("beta", i) for i in range(16, 21)]
    sheet, _ = build_sheet(pairs, profiles, dataset, seed=7)
    order = sheet["프로필"].tolist()
    assert order != sorted(order)


def test_sheet_shows_what_the_judge_needs(built):
    sheet, _ = built
    for col in ("좋아하는_게임", "좋아하는_게임_장르", "추천_게임", "추천_게임_장르", "추천_게임_설명"):
        assert col in sheet.columns
    assert sheet.iloc[0]["추천_게임"].startswith("게임")


# ------------------------------------------------------------------ 무결성

def test_duplicate_pairs_are_scored_once(profiles, dataset):
    """판정은 (프로필, 게임)의 함수다 — 설정이 달라도 다시 채점하지 않는다."""
    sheet, mapping = build_sheet([("alpha", 10), ("alpha", 10), ("beta", 10)], profiles, dataset)
    assert len(sheet) == 2 and len(mapping) == 2


def test_mapping_and_sheet_stay_aligned(built):
    sheet, mapping = built
    assert sheet["pair_key"].tolist() == mapping["pair_key"].tolist()
    assert len(set(sheet["pair_key"])) == len(sheet)


# ------------------------------------------------------------------ merge

def _score(tmp_path, built, scores, tags=None, by_pair=None):
    """행 순서가 섞이므로, 특정 쌍을 지정하려면 by_pair 로 (profile_id, appid) → 점수를 준다."""
    from src.blind_judge import SCORED

    sheet, mapping = built
    write(tmp_path, sheet, mapping)
    s = pd.read_csv(tmp_path / "blind_sheet.csv")
    if by_pair is not None:
        key = mapping.set_index("pair_key")
        s[SCORE_COL] = [by_pair[(key.loc[k, "profile_id"], int(key.loc[k, "candidate_appid"]))]
                        for k in s["pair_key"]]
    else:
        s[SCORE_COL] = scores
    if tags is not None:
        s[TAG_COL] = tags
    s.to_csv(tmp_path / SCORED, index=False)
    return tmp_path


def test_merge_restores_real_ids(tmp_path, built):
    d = _score(tmp_path, built, [3, 2, 1, 0, 2])
    out, _ = merge(d)
    assert set(out) == {("alpha", 10), ("alpha", 11), ("beta", 12), ("gamma", 13), ("gamma", 14)}
    assert all(0 <= v[0] <= 3 for v in out.values())


def test_merge_emits_pastable_code(tmp_path, built):
    d = _score(tmp_path, built, [3, 2, 1, 0, 2])
    _, code = merge(d, name="ROUND9")
    assert code.startswith("ROUND9 = {")
    assert '("alpha", 10)' in code


def test_merge_rejects_unscored_rows(tmp_path, built):
    d = _score(tmp_path, built, [3, 2, None, 0, 2])
    with pytest.raises(ValueError, match="미채점"):
        merge(d)


def test_merge_rejects_out_of_range_scores(tmp_path, built):
    d = _score(tmp_path, built, [3, 2, 1, 0, 5])
    with pytest.raises(ValueError, match="0~3"):
        merge(d)


def test_merge_drops_tags_on_positive_scores(tmp_path, built):
    """2점 이상에 실패 태그가 붙으면 앞뒤가 안 맞는다."""
    d = _score(tmp_path, built, None, tags=["IRRELEVANT"] * 5, by_pair={
        ("alpha", 10): 3, ("alpha", 11): 1, ("beta", 12): 1,
        ("gamma", 13): 0, ("gamma", 14): 2})
    out, _ = merge(d)
    assert out[("alpha", 10)][1] == ""            # 3점 → 태그 제거
    assert out[("gamma", 14)][1] == ""            # 2점 → 태그 제거
    assert out[("alpha", 11)][1] == "IRRELEVANT"  # 1점 → 태그 유지
