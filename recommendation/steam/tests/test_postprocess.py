# tests/test_postprocess.py
import pandas as pd
import pytest

from src.postprocess import (
    apply_hard_filters,
    cap_series,
    interleave_by_seed,
    postprocess,
    series_key,
)


def _ranked(appids, seeds, scores=None):
    n = len(appids)
    return pd.DataFrame({
        "steam_appid": appids,
        "dominant_seed": seeds,
        "final_score": scores if scores is not None else [1.0 - i * 0.01 for i in range(n)],
        "rank": range(1, n + 1),
    })


def _dataset(rows):
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ series_key

@pytest.mark.parametrize("a,b", [
    ("Counter-Strike 2", "Counter-Strike: Source"),
    ("Euro Truck Simulator 2", "Euro Truck Simulator"),
    ("The Elder Scrolls V: Skyrim Special Edition", "The Elder Scrolls V: Skyrim VR"),
    ("DARK SOULS™: REMASTERED", "DARK SOULS III"),
])
def test_series_key_groups_same_franchise(a, b):
    assert series_key(a) == series_key(b)


def test_series_key_separates_different_games():
    assert series_key("Stardew Valley") != series_key("Slime Rancher")


# --------------------------------------------------------------- 인터리빙

def test_interleave_round_robins_across_seeds():
    """한 시드가 상위를 독식하던 것을 페이지 단위로 섞는다."""
    ranked = _ranked(list(range(1, 10)), [10, 10, 10, 10, 20, 20, 30, 30, 30])
    out = interleave_by_seed(ranked, top_n=9)
    # 앞 3칸에 세 시드가 모두 등장해야 한다
    assert set(out.head(3)["dominant_seed"]) == {10, 20, 30}
    assert out["rank"].tolist() == list(range(1, 10))


def test_interleave_strongest_seed_gets_first_slot():
    ranked = _ranked([1, 2, 3, 4], [20, 20, 10, 10], scores=[0.5, 0.4, 0.9, 0.8])
    out = interleave_by_seed(ranked, top_n=4)
    assert out.iloc[0]["dominant_seed"] == 10  # 1등 점수가 더 높은 버킷이 먼저


def test_interleave_skips_exhausted_buckets():
    """이웃이 적은 시드가 있어도 목록이 짧아지지 않는다."""
    ranked = _ranked([1, 2, 3, 4, 5], [10, 10, 10, 10, 20])
    out = interleave_by_seed(ranked, top_n=5)
    assert len(out) == 5
    assert out["dominant_seed"].tolist() == [10, 20, 10, 10, 10]


def test_interleave_reduces_max_seed_share():
    ranked = _ranked(list(range(1, 21)), [10] * 16 + [20] * 2 + [30] * 2)
    before = ranked.head(10)["dominant_seed"].value_counts().iloc[0] / 10
    after = interleave_by_seed(ranked, 10)["dominant_seed"].value_counts().iloc[0] / 10
    assert before == 1.0
    assert after < before


def test_interleave_passthrough_without_dominant_seed():
    ranked = pd.DataFrame({"steam_appid": [1, 2], "final_score": [1.0, 0.9], "rank": [1, 2]})
    assert len(interleave_by_seed(ranked, top_n=2)) == 2


# ------------------------------------------------------------ 시리즈 상한

def test_cap_series_keeps_only_best_of_franchise():
    ranked = _ranked([1, 2, 3], [10, 10, 10])
    ds = _dataset([
        {"steam_appid": 1, "name": "Counter-Strike 2"},
        {"steam_appid": 2, "name": "Counter-Strike: Source"},
        {"steam_appid": 3, "name": "Stardew Valley"},
    ])
    out = cap_series(ranked, ds, series_max=1)
    assert out["steam_appid"].tolist() == [1, 3]  # 상위 것만 남는다


# ------------------------------------------------------------ hard filter

def test_hard_filters_drop_adult_and_vr_only():
    ranked = _ranked([1, 2, 3], [10, 10, 10])
    ds = _dataset([
        {"steam_appid": 1, "name": "정상", "genres": ["액션"], "categories": ["싱글 플레이어"]},
        {"steam_appid": 2, "name": "성인", "genres": ["신체 노출"], "categories": ["싱글 플레이어"]},
        {"steam_appid": 3, "name": "VR", "genres": ["액션"], "categories": ["VR 전용"]},
    ])
    out = apply_hard_filters(ranked, ds, drop_unreleased=False)
    assert out["steam_appid"].tolist() == [1]


def test_hard_filters_keep_gore():
    """고어/폭력은 취향이지 결격 사유가 아니다."""
    ranked = _ranked([1], [10])
    ds = _dataset([{"steam_appid": 1, "name": "고어겜", "genres": ["고어", "액션"],
                    "categories": ["싱글 플레이어"]}])
    assert len(apply_hard_filters(ranked, ds, drop_unreleased=False)) == 1


# ---------------------------------------------------------------- 통합

def test_postprocess_applies_filter_then_cap_then_interleave():
    ranked = _ranked([1, 2, 3, 4], [10, 10, 20, 20])
    ds = _dataset([
        {"steam_appid": 1, "name": "Counter-Strike 2", "genres": ["액션"], "categories": []},
        {"steam_appid": 2, "name": "Counter-Strike: Source", "genres": ["액션"], "categories": []},
        {"steam_appid": 3, "name": "Stardew Valley", "genres": ["인디"], "categories": []},
        {"steam_appid": 4, "name": "신체 노출 게임", "genres": ["신체 노출"], "categories": []},
    ])
    out = postprocess(ranked, ds, top_n=10, hard_filters=True, series_max=1)
    ids = out["steam_appid"].tolist()
    assert 4 not in ids          # hard filter
    assert 2 not in ids          # 시리즈 상한
    assert set(ids) == {1, 3}


def test_postprocess_disabled_is_plain_topn():
    ranked = _ranked([1, 2, 3], [10, 10, 10])
    ds = _dataset([{"steam_appid": i, "name": f"g{i}", "genres": [], "categories": []} for i in [1, 2, 3]])
    out = postprocess(ranked, ds, top_n=2, seed_interleave=False, series_max=0, hard_filters=False)
    assert out["steam_appid"].tolist() == [1, 2]
    assert out["rank"].tolist() == [1, 2]


def test_unknown_appid_is_not_treated_as_unreleased():
    """trend_features 에 없는 appid 를 미출시로 오판하면 목록이 통째로 비어버린다."""
    ranked = _ranked([999999001, 999999002], [10, 20])
    ds = _dataset([
        {"steam_appid": 999999001, "name": "a", "genres": ["액션"], "categories": []},
        {"steam_appid": 999999002, "name": "b", "genres": ["액션"], "categories": []},
    ])
    assert len(apply_hard_filters(ranked, ds, drop_unreleased=True)) == 2


def test_postprocess_handles_everything_filtered_out():
    ranked = _ranked([1], [10])
    ds = _dataset([{"steam_appid": 1, "name": "성인", "genres": ["선정적 콘텐츠"], "categories": []}])
    assert postprocess(ranked, ds, top_n=10).empty
