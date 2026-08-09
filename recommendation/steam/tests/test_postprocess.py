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


def test_min_reviews_cuts_the_long_tail():
    """전체 코퍼스에서는 이진 신호(has_recommendations)로 부족하다.

    리뷰 수백 개짜리가 21,892개나 통과해 유명작을 밀어낸다 — 실측 P@10 0.625 → 0.562.
    """
    ranked = _ranked([1, 2, 3], [10, 10, 10])
    ds = _dataset([
        {"steam_appid": 1, "name": "유명작", "genres": [], "categories": [],
         "has_recommendations": True, "recommendations_total": 12000},
        {"steam_appid": 2, "name": "롱테일", "genres": [], "categories": [],
         "has_recommendations": True, "recommendations_total": 120},
        {"steam_appid": 3, "name": "경계값", "genres": [], "categories": [],
         "has_recommendations": True, "recommendations_total": 300},
    ])
    out = apply_hard_filters(ranked, ds, drop_unreleased=False, min_reviews=300)
    assert out["steam_appid"].tolist() == [1, 3]   # 하한은 포함(>=)


def test_min_reviews_treats_missing_as_zero():
    """Int64 결측을 그대로 비교하면 `boolean value of NA is ambiguous` 로 터진다."""
    ranked = _ranked([1, 2], [10, 10])
    ds = _dataset([
        {"steam_appid": 1, "name": "결측", "genres": [], "categories": [],
         "has_recommendations": False, "recommendations_total": pd.NA},
        {"steam_appid": 2, "name": "정상", "genres": [], "categories": [],
         "has_recommendations": True, "recommendations_total": 5000},
    ])
    ds["recommendations_total"] = ds["recommendations_total"].astype("Int64")
    out = apply_hard_filters(ranked, ds, drop_unreleased=False, min_reviews=300)
    assert out["steam_appid"].tolist() == [2]


def test_min_reviews_zero_is_a_noop():
    """구 실험 재현성 — 기본값 0 에서는 아무것도 걸러지지 않아야 한다."""
    ranked = _ranked([1], [10])
    ds = _dataset([{"steam_appid": 1, "name": "무명", "genres": [], "categories": [],
                    "has_recommendations": False, "recommendations_total": pd.NA}])
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


# --------------------------------------------------- publisher 기반 시리즈

def test_series_group_uses_publisher_to_catch_renamed_sequels():
    """이름만으로는 못 잡던 사례 — 판정에서 FRANCHISE_OR_VARIANT 로 걸렸던 것들."""
    from src.postprocess import series_group

    # 'WT2' 와 'War Trigger 3' 는 이름이 완전히 달라 이름 휴리스틱이 실패한다
    assert series_key("WT2") != series_key("War Trigger 3")
    # 'MadOut' 과 'MadOut Ice Storm' 은 앞 2단어가 달라 실패한다
    assert series_key("MadOut") != series_key("MadOut Ice Storm")
    from src.postprocess import series_group as sg

    assert sg("MadOut", "MadOut Games") == sg("MadOut Ice Storm", "MadOut Games")


def test_series_group_falls_back_to_name_without_publisher():
    """구 데이터에는 publisher 가 없다 — 이름 기반으로 되돌아가야 한다."""
    from src.postprocess import series_group

    assert series_group("Counter-Strike 2", "") == series_key("Counter-Strike 2")
    assert series_group("Counter-Strike 2", "") == series_group("Counter-Strike: Source", "")


def test_series_group_does_not_merge_unrelated_games_of_same_publisher():
    """같은 퍼블리셔라도 다른 시리즈면 따로 세야 한다 (Valve 의 CS 와 Portal)."""
    from src.postprocess import series_group

    assert series_group("Counter-Strike 2", "Valve") != series_group("Portal 2", "Valve")


def test_cap_series_uses_publisher_column_when_present():
    ranked = _ranked([1, 2, 3], [10, 10, 10])
    ds = _dataset([
        {"steam_appid": 1, "name": "MadOut", "publisher": "MadOut Games"},
        {"steam_appid": 2, "name": "MadOut Ice Storm", "publisher": "MadOut Games"},
        {"steam_appid": 3, "name": "Stardew Valley", "publisher": "ConcernedApe"},
    ])
    out = cap_series(ranked, ds, series_max=1)
    assert out["steam_appid"].tolist() == [1, 3]


# ------------------------------------------------- 퍼블리셔 상한

def test_cap_publisher_catches_what_series_key_misses():
    """'War Trigger 3' / 'WT2' 는 이름이 달라 시리즈 판정이 못 잡는다(실측)."""
    from src.postprocess import cap_publisher, series_group

    assert series_group("War Trigger 3", "Rocketeer") != series_group("WT2", "Rocketeer")
    ranked = _ranked([1, 2, 3], [10, 10, 10])
    ds = _dataset([
        {"steam_appid": 1, "name": "War Trigger 3", "publisher": "Rocketeer"},
        {"steam_appid": 2, "name": "WT2", "publisher": "Rocketeer"},
        {"steam_appid": 3, "name": "War Trigger 2", "publisher": "Rocketeer"},
    ])
    out = cap_publisher(ranked, ds, publisher_max=2)
    assert out["steam_appid"].tolist() == [1, 2]  # 3개 중 2개만


def test_cap_publisher_allows_different_series_of_same_publisher():
    """Skyrim 과 Elder Scrolls Online 은 같은 Bethesda 지만 다른 경험이다."""
    from src.postprocess import cap_publisher

    ranked = _ranked([1, 2], [10, 10])
    ds = _dataset([
        {"steam_appid": 1, "name": "Skyrim", "publisher": "Bethesda"},
        {"steam_appid": 2, "name": "The Elder Scrolls Online", "publisher": "Bethesda"},
    ])
    assert len(cap_publisher(ranked, ds, publisher_max=2)) == 2


def test_cap_publisher_ignores_unknown_publisher():
    from src.postprocess import cap_publisher

    ranked = _ranked([1, 2, 3], [10, 10, 10])
    ds = _dataset([{"steam_appid": i, "name": f"g{i}", "publisher": ""} for i in [1, 2, 3]])
    assert len(cap_publisher(ranked, ds, publisher_max=1)) == 3


def test_cap_publisher_noop_without_column():
    """구 데이터에는 publisher 컬럼이 없다."""
    from src.postprocess import cap_publisher

    ranked = _ranked([1, 2], [10, 10])
    ds = _dataset([{"steam_appid": 1, "name": "a"}, {"steam_appid": 2, "name": "b"}])
    assert len(cap_publisher(ranked, ds, publisher_max=1)) == 2


# ------------------------------------------------- 시드 개수 계약 (Step 5)

def _multi_seed(n_seeds, per_seed=4):
    """시드 n개 × per_seed개 후보. 시드 번호가 작을수록 강하다."""
    rows = []
    for s in range(n_seeds):
        for j in range(per_seed):
            rows.append({"steam_appid": s * 100 + j, "dominant_seed": s,
                         "final_score": 1.0 - s * 0.01 - j * 0.001})
    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def test_single_seed_takes_the_whole_page():
    """시드가 하나면 인터리빙이 할 일이 없다 — 10/10 독점이 정상이다."""
    out = interleave_by_seed(_multi_seed(1, per_seed=20), top_n=10)
    assert len(out) == 10
    assert set(out["dominant_seed"]) == {0}


def test_ten_seeds_get_one_slot_each():
    out = interleave_by_seed(_multi_seed(10), top_n=10)
    assert len(out) == 10
    assert sorted(out["dominant_seed"]) == list(range(10))


def test_fifteen_seeds_drop_the_five_weakest_from_page_one():
    """10칸에 15개 취향은 안 들어간다. 약한 5개가 1페이지에서 빠지는 것이 계약이다."""
    out = interleave_by_seed(_multi_seed(15), top_n=10)
    assert len(out) == 10
    assert sorted(out["dominant_seed"]) == list(range(10))   # 강한 순으로 10개
    assert set(range(10, 15)).isdisjoint(set(out["dominant_seed"]))


def test_weak_seeds_starve_without_bucket_offset():
    """회전이 없으면 매 페이지가 처음부터 정렬돼 같은 상위 10개 버킷이 계속 이긴다.

    실측: 버킷당 후보가 30개만 넘으면 11~15번째 시드는 30페이지 안에 한 번도 안 나온다.
    후보 4개짜리 인공 데이터에서는 5페이지에 나와서 처음엔 문제가 안 보였다.
    """
    ranked = _multi_seed(15, per_seed=30)
    seen, appeared = set(), set()
    for _ in range(20):
        rest = ranked[~ranked["steam_appid"].isin(seen)]
        page = interleave_by_seed(rest, top_n=10)          # bucket_offset 없음
        seen |= set(page["steam_appid"])
        appeared |= {int(s) for s in page["dominant_seed"]}
    assert appeared.isdisjoint({11, 12, 13, 14})           # 20페이지 동안 한 번도 안 나온다


def test_bucket_offset_gives_every_seed_a_turn():
    """시드 N개면 ceil(N/page_size) 페이지 안에 전부 한 번은 등장해야 한다."""
    ranked = _multi_seed(15, per_seed=30)
    seen, appeared = set(), set()
    for _ in range(2):                                      # ceil(15/10) = 2
        rest = ranked[~ranked["steam_appid"].isin(seen)]
        # next_page 와 같은 방식: 이미 본 개수를 그대로 넘긴다 → 회전 폭이 page_size
        pg = interleave_by_seed(rest, top_n=10, bucket_offset=len(seen))
        seen |= set(pg["steam_appid"])
        appeared |= {int(s) for s in pg["dominant_seed"]}
    assert appeared == set(range(15))


def test_bucket_offset_keeps_all_seeds_present_when_they_fit():
    """시드가 page_size 이하면 회전해도 세 시드가 모두 남아야 한다.

    칸이 나누어떨어지지 않을 때 여분 칸의 주인이 바뀔 뿐이다(4/3/3 → 3/4/3).
    """
    ranked = _multi_seed(3, per_seed=20)
    for off in range(4):
        counts = interleave_by_seed(ranked, top_n=10, bucket_offset=off)["dominant_seed"].value_counts()
        assert len(counts) == 3
        assert sorted(counts.tolist()) == [3, 3, 4]


def test_bucket_offset_wraps_around():
    ranked = _multi_seed(3, per_seed=20)
    assert (interleave_by_seed(ranked, 10, bucket_offset=0)["steam_appid"].tolist()
            == interleave_by_seed(ranked, 10, bucket_offset=3)["steam_appid"].tolist())


def test_page_always_fills_when_candidates_remain():
    for n in (1, 2, 3, 5, 10, 15, 20):
        out = interleave_by_seed(_multi_seed(n, per_seed=20), top_n=10)
        assert len(out) == 10, n
        assert out["rank"].tolist() == list(range(1, 11)), n


def test_uneven_buckets_do_not_shorten_the_page():
    """이웃이 1개뿐인 시드가 있어도 10칸을 채워야 한다."""
    ranked = pd.concat([_multi_seed(1, per_seed=1),
                        _multi_seed(2, per_seed=20).assign(dominant_seed=lambda d: d.dominant_seed + 1)])
    ranked = ranked.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    assert len(interleave_by_seed(ranked, top_n=10)) == 10


# --------------------------------- 성인 콘텐츠는 공식 descriptor 로 거른다

def _with_cd(rows):
    df = pd.DataFrame(rows)
    return df


def test_official_descriptor_blocks_explicit_games():
    """3(노골적 성적 묘사) / 4(성인 전용)가 있으면 거른다."""
    ranked = _ranked([1, 2], [10, 10])
    ds = _with_cd([
        {"steam_appid": 1, "name": "일반", "genres": ["액션"], "categories": [],
         "content_descriptorids": [1, 2, 5]},
        {"steam_appid": 2, "name": "성인", "genres": ["액션"], "categories": [],
         "content_descriptorids": [1, 3, 4, 5]},
    ])
    out = apply_hard_filters(ranked, ds, drop_unreleased=False)
    assert out["steam_appid"].tolist() == [1]


def test_mainstream_games_with_mature_content_pass():
    """Cyberpunk[1,2,5] · Witcher3[1,5] · GTA V[1,2,5] 는 통과해야 한다.

    사용자 태그로 거르면 이것들이 전부 차단됐다 — 그래서 descriptor 를 쓴다.
    """
    ranked = _ranked([1, 2, 3], [10, 10, 10])
    ds = _with_cd([
        {"steam_appid": 1, "name": "Cyberpunk", "genres": ["RPG"], "categories": [],
         "content_descriptorids": [1, 2, 5]},
        {"steam_appid": 2, "name": "Witcher3", "genres": ["RPG"], "categories": [],
         "content_descriptorids": [1, 5]},
        {"steam_appid": 3, "name": "PAYDAY 3", "genres": ["액션"], "categories": [],
         "content_descriptorids": [2, 5]},
    ])
    assert len(apply_hard_filters(ranked, ds, drop_unreleased=False)) == 3


def test_missing_descriptor_column_is_not_fatal():
    """구 dataset 에는 컬럼이 없다 — 장르 기준만으로 돌아야 한다."""
    ranked = _ranked([1], [10])
    ds = _dataset([{"steam_appid": 1, "name": "g", "genres": ["액션"], "categories": []}])
    assert len(apply_hard_filters(ranked, ds, drop_unreleased=False)) == 1


def test_empty_descriptor_list_passes():
    """개발사가 지정을 안 한 경우 — 놓치더라도 정상 게임을 잃지는 않는다."""
    ranked = _ranked([1], [10])
    ds = _with_cd([{"steam_appid": 1, "name": "미지정", "genres": ["액션"], "categories": [],
                    "content_descriptorids": []}])
    assert len(apply_hard_filters(ranked, ds, drop_unreleased=False)) == 1


def test_drop_adult_false_disables_descriptor_check():
    ranked = _ranked([1], [10])
    ds = _with_cd([{"steam_appid": 1, "name": "성인", "genres": [], "categories": [],
                    "content_descriptorids": [3, 4]}])
    assert len(apply_hard_filters(ranked, ds, drop_unreleased=False, drop_adult=False)) == 1


def test_descriptor_check_survives_parquet_roundtrip(tmp_path):
    """parquet 는 리스트를 numpy 배열로 돌려준다 — `x or ()` 로 쓰면 거기서 터진다."""
    ds = pd.DataFrame([
        {"steam_appid": 1, "name": "일반", "genres": ["액션"], "categories": [],
         "content_descriptorids": [1, 2, 5]},
        {"steam_appid": 2, "name": "성인", "genres": ["액션"], "categories": [],
         "content_descriptorids": [3, 4]},
        {"steam_appid": 3, "name": "미지정", "genres": ["액션"], "categories": [],
         "content_descriptorids": []},
    ])
    path = tmp_path / "ds.parquet"
    ds.to_parquet(path, index=False)
    out = apply_hard_filters(_ranked([1, 2, 3], [10, 10, 10]),
                             pd.read_parquet(path), drop_unreleased=False)
    assert out["steam_appid"].tolist() == [1, 3]
