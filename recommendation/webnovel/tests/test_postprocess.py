"""후처리 술어. 한국어 시리즈 판정 · 연령 필터 · 관심 수 하한 · 다양성."""
import pandas as pd
import pytest

from src.postprocess import (
    apply_hard_filters,
    cap_author,
    cap_series,
    interleave_by_seed,
    postprocess,
    series_group,
    series_key,
)


def make_dataset(rows):
    """rows: (item_id, name, publisher, author, age_limit, interest_count)"""
    return pd.DataFrame(
        rows, columns=["item_id", "name", "publisher", "author", "age_limit", "interest_count"]
    )


def make_ranked(ids, scores=None, seeds=None):
    n = len(ids)
    return pd.DataFrame({
        "item_id": ids,
        "final_score": scores if scores is not None else [1.0 - i * 0.01 for i in range(n)],
        "dominant_seed": seeds if seeds is not None else [0] * n,
        "rank": range(1, n + 1),
    })


class TestSeriesKey:
    @pytest.mark.parametrize("name,expected", [
        ("나 혼자만 레벨업", "나 혼자만"),
        ("나 혼자만 레벨업 2부", "나 혼자만"),
        ("전지적 독자 시점 외전", "전지적 독자"),
        ("재벌집 막내아들 [단행본]", "재벌집 막내아들"),
        ("달빛조각사 시즌 2", "달빛조각사"),
        ("검술명가 막내아들 개정판", "검술명가 막내아들"),
    ])
    def test_strips_korean_series_markers(self, name, expected):
        assert series_key(name) == expected

    def test_editions_of_same_work_collapse(self):
        """[독점]·[단행본]은 같은 작품의 다른 상품이라 반드시 한 그룹이어야 한다."""
        assert series_key("환생검혼 [독점]") == series_key("환생검혼 [단행본]")


class TestSeriesGroup:
    def test_publisher_disambiguates(self):
        a = series_group("검신", "아르데오")
        b = series_group("검신", "다른출판사")
        assert a != b

    def test_falls_back_to_name_without_publisher(self):
        assert series_group("검신 2부", "") == series_key("검신 2부")

    def test_same_publisher_different_work_not_merged(self):
        """대형 출판사 다작이 통째로 한 그룹이 되면 안 된다."""
        assert series_group("검신", "아르데오") != series_group("마탑주", "아르데오")


class TestHardFilters:
    def test_drops_19_rated(self):
        ds = make_dataset([
            (1, "전체가", "P", "A", 0, 500),
            (2, "성인물", "P", "B", 19, 500),
            (3, "십오세", "P", "C", 15, 500),
        ])
        out = apply_hard_filters(make_ranked([1, 2, 3]), ds)
        assert set(out["item_id"]) == {1, 3}

    def test_min_interest_count_is_a_threshold_not_a_null_check(self):
        """Steam 은 결측이 이진 신호였지만 관심 수는 연속값이라 임계로 걸러야 한다."""
        ds = make_dataset([
            (1, "인기작", "P", "A", 0, 5000),
            (2, "관심2", "P", "B", 0, 2),
            (3, "경계값", "P", "C", 0, 100),
        ])
        out = apply_hard_filters(make_ranked([1, 2, 3]), ds, min_interest_count=100)
        assert set(out["item_id"]) == {1, 3}

    def test_missing_interest_is_treated_as_below_floor(self):
        ds = make_dataset([(1, "값있음", "P", "A", 0, 500), (2, "결측", "P", "B", 0, None)])
        out = apply_hard_filters(make_ranked([1, 2]), ds, min_interest_count=100)
        assert set(out["item_id"]) == {1}

    def test_no_floor_keeps_everything(self):
        ds = make_dataset([(1, "a", "P", "A", 0, 1), (2, "b", "P", "B", 0, None)])
        out = apply_hard_filters(make_ranked([1, 2]), ds, min_interest_count=None)
        assert len(out) == 2

    def test_empty_input_is_safe(self):
        ds = make_dataset([(1, "a", "P", "A", 0, 5)])
        assert apply_hard_filters(make_ranked([]), ds).empty


class TestCapSeries:
    def test_keeps_one_per_series(self):
        ds = make_dataset([
            (1, "환생검혼 [독점]", "아르데오", "A", 0, 100),
            (2, "환생검혼 [단행본]", "아르데오", "A", 0, 100),
            (3, "다른작품", "다른곳", "B", 0, 100),
        ])
        out = cap_series(make_ranked([1, 2, 3]), ds, series_max=1)
        assert set(out["item_id"]) == {1, 3}

    def test_higher_ranked_edition_wins(self):
        ds = make_dataset([
            (1, "검신 [독점]", "P", "A", 0, 100),
            (2, "검신 [단행본]", "P", "A", 0, 100),
        ])
        out = cap_series(make_ranked([2, 1]), ds, series_max=1)
        assert list(out["item_id"]) == [2]


class TestCapAuthor:
    def test_limits_one_author(self):
        """한 작가가 페이지를 점령하는 것을 막는다 — 이 도메인 특유의 축이다."""
        ds = make_dataset([(i, f"작품{i}", "P", "김작가", 0, 100) for i in range(1, 5)])
        out = cap_author(make_ranked([1, 2, 3, 4]), ds, author_max=2)
        assert len(out) == 2

    def test_unknown_author_is_not_grouped(self):
        """작가 미상끼리 묶이면 서로 무관한 작품이 잘려나간다."""
        ds = make_dataset([(i, f"작품{i}", "P", "", 0, 100) for i in range(1, 5)])
        out = cap_author(make_ranked([1, 2, 3, 4]), ds, author_max=2)
        assert len(out) == 4


class TestInterleave:
    def test_round_robin_mixes_seeds(self):
        """전역 상한이면 '1~50위 로판, 51~100위 무협'이 된다. 라운드로빈이어야 한다."""
        ranked = make_ranked(
            [1, 2, 3, 4, 5, 6],
            scores=[0.9, 0.89, 0.88, 0.5, 0.49, 0.48],
            seeds=[10, 10, 10, 20, 20, 20],
        )
        out = interleave_by_seed(ranked, top_n=6)
        assert list(out["dominant_seed"])[:4] == [10, 20, 10, 20]

    def test_strongest_seed_takes_first_place(self):
        ranked = make_ranked([1, 2], scores=[0.5, 0.9], seeds=[10, 20])
        assert interleave_by_seed(ranked, 2).iloc[0]["dominant_seed"] == 20

    def test_uneven_buckets_do_not_shorten_list(self):
        ranked = make_ranked([1, 2, 3, 4], scores=[.9, .8, .7, .6], seeds=[10, 10, 10, 20])
        assert len(interleave_by_seed(ranked, 4)) == 4

    def test_rank_is_renumbered(self):
        ranked = make_ranked([1, 2, 3], seeds=[10, 20, 10])
        assert list(interleave_by_seed(ranked, 3)["rank"]) == [1, 2, 3]


class TestPostprocessOrder:
    def test_full_chain(self):
        ds = make_dataset([
            (1, "환생검혼 [독점]", "아르데오", "김작가", 0, 5000),
            (2, "환생검혼 [단행본]", "아르데오", "김작가", 0, 5000),  # 시리즈 중복
            (3, "성인작품", "P", "이작가", 19, 5000),                # 연령
            (4, "저관심작", "P", "박작가", 0, 3),                    # 품질 하한
            (5, "정상작품", "P", "최작가", 0, 5000),
        ])
        out = postprocess(
            make_ranked([1, 2, 3, 4, 5], seeds=[10, 10, 20, 20, 30]),
            ds, top_n=10, min_interest_count=100,
        )
        assert set(out["item_id"]) == {1, 5}
