"""정제 계약. 무엇을 버리고 무엇을 남기는지."""
import pandas as pd

from src.data_loader import build_dataset, compute_profile, parse_age, record_to_row


def raw(**over):
    base = {
        "product_no": 1, "title": "환생검혼", "synopsis": "죽었다 깨어나니 검이 되어 있었다." * 3,
        "genres": ["무협"], "author": "김작가", "publisher": "아르데오",
        "age_rating": "전체 이용가", "status": "연재중", "rating": 9.5,
        "interest_count": 5000, "comment_count": 30, "episode_count": 120,
        "cover_image": "http://x/y.jpg", "url": "http://x",
    }
    base.update(over)
    return base


class TestParseAge:
    def test_numeric_grades(self):
        assert parse_age("15세 이용가") == 15
        assert parse_age("19세 이용가") == 19

    def test_all_ages_is_zero(self):
        assert parse_age("전체 이용가") == 0

    def test_missing_is_zero(self):
        assert parse_age("") == 0 and parse_age(None) == 0


class TestRecordToRow:
    def test_happy_path(self):
        r = record_to_row(raw(), min_text_chars=20)
        assert r["item_id"] == 1
        assert r["name"] == "환생검혼"
        assert r["genres"] == ["무협"]
        assert r["has_interest"] is True
        assert r["is_completed"] is False
        assert r["age_limit"] == 0

    def test_completed_flag(self):
        assert record_to_row(raw(status="완결"), 20)["is_completed"] is True

    def test_rejects_missing_id(self):
        assert record_to_row(raw(product_no=None), 20) is None

    def test_rejects_missing_title(self):
        assert record_to_row(raw(title=""), 20) is None

    def test_rejects_site_name_title(self):
        """판매중지 페이지가 리다이렉트로 섞이면 제목이 사이트명이 된다."""
        assert record_to_row(raw(title="SERIES"), 20) is None

    def test_short_synopsis_survives_when_title_carries_signal(self):
        """줄거리가 카피 문구뿐이어도 제목이 로그라인이면 버리지 않는다."""
        r = record_to_row(
            raw(title="FA 먹튀 선수가 돈값 하기 시작함", synopsis="드디어 200억의 값어치를 하기 시작했다"),
            min_text_chars=20,
        )
        assert r is not None

    def test_rejects_when_combined_text_too_short(self):
        assert record_to_row(raw(title="가", synopsis="", genres=[]), min_text_chars=20) is None

    def test_missing_interest_is_flagged_not_zeroed(self):
        """결측과 0을 구분해야 품질 하한이 의미를 갖는다."""
        r = record_to_row(raw(interest_count=None), 20)
        assert r["has_interest"] is False and r["interest_count"] is None

    def test_html_is_stripped(self):
        r = record_to_row(raw(synopsis="<p>죽었다 &amp; 깨어났다</p>" * 3), 20)
        assert "<p>" not in r["synopsis"] and "&amp;" not in r["synopsis"]


class TestBuildDataset:
    def test_dedup_keeps_longer_synopsis(self):
        df = build_dataset([
            raw(product_no=1, synopsis="짧은 줄거리인데 길이는 채운다" * 2),
            raw(product_no=1, synopsis="훨씬 더 긴 줄거리가 들어 있는 레코드" * 5),
        ], 20)
        assert len(df) == 1
        assert "훨씬 더 긴" in df.iloc[0]["synopsis"]

    def test_nullable_int_columns(self):
        # 제목을 달리해야 한다 — 제목·출판사·줄거리가 같으면 같은 작품으로 합쳐진다
        df = build_dataset([raw(), raw(product_no=2, title="다른작품", interest_count=None)], 20)
        assert str(df["interest_count"].dtype) == "Int64"
        assert df["interest_count"].isna().sum() == 1

    def test_empty_input(self):
        assert build_dataset([], 20).empty


class TestProfile:
    def test_records_distributions_used_for_thresholds(self):
        df = build_dataset(
            [raw(product_no=i, title=f"작품{i}", interest_count=i * 100) for i in range(1, 11)], 20
        )
        p = compute_profile(10, df)
        assert p["valid_corpus_records"] == 10
        assert p["interest_coverage"] == 1.0
        # 품질 하한 임계를 이 분포에서 정한다
        assert p["interest_count"]["median"] == 550.0
        assert "median" in p["synopsis_chars"]


class TestPreviewEditions:
    """같은 작품의 미리보기판(화수 적고 관심 0)과 본편이 함께 올라온다 — 6% 실측."""

    def _pair(self):
        return [
            raw(product_no=13114576, title="아폴론 저축은행", publisher="요다",
                episode_count=1, interest_count=0),
            raw(product_no=12283752, title="아폴론 저축은행", publisher="요다",
                episode_count=89, interest_count=13000),
        ]

    def test_keeps_full_edition_not_preview(self):
        df = build_dataset(self._pair(), 20)
        assert len(df) == 1
        assert df.iloc[0]["item_id"] == 12283752
        assert df.iloc[0]["episode_count"] == 89

    def test_order_does_not_matter(self):
        df = build_dataset(list(reversed(self._pair())), 20)
        assert df.iloc[0]["item_id"] == 12283752

    def test_different_works_are_not_merged(self):
        """제목이 같아도 출판사·줄거리가 다르면 다른 작품이다."""
        df = build_dataset([
            raw(product_no=1, title="검신", publisher="A", synopsis="첫 번째 이야기입니다." * 3),
            raw(product_no=2, title="검신", publisher="B", synopsis="완전히 다른 이야기." * 3),
        ], 20)
        assert len(df) == 2

    def test_missing_episode_count_does_not_win(self):
        """화수 결측이 최댓값으로 정렬돼 미리보기가 살아남으면 안 된다."""
        df = build_dataset([
            raw(product_no=1, title="X", publisher="P", episode_count=None, interest_count=0),
            raw(product_no=2, title="X", publisher="P", episode_count=120, interest_count=900),
        ], 20)
        assert len(df) == 1 and df.iloc[0]["item_id"] == 2
