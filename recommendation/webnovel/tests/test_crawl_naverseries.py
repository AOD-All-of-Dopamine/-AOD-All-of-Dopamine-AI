"""크롤러 파싱 계약. 네트워크 없이 저장된 HTML 조각으로 돈다."""
import pytest

from src.crawl_naverseries import (
    clean_title,
    is_adult,
    parse_detail,
    parse_info_list,
    parse_korean_count,
)

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    pytest.skip("bs4 미설치", allow_module_level=True)


# 실제 페이지(productNo=14504924)에서 필요한 부분만 추린 픽스처.
# 구조가 바뀌면 이 테스트가 먼저 깨져야 한다 — 조용히 빈 값이 들어가는 게 최악이다.
DETAIL_HTML = """
<html><head>
<meta property="og:title" content="FA 먹튀 선수가 돈값 하기 시작함"/>
<meta property="og:image" content="https://example.test/cover.jpg"/>
</head><body>
<div class="end_head">FA 먹튀 선수가 돈값 하기 시작함 평점 9.8 관심 2.5만 공유</div>
<div class="score_area">평점 9.8</div>
<h5 class="end_total_episode">총 <strong>104</strong> 화</h5>
<span id="commentCount">1,234</span>
<ul class="end_info"><li class="info_lst"><ul>
  <li>연재중</li>
  <li><a href="/novel/categoryProductList.series?genreCode=208">현판</a></li>
  <li><span>글</span><a href="#">유리몸</a></li>
  <li><span>출판사</span><a href="#">아르데오</a></li>
  <li>전체 이용가</li>
</ul></li></ul>
<div class="end_dsc"><div class="_synopsis">짧은 버전</div>
<div class="_synopsis">드디어 200억의 값어치를 하기 시작했다 접기</div></div>
</body></html>
"""


class TestKoreanCount:
    """네이버는 큰 수를 항상 축약해서 보여준다. 이게 인기도 신호의 유일한 입력이다."""

    @pytest.mark.parametrize("raw,expected", [
        ("2억 5,006만", 250_060_000),
        ("139.3만", 1_393_000),
        ("2.5천", 2_500),
        ("1,393,475", 1_393_475),
        ("2", 2),
        ("0", 0),
    ])
    def test_parses(self, raw, expected):
        assert parse_korean_count(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   ", "없음"])
    def test_returns_none_when_unparseable(self, raw):
        assert parse_korean_count(raw) is None


class TestCleanTitle:
    @pytest.mark.parametrize("raw,expected", [
        ("죽여주는 호텔지배인 [독점]", "죽여주는 호텔지배인"),
        ("전당포는 영업 중 [단행본]", "전당포는 영업 중"),
        ("망장천 [독점][단행본]", "망장천"),
        ("개룡전기", "개룡전기"),
    ])
    def test_strips_bracket_tags(self, raw, expected):
        assert clean_title(raw) == expected

    def test_none_becomes_empty(self):
        assert clean_title(None) == ""


class TestInfoList:
    def test_extracts_all_fields(self):
        info = parse_info_list(BeautifulSoup(DETAIL_HTML, "lxml"))
        assert info["status"] == "연재중"
        assert info["author"] == "유리몸"
        assert info["publisher"] == "아르데오"
        assert info["age_rating"] == "전체 이용가"
        assert info["genres"] == ["현판"]

    def test_author_and_publisher_are_not_genres(self):
        """작가/출판사도 <a> 링크라 라벨을 안 보면 장르로 새어 들어간다."""
        info = parse_info_list(BeautifulSoup(DETAIL_HTML, "lxml"))
        assert "유리몸" not in info["genres"]
        assert "아르데오" not in info["genres"]

    def test_missing_info_block_returns_empty(self):
        info = parse_info_list(BeautifulSoup("<html><body></body></html>", "lxml"))
        assert info["genres"] == [] and info["author"] == ""

    def test_illustrator_credit_does_not_leak_into_genres(self):
        """`그림`(삽화가) 라벨이 있는 작품에서 삽화가 이름이 장르로 새던 버그.

        라벨 화이트리스트가 아니라 링크 대상(categoryProductList)으로 판정해야 한다.
        """
        html = """<ul class="info_lst">
          <li class="ing"><span>연재중</span></li>
          <li><span><a href="/novel/categoryProductList.series?genreCode=202">판타지</a></span></li>
          <li><span>글</span><a href="/search/search.series?q=문백경">문백경</a></li>
          <li><span>그림</span><a href="/search/search.series?q=망기">망기</a></li>
          <li><span>출판사</span><a href="/search/search.series?q=제이비북스">제이비북스</a></li>
        </ul>"""
        info = parse_info_list(BeautifulSoup(html, "lxml"))
        assert info["genres"] == ["판타지"]
        assert info["author"] == "문백경"
        assert info["publisher"] == "제이비북스"


class TestParseDetail:
    def test_full_record(self):
        rec = parse_detail(DETAIL_HTML, "14504924")
        assert rec["product_no"] == 14504924
        assert rec["title"] == "FA 먹튀 선수가 돈값 하기 시작함"
        assert rec["genres"] == ["현판"]
        assert rec["author"] == "유리몸"
        assert rec["status"] == "연재중"
        assert rec["rating"] == 9.8
        assert rec["episode_count"] == 104
        assert rec["comment_count"] == 1234
        assert rec["cover_image"] == "https://example.test/cover.jpg"

    def test_interest_count_comes_from_end_head(self):
        """Java 의 1차 셀렉터 `a.btn_download > span` 은 현재 페이지에 없다.

        end_head 텍스트 파싱이 유일한 경로이고, 이게 품질 하한의 입력이다.
        """
        assert parse_detail(DETAIL_HTML, "1")["interest_count"] == 25_000

    def test_synopsis_takes_longest_node_and_strips_fold_marker(self):
        """접힘/펼침 두 노드가 있어 짧은 쪽을 잡으면 본문을 잃는다."""
        syn = parse_detail(DETAIL_HTML, "1")["synopsis"]
        assert syn == "드디어 200억의 값어치를 하기 시작했다"
        assert "접기" not in syn

    def test_adult_page_is_rejected(self):
        html = '<html><body><div id="adult_msg">19세</div></body></html>'
        assert parse_detail(html, "1") is None

    def test_enctp_19_is_rejected(self):
        html = '<html><head><meta property="og:title" content="X"/></head>' \
               '<body><input name="enctp" value="19"/></body></html>'
        assert parse_detail(html, "1") is None

    def test_missing_title_is_rejected(self):
        assert parse_detail("<html><body></body></html>", "1") is None

    def test_missing_optional_fields_become_none(self):
        """필드가 없다고 죽으면 안 된다 — 페이지 구조가 작품마다 조금씩 다르다."""
        html = '<html><head><meta property="og:title" content="제목만 있는 작품"/></head>' \
               '<body></body></html>'
        rec = parse_detail(html, "7")
        assert rec["title"] == "제목만 있는 작품"
        assert rec["rating"] is None
        assert rec["interest_count"] is None
        assert rec["episode_count"] is None
        assert rec["genres"] == []


class TestIsAdult:
    def test_clean_page_is_not_adult(self):
        assert is_adult(BeautifulSoup(DETAIL_HTML, "lxml")) is False

    def test_enctp_other_value_is_not_adult(self):
        soup = BeautifulSoup('<input name="enctp" value="15"/>', "lxml")
        assert is_adult(soup) is False
