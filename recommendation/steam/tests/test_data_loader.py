# tests/test_data_loader.py
from src.data_loader import (
    clean_text,
    parse_categories,
    parse_genres,
    parse_metacritic,
    parse_recommendations,
)


def test_clean_text_html_and_whitespace():
    assert clean_text("  A &amp; B <br>  fun   game ") == "A & B fun game"


def test_clean_text_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_parse_genres_flat_strings():
    obj = {"genres": ["액션", "인디"]}
    assert parse_genres(obj) == ["액션", "인디"]


def test_parse_categories_flat_strings():
    obj = {"categories": ["멀티플레이어", "PvP", "온라인 PvP"]}
    assert parse_categories(obj) == ["멀티플레이어", "PvP", "온라인 PvP"]


def test_parse_genres_missing_or_bad():
    assert parse_genres({}) == []
    assert parse_genres({"genres": None}) == []
    assert parse_genres({"genres": "액션"}) == []
    assert parse_genres({"genres": ["액션", 3, " "]}) == ["액션"]


def test_parse_metacritic():
    assert parse_metacritic({"metacritic": {"score": 88}}) == (True, 88)
    assert parse_metacritic({}) == (False, None)
    assert parse_metacritic({"metacritic": {"url": "x"}}) == (False, None)


def test_parse_recommendations():
    assert parse_recommendations({"recommendations": {"total": 500}}) == (True, 500)
    assert parse_recommendations({}) == (False, None)


from src.data_loader import build_dataset


def _rec(appid, name="Game", desc="A" * 40, type_="game", genres=None, rec_total=None, meta=None):
    obj = {
        "steam_appid": appid,
        "name": name,
        "type": type_,
        "short_description": desc,
        "genres": genres or [],
        "categories": [],
    }
    if rec_total is not None:
        obj["recommendations"] = {"total": rec_total}
    if meta is not None:
        obj["metacritic"] = {"score": meta}
    return obj


def test_build_dataset_filters():
    records = [
        _rec(1),
        _rec(2, type_="dlc"),            # type 필터
        _rec(3, desc="short"),           # 30자 미만 필터
        _rec(4, name="  "),              # 빈 name 필터
        _rec(5),
    ]
    df = build_dataset(records, min_desc_chars=30)
    assert sorted(df["steam_appid"]) == [1, 5]


def test_build_dataset_dedup_longer_desc_wins():
    records = [
        _rec(1, desc="A" * 40),
        _rec(1, desc="B" * 100),   # 더 긴 쪽이 살아남아야 함
    ]
    df = build_dataset(records, min_desc_chars=30)
    assert len(df) == 1
    assert df.iloc[0]["short_description"] == "B" * 100


def test_build_dataset_quality_flags():
    records = [_rec(1, rec_total=500, meta=88), _rec(2)]
    df = build_dataset(records, min_desc_chars=30)
    row1 = df[df["steam_appid"] == 1].iloc[0]
    row2 = df[df["steam_appid"] == 2].iloc[0]
    assert bool(row1["has_recommendations"]) and row1["recommendations_total"] == 500
    assert bool(row1["has_metacritic"]) and row1["metacritic_score"] == 88
    assert not bool(row2["has_recommendations"]) and not bool(row2["has_metacritic"])
