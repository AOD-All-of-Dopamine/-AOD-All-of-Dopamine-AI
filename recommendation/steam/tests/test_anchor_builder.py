# tests/test_anchor_builder.py
import pandas as pd

from src.anchor_builder import assign_buckets, primary_stratum


def _df(totals):
    return pd.DataFrame({
        "steam_appid": list(range(len(totals))),
        "has_recommendations": [t is not None for t in totals],
        "recommendations_total": pd.array(totals, dtype="Int64"),
    })


def test_assign_buckets():
    # 10개 known: 100..1000 → pandas 선형보간 q40=460, q80=820
    totals = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, None]
    buckets = assign_buckets(_df(totals)).tolist()
    assert buckets == [
        "KNOWN_LOW", "KNOWN_LOW", "KNOWN_LOW", "KNOWN_LOW",
        "KNOWN_MID", "KNOWN_MID", "KNOWN_MID", "KNOWN_MID",
        "KNOWN_HIGH", "KNOWN_HIGH",
        "UNKNOWN",
    ]


def test_primary_stratum_skips_non_core():
    assert primary_stratum(["인디", "RPG", "액션"]) == "RPG"
    assert primary_stratum(["액션", "인디"]) == "액션"


def test_primary_stratum_none_when_only_non_core():
    assert primary_stratum(["인디", "무료 플레이"]) is None
    assert primary_stratum([]) is None


import pytest

from src.anchor_builder import finalize_selection


def _corpus():
    rows = []
    for i in range(50):
        rows.append({
            "steam_appid": i,
            "name": f"Game {i}",
            "genres": ["인디", "액션"] if i % 2 == 0 else ["RPG"],
            "semantic_text": f"Description: game {i}",
            "bucket": "KNOWN_MID",
            "primary_stratum": "액션" if i % 2 == 0 else "RPG",
        })
    return pd.DataFrame(rows)


def _xlsx(tmp_path, y_ids):
    import pandas as pd
    df = pd.DataFrame({
        "steam_appid": list(range(50)),
        "include_YN": ["Y" if i in y_ids else "" for i in range(50)],
    })
    p = tmp_path / "cand.xlsx"
    df.to_excel(p, index=False)
    return p


def test_finalize_happy_path(tmp_path):
    df = _corpus()
    # 인디 포함 = 짝수 appid. Y를 홀수 위주 40개 선택 → 인디 0개
    y_ids = [i for i in range(1, 50, 2)] + [i for i in range(0, 32, 2)]  # 25 + 16 = 41 -> 40으로
    y_ids = y_ids[:40]
    anchors = finalize_selection(_xlsx(tmp_path, y_ids), df, {"anchor": {"final_count": 40, "indie_max_count": 20}})
    assert len(anchors) == 40
    assert list(anchors.columns) == ["steam_appid", "name", "genres", "semantic_text", "bucket", "primary_stratum"]


def test_finalize_rejects_wrong_count(tmp_path):
    df = _corpus()
    with pytest.raises(ValueError, match="40"):
        finalize_selection(_xlsx(tmp_path, [1, 3, 5]), df, {"anchor": {"final_count": 40, "indie_max_count": 20}})


def test_finalize_rejects_indie_over_cap(tmp_path):
    df = _corpus()
    y_ids = list(range(0, 40, 2)) + list(range(1, 41, 2))  # 인디 20 + 비인디 20 = 40
    y_ids = y_ids[:40]
    # cap을 5로 낮추면 실패해야 함
    with pytest.raises(ValueError, match="indie"):
        finalize_selection(_xlsx(tmp_path, y_ids), df, {"anchor": {"final_count": 40, "indie_max_count": 5}})
