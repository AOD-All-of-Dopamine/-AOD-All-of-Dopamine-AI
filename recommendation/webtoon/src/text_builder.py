"""임베딩에 넣을 `semantic_text`.

웹소설 판(`제목 + 장르 + 줄거리`)을 본으로 삼되 **웹툰은 태그가 두껍다** — 작품당 10~12개
(`사이다`·`먼치킨`·`학원물`·`다크히어로`·`소설원작`…). 이건 Steam 태그에 가까운 밀도라
웹소설의 얇은 장르(1개)와 사정이 다르다.

그래서 표현을 두 갈래로 만들어 두고 **`diagnose_representation` 으로 변별력을 재고 고른다.**
측정 전에는 무엇이 기본인지 정하지 않는다.
    rep_v1 : 제목 + 장르 + 줄거리            (웹소설과 같은 모양, 대조군)
    rep_v2 : 제목 + 장르 + 태그 + 줄거리      (웹툰의 두꺼운 태그를 쓴다)

인기 수·평점·회차 수는 **넣지 않는다.** 의미 신호가 아니라 랭킹 신호이고, 랭커가 따로 다룬다.
"""
from __future__ import annotations
import pandas as pd

# 태그에서 뺄 것: 의미가 아니라 유통·홍보 라벨
_TAG_STOP = {"명작", "인기", "완결", "신작", "웹툰", "네이버웹툰"}


def _lst(v) -> list[str]:
    if v is None: return []
    if isinstance(v, str): return [v]
    try: return [str(x) for x in v if str(x)]
    except TypeError: return []


def build_semantic_text(name, genres, synopsis, tags=None, with_tags: bool = True) -> str:
    parts = [f"제목: {name}"]
    g = _lst(genres)
    if g: parts.append(f"장르: {', '.join(g)}")
    if with_tags:
        t = [x for x in _lst(tags) if x not in set(g) and x not in _TAG_STOP]
        if t: parts.append(f"태그: {', '.join(t[:12])}")
    if synopsis: parts.append(f"줄거리: {synopsis}")
    return "\n".join(parts)


def add_semantic_text(df: pd.DataFrame, with_tags: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["semantic_text"] = [
        build_semantic_text(r["name"], r.get("genres"), r.get("synopsis"),
                            r.get("tags"), with_tags)
        for _, r in df.iterrows()
    ]
    return df
