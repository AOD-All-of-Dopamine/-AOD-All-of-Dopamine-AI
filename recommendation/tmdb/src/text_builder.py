"""TMDB semantic_text — **Steam 관례를 그대로 따른다.**

Steam:   Description: … / Tags: … / Genres: … / Modes: …
웹소설:  제목: … / 장르: … / 줄거리: …
TMDB:    줄거리: … / 키워드: …(상위 15) / 장르: …

**제목은 넣지 않는다.** 웹소설이 제목을 넣은 이유는 웹소설 제목이 로그라인이기
때문이다(「나 혼자만 레벨업」). 영화·드라마 제목은 로그라인이 아니고,
프랜차이즈 이름이 속편을 한 덩어리로 뭉치게 만든다 — 교차 도메인 추천에서
「반지의 제왕 1·2·3」이 서로를 끌어당기는 건 원하는 신호가 아니다.

키워드 상한 15는 Steam 의 top-15 태그와 맞춘 것이다.

품질 필터는 웹소설의 기존 관례 min_text_chars=20 을 재사용한다.
결과를 보고 고른 값이 아니어야 하기 때문이다(사전 등록 h43 보정 1).
"""
import json
from pathlib import Path

import pandas as pd

MAX_KEYWORDS = 15
MIN_TEXT_CHARS = 20        # webnovel/src/config.py 관례 재사용


def build_semantic_text(overview: str, keywords: list, genres: list) -> str:
    parts = [f"줄거리: {overview}"]
    if keywords:
        parts.append(f"키워드: {', '.join(keywords[:MAX_KEYWORDS])}")
    if genres:
        parts.append(f"장르: {', '.join(genres)}")
    return "\n".join(parts)


def build_dataset(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        if not Path(p).exists():
            continue
        for line in open(p):
            d = json.loads(line)
            ov, lang = d['overview'], 'ko'
            if not ov:
                ov, lang = d.get('overview_en', ''), 'en'
            if not ov:
                continue
            text = build_semantic_text(ov, d['keywords'], d['genres'])
            if len(text) < MIN_TEXT_CHARS:
                continue
            rows.append({
                'item_id': f"{d['media']}_{d['id']}",
                'tmdb_id': d['id'], 'media': d['media'], 'name': d['title'],
                'overview': ov, 'lang': lang,
                'genres': d['genres'], 'keywords': d['keywords'],
                'n_keywords': len(d['keywords']), 'overview_len': len(ov),
                'vote_count': d['vote_count'], 'vote_average': d['vote_average'],
                'date': d['date'], 'adult': d['adult'],
                'semantic_text': text,
            })
    df = pd.DataFrame(rows).drop_duplicates('item_id').reset_index(drop=True)
    return df
