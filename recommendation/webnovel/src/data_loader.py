# src/data_loader.py
"""`webnovels.jsonl` → 정제된 `dataset.parquet`.

Steam 판의 구조(정제 → dedup → 프로파일)를 그대로 따르되 필드만 웹소설로 바꿨다.
컬럼 이름은 `steam_appid` 가 아니라 도메인 중립적인 `item_id` 를 쓴다 — 이 파이프라인의
아래쪽(personalization, postprocess, retrieve)은 전부 `item_id` 만 안다.
"""
import html
import json
import re
from pathlib import Path

import pandas as pd

from src.config import ensure_artifacts_dir, load_config, resolve_path

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# 연령 등급 정규화. 원본은 "전체 이용가" · "12세 이용가" · "15세 이용가" · "19세 이용가"
_AGE_RE = re.compile(r"(\d+)\s*세")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def parse_age(age_rating: str) -> int:
    """"15세 이용가" → 15, "전체 이용가" → 0. 모르면 0.

    19금은 크롤 단계에서 이미 걸러지지만, 목록 경로가 바뀌면 새는 경우가 있어 여기서도 본다.
    """
    if not isinstance(age_rating, str) or not age_rating:
        return 0
    m = _AGE_RE.search(age_rating)
    return int(m.group(1)) if m else 0


def parse_str_list(obj: dict, key: str) -> list[str]:
    values = obj.get(key) or []
    if not isinstance(values, list):
        return []
    return [c for v in values if isinstance(v, str) and (c := clean_text(v))]


def _int_or_none(v) -> int | None:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def record_to_row(obj: dict, min_text_chars: int) -> dict | None:
    """jsonl 한 줄 → dataset 한 행. 쓸 수 없는 레코드는 None.

    **길이 게이트를 시놉시스 단독으로 걸지 않는다.** 시놉시스는 대체로 충분히 길지만
    (49건 실측: 중앙값 482자, p25 346자, 30자 미만 0건 — Steam 의 `short_description`
    300자 내외보다 오히려 길다) **꼬리에 카피 문구만 있는 작품이 있다.** 실제로 전문이
    "드디어 200억의 값어치를 하기 시작했다"(22자)인 작품을 확인했다. 그런 작품도 제목이
    `"FA 먹튀 선수가 돈값 하기 시작함"` 처럼 그 자체로 로그라인이라 버릴 이유가 없다.
    그래서 **임베딩에 실제로 들어가는 텍스트(제목+장르+줄거리)** 합산 길이로 본다.
    """
    pid = _int_or_none(obj.get("product_no"))
    if pid is None:
        return None
    title = clean_text(obj.get("title", ""))
    if not title:
        return None
    # 판매중지 페이지가 리다이렉트로 섞여 들어오면 제목이 사이트명이 된다
    if title.upper() == "SERIES":
        return None

    synopsis = clean_text(obj.get("synopsis", ""))
    genres = parse_str_list(obj, "genres")
    if len(title) + len(synopsis) + sum(len(g) for g in genres) < min_text_chars:
        return None

    interest = _int_or_none(obj.get("interest_count"))
    rating = obj.get("rating")
    return {
        "item_id": pid,
        "name": title,
        "synopsis": synopsis,
        "genres": genres,
        "author": clean_text(obj.get("author", "")),
        # 시리즈 판정에 쓴다. 웹소설은 같은 출판사가 같은 시리즈를 내는 경향이 Steam 보다 강하다.
        "publisher": clean_text(obj.get("publisher", "")),
        "age_rating": clean_text(obj.get("age_rating", "")),
        "age_limit": parse_age(obj.get("age_rating", "")),
        "status": clean_text(obj.get("status", "")),
        "is_completed": clean_text(obj.get("status", "")) == "완결",
        # 관심 수 = 이 도메인의 인기도/품질 신호. Steam 의 recommendations_total 자리다.
        # Steam 은 결측이 깨끗한 이진 신호였지만 여기는 연속값이라 임계값으로 다뤄야 한다.
        "has_interest": interest is not None,
        "interest_count": interest,
        "comment_count": _int_or_none(obj.get("comment_count")),
        "episode_count": _int_or_none(obj.get("episode_count")),
        # 평점은 참여자 수가 없어 단독으로는 노이즈다(관심 2에 평점 10.0). 랭킹에 쓰지 않는다.
        "rating": float(rating) if isinstance(rating, (int, float)) else None,
        "cover_image": obj.get("cover_image") or "",
        "url": obj.get("url") or "",
    }


def iter_records(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def build_dataset(records, min_text_chars: int) -> pd.DataFrame:
    rows = [r for obj in records if (r := record_to_row(obj, min_text_chars))]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # dedup: 시놉시스가 더 긴 레코드 우선, 동률이면 최초 등장 (stable sort)
    df["_syn_len"] = df["synopsis"].str.len()
    df = df.sort_values("_syn_len", ascending=False, kind="mergesort")
    df = df.drop_duplicates(subset="item_id", keep="first")
    df = df.drop(columns="_syn_len").reset_index(drop=True)
    for col in ("interest_count", "comment_count", "episode_count"):
        df[col] = df[col].astype("Int64")
    return df


def compute_profile(raw_count: int, df: pd.DataFrame) -> dict:
    n = len(df)
    if not n:
        return {"total_raw_records": raw_count, "valid_corpus_records": 0}
    syn_len = df["synopsis"].str.len()
    interest = df["interest_count"].dropna()
    return {
        "total_raw_records": raw_count,
        "valid_corpus_records": n,
        "genres_coverage": float((df["genres"].str.len() > 0).mean()),
        "author_coverage": float((df["author"].str.len() > 0).mean()),
        "publisher_coverage": float((df["publisher"].str.len() > 0).mean()),
        "interest_coverage": float(df["has_interest"].mean()),
        "completed_ratio": float(df["is_completed"].mean()),
        # 시놉시스가 이 도메인의 핵심 약점이라 분포를 항상 기록한다
        "synopsis_chars": {
            "median": float(syn_len.median()),
            "p25": float(syn_len.quantile(0.25)),
            "p75": float(syn_len.quantile(0.75)),
            "under_30_ratio": float((syn_len < 30).mean()),
            "empty_ratio": float((syn_len == 0).mean()),
        },
        # 품질 하한(min_interest_count) 임계를 여기 분포를 보고 정한다
        "interest_count": {
            "median": float(interest.median()) if len(interest) else None,
            "p25": float(interest.quantile(0.25)) if len(interest) else None,
            "p75": float(interest.quantile(0.75)) if len(interest) else None,
            "p90": float(interest.quantile(0.90)) if len(interest) else None,
        },
        "genre_distribution": df["genres"].explode().value_counts().head(15).to_dict(),
    }


def main():
    cfg = load_config()
    out = ensure_artifacts_dir()
    raw = list(iter_records(resolve_path(cfg["data"]["input_path"])))
    df = build_dataset(raw, cfg["data"]["min_text_chars"])
    profile = compute_profile(len(raw), df)
    df.to_parquet(out / "dataset.parquet", index=False)
    with open(out / "dataset_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
