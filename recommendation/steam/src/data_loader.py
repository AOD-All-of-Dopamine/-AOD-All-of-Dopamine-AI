# src/data_loader.py
import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _parse_str_list(obj: dict, key: str) -> list[str]:
    values = obj.get(key) or []
    if not isinstance(values, list):
        return []
    return [c for v in values if isinstance(v, str) and (c := clean_text(v))]


def parse_genres(obj: dict) -> list[str]:
    return _parse_str_list(obj, "genres")


def parse_categories(obj: dict) -> list[str]:
    return _parse_str_list(obj, "categories")


def parse_metacritic(obj: dict) -> tuple[bool, int | None]:
    m = obj.get("metacritic")
    if isinstance(m, dict) and isinstance(m.get("score"), (int, float)):
        return True, int(m["score"])
    return False, None


def parse_recommendations(obj: dict) -> tuple[bool, int | None]:
    r = obj.get("recommendations")
    if isinstance(r, dict) and isinstance(r.get("total"), (int, float)):
        return True, int(r["total"])
    return False, None


import json
from pathlib import Path

import pandas as pd

from src.config import ARTIFACTS_DIR, ensure_artifacts_dir, load_config


def iter_records(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def record_to_row(obj: dict, min_desc_chars: int) -> dict | None:
    if obj.get("type") != "game":
        return None
    appid = obj.get("steam_appid")
    if not isinstance(appid, int):
        return None
    name = clean_text(obj.get("name", ""))
    if not name:
        return None
    desc = clean_text(obj.get("short_description", ""))
    if len(desc) < min_desc_chars:
        return None
    has_meta, meta_score = parse_metacritic(obj)
    has_rec, rec_total = parse_recommendations(obj)
    return {
        "steam_appid": appid,
        "name": name,
        "short_description": desc,
        "genres": parse_genres(obj),
        "categories": parse_categories(obj),
        "has_metacritic": has_meta,
        "metacritic_score": meta_score,
        "has_recommendations": has_rec,
        "recommendations_total": rec_total,
    }


def build_dataset(records, min_desc_chars: int) -> pd.DataFrame:
    rows = [r for obj in records if (r := record_to_row(obj, min_desc_chars))]
    df = pd.DataFrame(rows)
    # dedup: description 더 긴 레코드 우선, 동률이면 최초 등장 (stable sort)
    df["_desc_len"] = df["short_description"].str.len()
    df = df.sort_values("_desc_len", ascending=False, kind="mergesort")
    df = df.drop_duplicates(subset="steam_appid", keep="first")
    df = df.drop(columns="_desc_len").reset_index(drop=True)
    df["metacritic_score"] = df["metacritic_score"].astype("Int64")
    df["recommendations_total"] = df["recommendations_total"].astype("Int64")
    return df


def join_ranking(df: pd.DataFrame, ranking_path: str | Path | None) -> pd.DataFrame:
    if ranking_path is None or not Path(ranking_path).exists():
        df["steam_rank"] = pd.NA
        df["steam_rank"] = df["steam_rank"].astype("Int64")
        return df
    ranking = pd.read_parquet(ranking_path)[["steam_appid", "steam_rank"]]
    df = df.merge(ranking, on="steam_appid", how="left")
    df["steam_rank"] = df["steam_rank"].astype("Int64")
    return df


def compute_profile(raw_count: int, df: pd.DataFrame) -> dict:
    n = len(df)
    return {
        "total_raw_records": raw_count,
        "valid_corpus_records": n,
        "short_description_coverage": 1.0 if n else 0.0,
        "genres_coverage": float((df["genres"].str.len() > 0).mean()) if n else 0.0,
        "metacritic_coverage": float(df["has_metacritic"].mean()) if n else 0.0,
        "recommendations_coverage": float(df["has_recommendations"].mean()) if n else 0.0,
    }


def main():
    cfg = load_config()
    out = ensure_artifacts_dir()
    raw = list(iter_records(cfg["data"]["input_path"]))
    df = build_dataset(raw, cfg["data"]["min_description_chars"])
    df = join_ranking(df, cfg["data"].get("ranking_path"))
    profile = compute_profile(len(raw), df)
    df.to_parquet(out / "dataset.parquet", index=False)
    with open(out / "dataset_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
