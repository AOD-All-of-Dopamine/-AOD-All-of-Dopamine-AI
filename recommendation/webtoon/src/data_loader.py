"""jsonl → dataset.parquet. 정제 규칙은 여기 한 곳에만 둔다."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.config import DATA_FILE, ensure_artifacts_dir
from src.text_builder import add_semantic_text

MIN_SYNOPSIS = 10          # 줄거리가 이보다 짧으면 의미 신호가 없다


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path or DATA_FILE)
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows).drop_duplicates(subset=["item_id"], keep="last")
    return df.reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """제외 규칙 — 전부 **내용 기준**이고 인기도로는 자르지 않는다."""
    n0 = len(df)
    df = df[df["name"].astype(str).str.len() > 0]
    df = df[~df["adult"].fillna(False).astype(bool)]                 # 성인
    df = df[df["age_type"].fillna("") != "RATE_19"]
    df = df[df["synopsis"].fillna("").astype(str).str.len() >= MIN_SYNOPSIS]
    df = df[df["level"].fillna("") == "WEBTOON"]                     # 도전만화 제외
    print(f"정제: {n0} → {len(df)} (성인·짧은 줄거리·비정식 제외)")
    return df.reset_index(drop=True)


def build(artifacts=None, path=None) -> pd.DataFrame:
    df = add_semantic_text(clean(load_raw(path)))
    out = ensure_artifacts_dir(artifacts)
    df.to_parquet(out / "dataset.parquet", index=False)
    print(f"저장 {len(df):,}편 → {out/'dataset.parquet'}")
    return df


if __name__ == "__main__":
    build()
