# src/anchor_builder.py
import pandas as pd

ANCHOR_STRATA = [
    "액션", "어드벤처", "캐주얼", "전략",
    "시뮬레이션", "RPG", "스포츠", "레이싱",
]

NON_CORE_GENRE_LABELS = {
    "인디", "앞서 해보기", "무료 플레이", "대규모 멀티플레이어",
    "폭력적", "고어", "신체 노출", "선정적 콘텐츠",
}

BUCKETS = ["KNOWN_HIGH", "KNOWN_MID", "KNOWN_LOW", "UNKNOWN"]

# stratum당 quota (합계 10 = candidate_count 80 / strata 8)
BUCKET_QUOTA = {"KNOWN_HIGH": 3, "KNOWN_MID": 3, "KNOWN_LOW": 2, "UNKNOWN": 2}


def assign_buckets(df: pd.DataFrame) -> pd.Series:
    known = df.loc[df["has_recommendations"], "recommendations_total"].dropna()
    q80 = known.quantile(0.8)
    q40 = known.quantile(0.4)

    def bucket(row):
        if not row["has_recommendations"] or pd.isna(row["recommendations_total"]):
            return "UNKNOWN"
        t = row["recommendations_total"]
        if t >= q80:
            return "KNOWN_HIGH"
        if t >= q40:
            return "KNOWN_MID"
        return "KNOWN_LOW"

    return df.apply(bucket, axis=1)


def primary_stratum(genres: list[str]) -> str | None:
    for g in genres:
        if g in ANCHOR_STRATA:
            return g
    return None


import math
import sys

from src.config import ensure_artifacts_dir, load_config


def generate_candidates(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = cfg["experiment"]["seed"]
    total = cfg["anchor"]["candidate_count"]
    df = df.copy()
    df["bucket"] = assign_buckets(df)
    df["primary_stratum"] = df["genres"].apply(primary_stratum)

    parts = []
    for stratum in ANCHOR_STRATA:
        pool = df[df["genres"].apply(lambda gs: stratum in gs)]
        for bucket, quota in BUCKET_QUOTA.items():
            sub = pool[pool["bucket"] == bucket]
            n = min(quota, len(sub))
            if n:
                parts.append(sub.sample(n=n, random_state=seed))
    cand = pd.concat(parts).drop_duplicates(subset="steam_appid", keep="first")

    if len(cand) < total:  # 부족분: 추천 수 높은 순으로 충원
        rest = df[~df["steam_appid"].isin(cand["steam_appid"])]
        fill = rest.sort_values(
            "recommendations_total", ascending=False, na_position="last"
        ).head(total - len(cand))
        cand = pd.concat([cand, fill])

    cand = cand.head(total).reset_index(drop=True)
    cand["include_YN"] = ""
    cols = [
        "steam_appid", "name", "genres", "primary_stratum", "bucket",
        "recommendations_total", "short_description", "include_YN",
    ]
    return cand[cols]


def finalize_selection(xlsx_path, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = pd.read_excel(xlsx_path)
    y_ids = c.loc[
        c["include_YN"].astype(str).str.strip().str.upper() == "Y", "steam_appid"
    ].tolist()

    errors = []
    if len(y_ids) != cfg["anchor"]["final_count"]:
        errors.append(f"Y count = {len(y_ids)}, expected {cfg['anchor']['final_count']}")
    if len(set(y_ids)) != len(y_ids):
        errors.append("duplicate steam_appid in selection")
    corpus = df.set_index("steam_appid")
    missing = [a for a in y_ids if a not in corpus.index]
    if missing:
        errors.append(f"appids not in corpus: {missing}")
    if errors:
        raise ValueError("; ".join(errors))

    anchors = corpus.loc[y_ids].reset_index()
    indie_count = int(anchors["genres"].apply(lambda gs: "인디" in gs).sum())
    if indie_count > cfg["anchor"]["indie_max_count"]:
        raise ValueError(
            f"indie count {indie_count} > cap {cfg['anchor']['indie_max_count']}"
        )
    return anchors[
        ["steam_appid", "name", "genres", "semantic_text", "bucket", "primary_stratum"]
    ]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = load_config()
    out = ensure_artifacts_dir()
    df = pd.read_parquet(out / "dataset.parquet")

    if mode == "generate":
        cand = generate_candidates(df, cfg)
        path = out / "anchor_candidates_80.xlsx"
        cand.to_excel(path, index=False)
        print(f"wrote {path} rows={len(cand)}")
        print(cand["bucket"].value_counts().to_string())
        print(cand["primary_stratum"].value_counts(dropna=False).to_string())
    elif mode == "finalize":
        df = df.copy()
        df["bucket"] = assign_buckets(df)
        df["primary_stratum"] = df["genres"].apply(primary_stratum)
        anchors = finalize_selection(out / "anchor_candidates_80.xlsx", df, cfg)
        anchors.to_parquet(out / "anchors_40.parquet", index=False)
        print(f"wrote {out / 'anchors_40.parquet'} rows={len(anchors)}")
        print(anchors["bucket"].value_counts().to_string())
    else:
        raise SystemExit("usage: python -m src.anchor_builder [generate|finalize]")


if __name__ == "__main__":
    main()
