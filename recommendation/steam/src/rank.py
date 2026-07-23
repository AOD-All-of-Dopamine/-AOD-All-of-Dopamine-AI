import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ensure_artifacts_dir, load_config

RANKING_DIR = Path("artifacts/s1_v2")


def corpus_percentile(dataset_path: Path, column: str, out_column: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    known = df[column].notna()
    out = df[["steam_appid"]].copy()
    col = out_column or f"{column}_percentile"
    out[col] = 0.0
    out.loc[known, col] = (
        df.loc[known, column].rank(method="average", pct=True)
    )
    return out[["steam_appid", col]]


def build_corpus_percentile(dataset_path: Path) -> pd.DataFrame:
    return corpus_percentile(dataset_path, "recommendations_total", "recommendations_percentile")


def build_metacritic_signal(dataset_path: Path) -> pd.DataFrame:
    metacritic_path = Path("data/metacritic.parquet") if Path("data/metacritic.parquet").exists() else Path("../data/metacritic.parquet")
    meta = pd.read_parquet(metacritic_path)
    df = pd.read_parquet(dataset_path)
    merged = df[["steam_appid"]].merge(meta, on="steam_appid", how="left")
    known = merged["metacritic_score"].notna()
    merged["metacritic_signal"] = 0.0
    if known.any():
        merged.loc[known, "metacritic_signal"] = (
            merged.loc[known, "metacritic_score"].rank(method="average", pct=True)
        )
        merged.loc[known, "metacritic_signal"] = (
            merged.loc[known, "metacritic_signal"].sub(0.5).mul(2).clip(lower=0)
        )
    return merged[["steam_appid", "metacritic_signal"]]


def compute_boosted_scores(
    top: pd.DataFrame,
    percentile_map: pd.DataFrame,
    max_boost: float = 0.05,
    extra_signals: list[tuple[pd.DataFrame, str, str, float]] | None = None,
) -> pd.DataFrame:
    df = top.copy()
    df = df.merge(percentile_map, left_on="candidate_steam_appid", right_on="steam_appid", how="left")
    df["boost_rec"] = df["recommendations_percentile"].fillna(0.0) * max_boost
    df["boost_total"] = df["boost_rec"]
    if extra_signals:
        for sig_map, sig_col, out_prefix, sig_max_boost in extra_signals:
            df = df.merge(sig_map, left_on="candidate_steam_appid", right_on="steam_appid", how="left")
            df[f"boost_{out_prefix}"] = df[sig_col].fillna(0.0) * sig_max_boost
            df["boost_total"] = df["boost_total"] + df[f"boost_{out_prefix}"]
    df["final_score"] = df["similarity"] * (1.0 + df["boost_total"])
    return df


def rerank_per_anchor(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for _, grp in df.groupby("anchor_steam_appid", sort=False):
        grp = grp.sort_values("final_score", ascending=False).reset_index(drop=True)
        grp["rank"] = grp.index + 1
        out = pd.concat([out, grp], ignore_index=True)
    return out


def _parse_float_arg(prefix: str) -> float | None:
    for i, arg in enumerate(sys.argv):
        if arg.startswith(f"{prefix}="):
            return float(arg.split("=")[1])
        if arg == prefix and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return None


def main():
    if "--r-all-debug" in sys.argv:
        _run_all_debug()
        return
    if "--r1" in sys.argv:
        _run_r1()
        return
    r1_boost = _parse_float_arg("--r1-boost")
    if r1_boost is not None:
        _run_r1(r1_boost)
        return
    meta_boost = _parse_float_arg("--meta-boost")
    if meta_boost is not None:
        _run_r2(meta_boost)
    else:
        _run_r2()


def _run_r1(rec_boost: float | None = None):
    cfg = load_config()
    out = ensure_artifacts_dir()
    dataset_path = out / "dataset.parquet"
    percentile = build_corpus_percentile(dataset_path)
    print(f"  corpus percentile: {percentile['recommendations_percentile'].notna().sum()} mapped, "
          f"known (non-zero): {(percentile['recommendations_percentile'] > 0).sum()}")
    top = pd.read_parquet(RANKING_DIR / "qwen_top100.parquet")
    max_boost = rec_boost if rec_boost is not None else 0.05
    boosted = compute_boosted_scores(top, percentile, max_boost=max_boost)
    ranked = rerank_per_anchor(boosted)
    ranked["rank_original"] = ranked.groupby("anchor_steam_appid")["similarity"].rank(ascending=False, method="first").astype(int)
    ranked["rank_delta"] = ranked["rank_original"] - ranked["rank"]
    boost_label = f"{int(max_boost * 100):03d}"
    ranked["experiment_id"] = f"steam_s1_ranking_r1_{boost_label}"
    out_name = f"ranked_top100_r1_{boost_label}.parquet"
    ranked.to_parquet(out / out_name, index=False)
    print(f"wrote {out / out_name} rows={len(ranked)}")
    print(f"  boost_rec: mean={ranked['boost_rec'].mean():.3f} max={ranked['boost_rec'].max():.3f} "
          f"p50={ranked['boost_rec'].quantile(0.5):.3f} p90={ranked['boost_rec'].quantile(0.9):.3f}")
    print(f"  boosted: {(ranked['rank_delta'] > 0).sum()}  demoted: {(ranked['rank_delta'] < 0).sum()}")


def _run_r2(meta_boost: float | None = None):
    cfg = load_config()
    out = ensure_artifacts_dir()
    dataset_path = Path(cfg["data"]["input_path"]).parent / "artifacts" / "s1_v2" / "dataset.parquet"
    dataset_path = out / "dataset.parquet"
    rec_percentile = build_corpus_percentile(dataset_path)
    print(f"  rec corpus percentile: {rec_percentile['recommendations_percentile'].notna().sum()} mapped, "
          f"known (non-zero): {(rec_percentile['recommendations_percentile'] > 0).sum()}")

    if meta_boost is not None:
        meta_signal = build_metacritic_signal(dataset_path)
        print(f"  meta signal: non-zero={((meta_signal['metacritic_signal'] > 0).sum())} "
              f"mean={meta_signal['metacritic_signal'].mean():.3f} "
              f"max={meta_signal['metacritic_signal'].max():.3f}")
        exp_id = f"steam_s1_ranking_r2_rec003_meta{int(meta_boost * 100):03d}"
        out_name = f"ranked_top100_r2_meta{int(meta_boost * 100):03d}.parquet"
    else:
        exp_id = "steam_s1_ranking_r2"
        out_name = "ranked_top100.parquet"

    top = pd.read_parquet(RANKING_DIR / "qwen_top100.parquet")
    if meta_boost is not None:
        boosted = compute_boosted_scores(top, rec_percentile, max_boost=0.03,
                                         extra_signals=[(meta_signal, "metacritic_signal", "metacritic", meta_boost)])
    else:
        boosted = compute_boosted_scores(top, rec_percentile)
    ranked = rerank_per_anchor(boosted)
    ranked["rank_original"] = ranked.groupby("anchor_steam_appid")["similarity"].rank(ascending=False, method="first").astype(int)
    ranked["rank_delta"] = ranked["rank_original"] - ranked["rank"]
    ranked["experiment_id"] = exp_id
    ranked.to_parquet(out / out_name, index=False)
    print(f"wrote {out / out_name} rows={len(ranked)}")
    print(f"  anchors={ranked['anchor_steam_appid'].nunique()}")
    for col in ["boost_rec", "boost_metacritic", "boost_total"]:
        if col in ranked.columns:
            print(f"  {col}: mean={ranked[col].mean():.3f} max={ranked[col].max():.3f} "
                  f"p50={ranked[col].quantile(0.5):.3f} p90={ranked[col].quantile(0.9):.3f}")
    print(f"  boosted (final rank < original): {(ranked['rank_delta'] > 0).sum()}")
    print(f"  demoted (final rank > original): {(ranked['rank_delta'] < 0).sum()}")

    ranked["rank_jump"] = (ranked["rank_original"] - ranked["rank"]).abs()
    print(f"  rank_jump: median={ranked['rank_jump'].median():.0f} p90={ranked['rank_jump'].quantile(0.9):.0f} "
          f"p95={ranked['rank_jump'].quantile(0.95):.0f}")
    extreme = ranked[ranked["rank_jump"] >= 50]
    print(f"  extreme jumps (>=50): {len(extreme)}")
    t10 = ranked[ranked["rank"] <= 10]
    freq = t10.groupby("candidate_name").size()
    if len(freq):
        print(f"  Top-10 unique candidates: {freq.count()}  max_freq={freq.max()}")
        over3 = freq[freq >= 4]
        if len(over3):
            print(f"  candidates >=4 anchors: {len(over3)} — {', '.join(f'{n}({c})' for n,c in over3.items())}")


def _run_all_debug():
    ranking_src = Path(__file__).resolve().parent.parent / "data" / "steam_ranking.parquet"
    ranking_df = pd.read_parquet(ranking_src) if ranking_src.exists() else None
    top = pd.read_parquet(RANKING_DIR / "qwen_top100.parquet")
    if ranking_df is not None:
        top = top.merge(ranking_df[["steam_appid", "steam_rank"]], left_on="candidate_steam_appid", right_on="steam_appid", how="left")
        top["steam_rank"] = top["steam_rank"].astype("Int64")

    def _rec_boost(rec):
        if rec is None or pd.isna(rec):
            return 0.0
        v = np.log10(max(rec, 1)) / 6.0
        return min(v, 0.10)

    def _meta_boost(row):
        if not row["has_metacritic"] or pd.isna(row["metacritic_score"]):
            return 0.0
        return 0.08 if row["metacritic_score"] >= 75 else 0.04

    def _ts_boost(rank):
        if rank is None or pd.isna(rank):
            return 0.0
        return min(1.0 / np.log2(float(rank) + 1), 0.10)

    top["boost_rec"] = top["recommendations_total"].apply(_rec_boost)
    top["boost_meta"] = top.apply(_meta_boost, axis=1)
    top["boost_ts"] = top["steam_rank"].apply(_ts_boost)
    top["boost_total"] = (top["boost_rec"] + top["boost_meta"] + top["boost_ts"]).clip(upper=0.25)
    top["final_score"] = top["similarity"] * (1.0 + top["boost_total"])
    ranked = rerank_per_anchor(top)
    ranked["rank_original"] = ranked.groupby("anchor_steam_appid")["similarity"].rank(ascending=False, method="first").astype(int)
    ranked["rank_delta"] = ranked["rank_original"] - ranked["rank"]
    ranked["experiment_id"] = "steam_s1_ranking_r_all_debug"
    ranked.to_parquet(RANKING_DIR / "ranked_top100_debug.parquet", index=False)
    print(f"wrote {RANKING_DIR / 'ranked_top100_debug.parquet'} rows={len(ranked)}")


if __name__ == "__main__":
    main()
