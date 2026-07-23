from collections import Counter
from pathlib import Path

import pandas as pd

from personalization.seed_loader import SeedLoader
from personalization.candidate_retriever import CandidateRetriever
from personalization.score_aggregator import ScoreAggregator
from personalization.personalized_ranker import PersonalizedRanker
from trend.trend_ranker import TrendRanker

PROFILES = [
    ("Cozy Craft", [413150, 105600, 648800]),
    ("Survival", [108600, 219740, 242760]),
    ("FPS", [730, 578080, 359550]),
    ("RPG", [374320, 292030, 489830]),
    ("Mixed FPS+Cozy+Cities", [730, 413150, 255710]),
    ("Vehicle Sim", [227300, 284160, 244210]),
    ("Strategy", [289070, 268500, 281990]),
    ("Indie Platformer", [367520, 588650, 504230]),
    ("Mixed Survival+FPS", [108600, 242760, 730]),
    ("Classic Multi", [4000, 440, 550]),
]

TOP_N = 20
TREND_WEIGHTS = [0.0, 0.005, 0.01, 0.02]
STRATEGY = "max"

EVAL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "trend_v1"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

NAME_MAP_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "s1_v2" / "dataset.parquet"
NAME_MAP = pd.read_parquet(NAME_MAP_PATH).set_index("steam_appid")["name"].to_dict()


def generate_all():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()
    base_ranker = PersonalizedRanker()
    trend_rankers = {w: TrendRanker(trend_weight=w) for w in TREND_WEIGHTS if w > 0}

    all_results = []

    for profile_name, liked in PROFILES:
        seed_embs = loader.load(liked)
        sim = retriever.compute_similarity_matrix(seed_embs)
        corpus = retriever.full_corpus_frame()
        results = aggregator.aggregate_all(sim, seed_embs, corpus, strategies=[STRATEGY])

        for tw, ranker in [(0.0, base_ranker)] + [(w, trend_rankers[w]) for w in TREND_WEIGHTS if w > 0]:
            ranked = ranker.rank(results[STRATEGY], exclude_appids=set(liked), top_n=TOP_N)
            ranked = ranked.copy()
            ranked["profile"] = profile_name
            ranked["trend_weight"] = tw
            ranked["liked_appids"] = str(liked)
            all_results.append(ranked)

    return pd.concat(all_results, ignore_index=True)


def compute_diff(all_df: pd.DataFrame):
    print("=" * 70)
    print("T1-14: AUTOMATIC DIFF ANALYSIS")
    print("=" * 70)

    weights = sorted(all_df["trend_weight"].unique())
    profiles = all_df["profile"].unique()

    # Top-10 overlap
    print("\n--- Top-10 Overlap vs Baseline (trend_weight=0) ---")
    for pname in profiles:
        print(f"\n  {pname}:")
        baseline = set(
            all_df[(all_df["profile"] == pname) & (all_df["trend_weight"] == 0.0) & (all_df["rank"] <= 10)]["steam_appid"]
        )
        for tw in sorted(w for w in weights if w > 0):
            variant = set(
                all_df[(all_df["profile"] == pname) & (all_df["trend_weight"] == tw) & (all_df["rank"] <= 10)]["steam_appid"]
            )
            overlap = len(baseline & variant)
            new_ = variant - baseline
            lost_ = baseline - variant
            new_names = [NAME_MAP.get(a, "?")[:20] for a in list(new_)[:5]]
            lost_names = [NAME_MAP.get(a, "?")[:20] for a in list(lost_)[:5]]
            print(f"    tw={tw:.1%}: overlap={overlap}/10  new={len(new_)}  lost={len(lost_)}")
            if new_:
                print(f"      New entrants: {', '.join(new_names)}")
            if lost_:
                print(f"      Displaced: {', '.join(lost_names)}")

    # Extreme jumps
    print("\n--- Extreme Rank Jumps (|delta| >= 30 for same profile) ---")
    jumps_found = 0
    for tw in sorted(w for w in weights if w > 0):
        for pname in profiles:
            base = all_df[(all_df["profile"] == pname) & (all_df["trend_weight"] == 0.0)].set_index("steam_appid")
            var = all_df[(all_df["profile"] == pname) & (all_df["trend_weight"] == tw)].set_index("steam_appid")
            common = base.index.intersection(var.index)
            for aid in common:
                delta = var.loc[aid, "rank"] - base.loc[aid, "rank"]
                if abs(delta) >= 30:
                    jumps_found += 1
                    nm = NAME_MAP.get(int(aid), "?")[:30]
                    print(f"    {nm:30s}  {pname:25s}  tw={tw:.1%}  delta={delta:+d}")
    if jumps_found == 0:
        print("    No extreme jumps (|delta| >= 30) found.")

    # Saturation check
    print("\n--- Candidate Saturation (appears in Top-10 of N profiles) ---")
    for tw in weights:
        top10 = all_df[(all_df["trend_weight"] == tw) & (all_df["rank"] <= 10)]
        appearances = top10.groupby("steam_appid")["profile"].nunique()
        multi = appearances[appearances >= 2].sort_values(ascending=False)
        if len(multi) > 0:
            print(f"  tw={tw:.1%}: {len(multi)} candidates in 2+ profiles")
            for aid in multi.head(10).index:
                cnt = multi[aid]
                print(f"    {NAME_MAP.get(int(aid), '?'):40s}  appears in {cnt} profiles")
        else:
            print(f"  tw={tw:.1%}: No multi-profile saturation")

    # New entrants detail (for trend_weight with most changes)
    print("\n--- New Entrants Detail (Top profile) ---")
    for tw in sorted(w for w in weights if w > 0):
        tw_df = all_df[all_df["trend_weight"] == tw]
        base_df = all_df[all_df["trend_weight"] == 0.0]
        merged = tw_df.merge(
            base_df[["profile", "steam_appid", "rank"]],
            on=["profile", "steam_appid"],
            how="left",
            suffixes=("", "_baseline"),
        )
        new_entrants = merged[merged["rank_baseline"].isna()]
        if len(new_entrants) > 0:
            print(f"\n  tw={tw:.1%}: {len(new_entrants)} new entrants (not in baseline Top-{TOP_N})")
            for _, r in new_entrants.head(10).iterrows():
                nm = NAME_MAP.get(int(r["steam_appid"]), "?")[:35]
                print(f"    {r['profile']:30s}  #{r['rank']:2d}  {nm:35s}  score={r['final_score']:.4f}")


def main():
    all_df = generate_all()
    path = EVAL_DIR / "trend_eval_profiles.parquet"
    all_df.to_parquet(path, index=False)
    print(f"Saved: {path}")

    compute_diff(all_df)

    Path(EVAL_DIR / ".trend_eval_done").touch()


if __name__ == "__main__":
    main()
