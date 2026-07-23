import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from personalized_retrieve import run_multi
from personalization.seed_loader import SeedLoader
from personalization.candidate_retriever import CandidateRetriever
from personalization.score_aggregator import ScoreAggregator
from personalization.personalized_ranker import PersonalizedRanker


def ndcg_at_k(rel_scores: list[float], k: int = 10) -> float:
    if not rel_scores:
        return 0.0
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel_scores[:k]))
    ideal = sorted(rel_scores, reverse=True)[:k]
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_from_judgments(
    judgments_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    score_col: str = "relevance",
    strategies: list[str] | None = None,
    top_k: list[int] | None = None,
) -> pd.DataFrame:
    if strategies is None:
        strategies = ["max", "mean", "top2_mean"]
    if top_k is None:
        top_k = [10]

    joined = judgments_df.merge(mapping_df, on="pair_key", how="inner")

    rows = []
    for strategy in strategies:
        score_key = f"{strategy}_score"
        rank_key = f"{strategy}_rank"

        scored = joined[joined[score_key].notna()].copy()

        for k in top_k:
            topk = scored.nlargest(k, score_key)
            rels = topk[score_col].tolist()
            rels = [float(r) for r in rels if pd.notna(r)]

            ndcg = ndcg_at_k(rels, k) if len(rels) > 0 else 0.0
            prec = (np.array(rels) >= 2).mean() if len(rels) > 0 else 0.0
            conf = np.mean(rels) if len(rels) > 0 else 0.0

            rows.append({
                "strategy": strategy.upper(),
                "k": k,
                "NDCG": round(ndcg, 4),
                f"P@{k}": round(prec, 4),
                f"Conf@{k}": round(conf, 4),
                "n_judged": len(rels),
            })

    return pd.DataFrame(rows)


def seed_dominance_from_rankings(
    profiles_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, profile in profiles_df.iterrows():
        liked = list(profile["liked_appids"])
        profile_id = profile["profile_id"]

        results = run_multi(liked_appids=liked, strategies=["max"], top_n=10)

        if "max" not in results:
            continue

        max_df = results["max"]
        seed_counts = max_df["dominant_seed"].value_counts()

        for seed_aid, count in seed_counts.items():
            rows.append({
                "profile_id": profile_id,
                "liked_appids": str(liked),
                "dominant_seed": int(seed_aid),
                "top10_count": int(count),
                "top10_share": round(count / 10, 2),
            })

        for _, r in max_df.iterrows():
            rows.append({
                "profile_id": profile_id,
                "candidate_appid": int(r["steam_appid"]),
                "candidate_name": r["name"],
                "dominant_seed": int(r["dominant_seed"]) if pd.notna(r.get("dominant_seed")) else None,
                "seed_similarity": round(r["seed_similarity"], 4),
                "final_score": round(r["final_score"], 4),
                "rank": int(r["rank"]),
            })

    return pd.DataFrame(rows)


def evaluate_synthetic_leave_one_out(
    profiles_df: pd.DataFrame,
    rec_boost: float = 0.03,
    strategies: list[str] | None = None,
    top_k: list[int] | None = None,
) -> pd.DataFrame:
    if strategies is None:
        strategies = ["max", "mean", "top2_mean"]
    if top_k is None:
        top_k = [10, 20, 100]

    rows = []
    for _, profile in profiles_df.iterrows():
        liked = list(profile["liked_appids"])

        results = run_multi(
            liked_appids=liked,
            strategies=strategies,
            top_n=max(top_k),
            rec_boost=rec_boost,
        )

        for strategy in strategies:
            ranked = results[strategy]

            for k in top_k:
                top = ranked.head(k)
                found = len(top)
                recall = found / len(liked) if len(liked) > 0 else 0
                rows.append({
                    "profile_id": profile["profile_id"],
                    "strategy": strategy.upper(),
                    "k": k,
                    "found": found,
                    "total_seeds": len(liked),
                    "recall": round(recall, 4),
                })

    return pd.DataFrame(rows)


def compute_pooled_ndcg(
    judgments_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    score_col: str = "relevance",
    k: int = 10,
) -> dict:
    joined = judgments_df.merge(mapping_df, on="pair_key", how="inner")
    all_rels = joined[score_col].dropna().tolist()
    all_rels = [float(r) for r in all_rels]

    ideal_all = sorted(all_rels, reverse=True)[:k]
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_all))

    results = {}
    for strategy in ["max", "mean", "top2_mean"]:
        score_key = f"{strategy}_score"
        strat = joined[joined[score_key].notna()].copy()
        if len(strat) == 0:
            results[strategy.upper()] = {"NDCG": 0.0, "P": 0.0, "Conf": 0.0}
            continue
        topk = strat.nlargest(k, score_key)
        rels = [float(r) for r in topk[score_col] if pd.notna(r)]
        dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        prec = (np.array(rels) >= 2).mean() if len(rels) > 0 else 0.0
        conf = np.mean(rels) if len(rels) > 0 else 0.0
        results[strategy.upper()] = {
            "NDCG": round(ndcg, 4),
            "P": round(prec, 4),
            "Conf": round(conf, 4),
            "n": len(rels),
        }

    return results


def main():
    import sys
    sys.path.insert(0, "src")
    from config import PROJECT_ROOT

    profiles_path = PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet"
    if not profiles_path.exists():
        print("Profiles not found. Run build_p1_profiles.py first.")
        sys.exit(1)

    profiles_df = pd.read_parquet(profiles_path)

    for split in ["dev", "val"]:
        subset = profiles_df[profiles_df["split"] == split]
        print(f"\n=== {split.upper()} ({len(subset)} profiles) ===")

        ev = evaluate_synthetic_leave_one_out(subset)
        for strategy in ["MAX", "MEAN", "TOP2_MEAN"]:
            strat = ev[ev["strategy"] == strategy]
            for k in [10, 20]:
                s = strat[strat["k"] == k]
                if len(s) > 0:
                    print(f"  {strategy:12s} Recall@{k:<3d} = {s['recall'].mean():.4f}")

        print(f"  Seed Dominance (MAX Top-10):")
        dom_df = seed_dominance_from_rankings(subset)
        if len(dom_df) > 0:
            seed_share = dom_df[dom_df["top10_share"].notna()]
            for _, r in seed_share.iterrows():
                print(f"    {r['profile_id']:20s} seed={r['dominant_seed']} count={int(r['top10_count'])} share={r['top10_share']}")


if __name__ == "__main__":
    main()
