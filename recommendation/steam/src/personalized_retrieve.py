import argparse
from pathlib import Path

import numpy as np
import yaml

from config import PROJECT_ROOT, ensure_artifacts_dir
from personalization.seed_loader import SeedLoader
from personalization.candidate_retriever import CandidateRetriever
from personalization.score_aggregator import ScoreAggregator
from personalization.personalized_ranker import PersonalizedRanker


def run_multi(
    liked_appids: list[int],
    strategies: list[str] | None = None,
    top_n: int = 300,
    rec_boost: float = 0.03,
    output_dir: str | None = None,
) -> dict[str, dict]:
    if strategies is None:
        strategies = ["max", "mean", "top2_mean"]

    loader = SeedLoader()
    seed_embs = loader.load(liked_appids)

    retriever = CandidateRetriever()
    sim_matrix = retriever.compute_similarity_matrix(seed_embs)
    corpus_df = retriever.full_corpus_frame()

    aggregator = ScoreAggregator()
    aggregated = aggregator.aggregate_all(sim_matrix, seed_embs, corpus_df, strategies=strategies)

    ranker = PersonalizedRanker(rec_boost=rec_boost)

    results = {}
    for strategy in strategies:
        ranked = ranker.rank(
            aggregated[strategy],
            exclude_appids=set(liked_appids),
            top_n=top_n,
        )
        results[strategy] = ranked
        if output_dir:
            out_path = Path(output_dir) / f"ranked_{strategy}.parquet"
            ranked.to_parquet(out_path, index=False)
            print(f"  Saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="P1 Multi-Seed Personalized Retrieval (Full Corpus)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--liked-appids", type=int, nargs="+", help="Seed game appids")
    parser.add_argument("--strategies", type=str, nargs="+", default=["max", "mean", "top2_mean"])
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--rec-boost", type=float, default=0.03)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        liked = cfg.get("liked_appids", args.liked_appids)
        strategies = cfg.get("aggregation", {}).get("strategies", args.strategies)
        top_n = cfg.get("aggregation", {}).get("top_n", args.top_n)
        rec_boost = cfg.get("aggregation", {}).get("rec_boost", args.rec_boost)
        output_dir = cfg.get("paths", {}).get("ranked_dir", args.output_dir)
    else:
        liked = args.liked_appids
        strategies = args.strategies
        top_n = args.top_n
        rec_boost = args.rec_boost
        output_dir = args.output_dir

    if not liked:
        parser.error("--liked-appids is required (or set liked_appids in config)")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_dir = str(ensure_artifacts_dir())

    results = run_multi(
        liked_appids=liked,
        strategies=strategies,
        top_n=top_n,
        rec_boost=rec_boost,
        output_dir=output_dir,
    )

    for strategy, df in results.items():
        print(f"\n=== {strategy.upper()} ===")
        print(f"  Candidates: {len(df)}")
        if len(df) > 0:
            print(f"  Top-5:")
            for _, r in df.head(5).iterrows():
                print(f"    #{r['rank']} {r['name']}  (score={r['final_score']:.4f})")


if __name__ == "__main__":
    main()
