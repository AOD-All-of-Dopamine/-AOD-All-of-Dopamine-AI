import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personalization.seed_loader import SeedLoader
from personalization.candidate_retriever import CandidateRetriever
from personalization.score_aggregator import ScoreAggregator
from personalization.personalized_ranker import PersonalizedRanker


def test_seed_loader_valid():
    loader = SeedLoader()
    embs = loader.load([730])
    assert 730 in embs
    assert embs[730].shape[0] > 0


def test_seed_loader_missing():
    loader = SeedLoader()
    try:
        loader.load([99999999])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_similarity_matrix_shape():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    seed_embs = loader.load([730, 578080, 359550])
    sim = retriever.compute_similarity_matrix(seed_embs)
    assert sim.shape == (3, 19476), f"Expected (3, 19476), got {sim.shape}"


def test_full_corpus_frame():
    retriever = CandidateRetriever()
    df = retriever.full_corpus_frame()
    assert len(df) == 19476
    assert "steam_appid" in df.columns
    assert "name" in df.columns


def test_aggregator_shapes():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()

    seed_embs = loader.load([730, 578080, 359550])
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()

    results = aggregator.aggregate_all(sim, seed_embs, corpus)
    for strategy in ["max", "mean", "top2_mean"]:
        assert strategy in results
        assert len(results[strategy]) == 19476
        assert "seed_similarity" in results[strategy].columns
        assert "strategy" in results[strategy].columns


def test_max_dominant_seed():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()

    seed_embs = loader.load([730, 578080, 359550])
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()

    results = aggregator.aggregate_all(sim, seed_embs, corpus, strategies=["max"])
    max_df = results["max"]
    assert max_df["dominant_seed"].notna().all()
    assert set(max_df["dominant_seed"].unique()) == {730, 578080, 359550}


def test_hidden_candidate_under_old_approach():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()

    seed_embs = loader.load([730, 578080, 489830])
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()

    top100_sets = []
    for i in range(sim.shape[0]):
        top100_idx = np.argsort(sim[i])[::-1][:100]
        top100_sets.append(set(top100_idx))

    union_top100 = set.union(*top100_sets)

    mean_sim = sim.mean(axis=0)
    mean_top50 = np.argsort(mean_sim)[::-1][:50]

    hidden = [idx for idx in mean_top50 if idx not in union_top100]
    assert len(hidden) > 0, (
        "Expected at least one MEAN Top-50 candidate outside all per-seed Top-100s"
    )


def test_ranker_excludes_seeds():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()
    ranker = PersonalizedRanker()

    liked = [730, 578080, 359550]
    seed_embs = loader.load(liked)
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()
    results = aggregator.aggregate_all(sim, seed_embs, corpus, strategies=["max"])
    ranked = ranker.rank(results["max"], exclude_appids=set(liked), top_n=300)

    ranked_appids = set(ranked["steam_appid"])
    for aid in liked:
        assert aid not in ranked_appids, f"Seed {aid} should be excluded"


def test_mean_top2_mean_equivalence_with_2_seeds():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()

    seed_embs = loader.load([730, 578080])
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()

    results = aggregator.aggregate_all(sim, seed_embs, corpus, strategies=["mean", "top2_mean"])
    mean_scores = results["mean"]["seed_similarity"].values
    top2_scores = results["top2_mean"]["seed_similarity"].values
    np.testing.assert_array_almost_equal(mean_scores, top2_scores)


def test_top2_mean_different_from_max_with_3_seeds():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()

    seed_embs = loader.load([730, 578080, 489830])
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()

    results = aggregator.aggregate_all(sim, seed_embs, corpus)
    max_scores = results["max"]["seed_similarity"].values
    top2_scores = results["top2_mean"]["seed_similarity"].values

    assert not np.array_equal(max_scores, top2_scores), (
        "MAX and TOP2_MEAN should differ for mixed profiles"
    )
