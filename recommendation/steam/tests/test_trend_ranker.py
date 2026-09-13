import sys
from pathlib import Path

import numpy as np
import pandas as pd


from src.personalization.seed_loader import SeedLoader
from src.personalization.candidate_retriever import CandidateRetriever
from src.personalization.score_aggregator import ScoreAggregator
from src.personalization.personalized_ranker import PersonalizedRanker
from src.trend.trend_ranker import TrendRanker


def test_trend_ranker_extends_personalized():
    ranker = TrendRanker()
    assert isinstance(ranker, PersonalizedRanker)


def test_trend_ranker_default_weight():
    ranker = TrendRanker()
    assert ranker.trend_weight == 0.01
    assert ranker.rec_boost == 0.03


def test_trend_ranker_variable_weight():
    for w in [0.005, 0.01, 0.02]:
        ranker = TrendRanker(trend_weight=w)
        assert ranker.trend_weight == w


def test_trend_rank_basic_pipeline():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()
    ranker = TrendRanker(trend_weight=0.01)

    liked = [730, 578080, 359550]
    seed_embs = loader.load(liked)
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()
    results = aggregator.aggregate_all(sim, seed_embs, corpus, strategies=["max"])
    ranked = ranker.rank(results["max"], exclude_appids=set(liked), top_n=50)

    assert len(ranked) == 50
    assert "final_score" in ranked.columns
    assert "trend_signal" in ranked.columns
    assert "recommendations_percentile" in ranked.columns


def test_trend_rank_excludes_seeds():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()
    ranker = TrendRanker()

    liked = [730, 578080, 359550]
    seed_embs = loader.load(liked)
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()
    results = aggregator.aggregate_all(sim, seed_embs, corpus, strategies=["max"])
    ranked = ranker.rank(results["max"], exclude_appids=set(liked), top_n=300)

    ranked_appids = set(ranked["steam_appid"])
    for aid in liked:
        assert aid not in ranked_appids, f"Seed {aid} should be excluded"


def test_trend_equal_personalized_when_weight_zero():
    loader = SeedLoader()
    retriever = CandidateRetriever()
    aggregator = ScoreAggregator()

    liked = [730, 578080, 359550]
    seed_embs = loader.load(liked)
    sim = retriever.compute_similarity_matrix(seed_embs)
    corpus = retriever.full_corpus_frame()
    results = aggregator.aggregate_all(sim, seed_embs, corpus, strategies=["max"])

    base_ranker = PersonalizedRanker(rec_boost=0.03)
    trend_ranker = TrendRanker(rec_boost=0.03, trend_weight=0.0)

    base_result = base_ranker.rank(results["max"], exclude_appids=set(liked), top_n=300)
    trend_result = trend_ranker.rank(results["max"], exclude_appids=set(liked), top_n=300)

    # With trend_weight=0, final_score should be identical
    np.testing.assert_array_almost_equal(
        base_result["final_score"].values,
        trend_result["final_score"].values,
    )
