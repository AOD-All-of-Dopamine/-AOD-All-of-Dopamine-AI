import sys
from pathlib import Path

import numpy as np
import pandas as pd


from src.personalization.seed_loader import SeedLoader
from src.personalization.candidate_retriever import CandidateRetriever
from src.personalization.score_aggregator import ScoreAggregator
from src.personalization.personalized_ranker import PersonalizedRanker


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


# ------------------------------------------------- 제품 경로의 집계 전략

def test_next_page_defaults_to_top2_mean():
    """`max` 는 시드 하나만 닮은 후보를 상위에 올린다 — 제품 경로는 top2_mean 이어야 한다.

    35프로필 미판정 0 비교에서 k=10 +0.057 [+0.020,+0.097], k=20 +0.077 [+0.050,+0.106]
    로 유의했다. 이 기본값이 바뀌면 그 근거가 무너지므로 고정한다.
    """
    import inspect

    from src.personalized_retrieve import next_page

    assert inspect.signature(next_page).parameters["strategy"].default == "top2_mean"


def test_top2_mean_equals_max_for_single_seed():
    """시드가 1개면 두 전략은 수학적으로 같다 — 단일 시드 프로필은 이 레버로 못 고친다.

    `niche_soulslike_solo`(Salt and Sanctuary 하나)와 `niche_puzzle_solo`(Opus Magnum 하나)가
    top2_mean 으로 전혀 안 움직인 이유가 이것이다. 그 둘은 다른 레버가 필요하다.
    """
    import numpy as np

    from src.personalization.score_aggregator import ScoreAggregator

    agg = ScoreAggregator()
    sim = np.array([[0.9, 0.4, 0.7]])                      # 시드 1 x 후보 3
    corpus = pd.DataFrame({"steam_appid": [1, 2, 3], "name": ["a", "b", "c"]})
    out = agg.aggregate_all(sim, {283640: None}, corpus, strategies=["max", "top2_mean"])
    assert np.allclose(out["max"]["seed_similarity"].values,
                       out["top2_mean"]["seed_similarity"].values)


def test_every_strategy_carries_dominant_seed():
    """집계 전략을 바꿔도 시드 인터리빙이 죽으면 안 된다.

    예전에는 `dominant_seed` 가 max 에만 있어서, 제품 기본값을 top2_mean 으로 바꾸자
    인터리빙이 조용히 무력화됐다(약한 시드가 굶는다). 점수와 소속 시드는 별개다.
    """
    import numpy as np

    from src.personalization.score_aggregator import ScoreAggregator

    sim = np.array([[0.9, 0.1], [0.2, 0.8]])          # 시드 2 x 후보 2
    corpus = pd.DataFrame({"steam_appid": [10, 20], "name": ["a", "b"]})
    out = ScoreAggregator().aggregate_all(
        sim, {111: None, 222: None}, corpus, strategies=["max", "mean", "top2_mean"])
    for strat, df in out.items():
        assert df["dominant_seed"].notna().all(), strat
        assert list(df["dominant_seed"]) == [111, 222], strat
