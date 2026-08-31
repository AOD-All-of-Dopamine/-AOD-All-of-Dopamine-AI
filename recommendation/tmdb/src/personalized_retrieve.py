"""TMDB 개인화 추천 — 조립.

    시드 로드 → 유사도 행렬 → 집계 → 랭킹(인기도·평점 보정) → 후처리(시리즈·인터리빙)

무거운 생성자(임베딩 233MB)는 `build_components()` 로 한 번만 만들고 재사용한다.
52프로필 평가는 이걸 52번 호출하므로 재사용이 필수다.
"""
import numpy as np, pandas as pd
from src.personalization.seed_loader import SeedLoader
from src.personalization.candidate_retriever import CandidateRetriever
from src.personalization.score_aggregator import ScoreAggregator
from src.personalization.personalized_ranker import PersonalizedRanker
from src.config import PRODUCTION_POSTPROCESS


def build_components(artifacts=None, hub_lambda: float | None = None,
                     vote_boost: float = 0.0, rating_boost: float = 0.0,
                     align_w: float = 0.0, min_overview_len: int = 0,
                     media_w: float = 0.0, genre_w: float = 0.0, vote_w: float = 0.0,
                     kw_w: float = 0.0):
    ret = CandidateRetriever(artifacts, min_overview_len=min_overview_len)
    if hub_lambda is not None: ret.hub_lambda = hub_lambda
    return (SeedLoader(artifacts), ret, ScoreAggregator(),
            PersonalizedRanker(artifacts, vote_boost=vote_boost,
                               rating_boost=rating_boost, align_w=align_w,
                               media_w=media_w, genre_w=genre_w, vote_w=vote_w, kw_w=kw_w))


def recommend(seed_rows, components=None, strategy: str = "top2_mean", top_n: int = 50,
              postprocess_on: bool = True, postprocess_kwargs: dict | None = None,
              exclude_rows=None, **comp_kwargs) -> pd.DataFrame:
    loader, retriever, agg, ranker = components or build_components(**comp_kwargs)
    embs = loader.load(list(seed_rows))
    sim = retriever.compute_similarity_matrix(embs)
    scored = agg.aggregate_all(sim, embs, retriever.full_corpus_frame(),
                               strategies=[strategy])[strategy]
    excl = set(int(r) for r in seed_rows) | set(int(r) for r in (exclude_rows or ()))
    # 후처리가 위에서부터 걸러내므로 넉넉히 뽑아 둔다
    rank_n = top_n * 8 if postprocess_on else top_n
    # 시드 인기 백분위 중앙 — 정합 항의 목표값 (D-31)
    seed_pct = float(np.median(ranker.vote_pct[list(int(r) for r in seed_rows)]))
    seed_medias = {ranker.dataset.iloc[int(r)]["media"] for r in seed_rows}
    seed_genres = None
    if ranker.genre_w:
        seed_genres = []
        for rr in seed_rows:
            g = ranker.dataset.iloc[int(rr)]["genres"]
            seed_genres.append(frozenset(g.tolist() if hasattr(g, "tolist") else (g or [])))
    seed_kws = None
    if ranker.kw_w:
        seed_kws = []
        for rr in seed_rows:
            kk = ranker.dataset.iloc[int(rr)]["keywords"]
            seed_kws.append(frozenset(kk.tolist() if hasattr(kk, "tolist") else (kk or [])))
    ranked = ranker.rank(scored, exclude_rows=excl, top_n=rank_n, seed_kws=seed_kws,
                         servable_mask=retriever.servable, seed_pct=seed_pct,
                         seed_medias=seed_medias, seed_genres=seed_genres)
    if postprocess_on:
        from src.postprocess import postprocess as pp
        ranked = pp(ranked, ranker.dataset, top_n=top_n, seed_rows=seed_rows,
                    **(postprocess_kwargs if postprocess_kwargs is not None
                       else PRODUCTION_POSTPROCESS))
    return ranked
