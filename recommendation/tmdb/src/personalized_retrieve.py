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
from src.config import PRODUCTION, PRODUCTION_POSTPROCESS


def build_components(artifacts=None, hub_lambda: float | None = None,
                     vote_boost: float = 0.0, rating_boost: float = 0.0,
                     align_w: float = 0.0, min_overview_len: int = 0,
                     require_korean: bool = True,
                     media_w: float = 0.0, genre_w: float = 0.0, vote_w: float = 0.0,
                     kw_w: float = 0.0):
    ret = CandidateRetriever(artifacts, min_overview_len=min_overview_len,
                             require_korean=require_korean)
    if hub_lambda is not None: ret.hub_lambda = hub_lambda
    return (SeedLoader(artifacts), ret, ScoreAggregator(),
            PersonalizedRanker(artifacts, vote_boost=vote_boost,
                               rating_boost=rating_boost, align_w=align_w,
                               media_w=media_w, genre_w=genre_w, vote_w=vote_w, kw_w=kw_w))


POOL_FLOOR = 400   # = 평가 경로의 풀 깊이(k=50 × 8)


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
    # 풀 깊이를 k 에 묶지 않는다 (2026-09-04 2차 검수). TMDB 실측: 서빙 top-20 vs 평가
    # top-20 이 15프로필 중 3개에서 달랐다(최소 겹침 80%). 하한 = 평가 깊이(50×8=400).
    rank_n = max(POOL_FLOOR, top_n * 8) if postprocess_on else top_n
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


# ── 제품 경로 ────────────────────────────────────────────────────────────────
# 2026-09-04 2차 검수: Steam·웹소설·웹툰에는 `next_page` 가 있는데 **TMDB 에만 없었다.**
# 제품이 페이지네이션(새로고침 / 더 보기)을 요구하면 TMDB 만 못 하는 상태였다.
#
# 계약은 세 플랫폼과 **똑같이** 맞춘다:
#   · 새로고침 한 번 = 이 함수 한 번
#   · 호출자가 반환된 키를 `seen_rows` 에 **누적**해서 다음 호출에 넘긴다
#   · 이미 본 것은 제외 집합으로 빠지고, 그 아래가 이어진다
#
# **Steam·웹소설과 달리 여기서는 확정값 말고 아무것도 바꾸지 않는다.**
# 두 플랫폼의 `next_page` 는 한때 제품 전용 기본값(rec_boost 0.15 · 리뷰 하한 300 등)을
# 들고 있었고 그게 평가와 갈라져 D-55/D-66/X-20 으로 되돌려졌다. 같은 실수를 새로 만들지
# 않는다 — 깊은 페이지에 별도 설정이 필요하다는 근거가 나오면 **그때 사전등록으로** 정한다.
def next_page(seed_rows, seen_rows=None, page_size: int = 10, components=None,
              strategy: str | None = None, postprocess_kwargs: dict | None = None,
              **comp_kwargs):
    """TMDB 새로고침 한 페이지. `seen_rows` 아래를 잇는다.

    반환은 `recommend` 와 같은 프레임이고 `row`(코퍼스 행)가 아이템 키다.

    시드 0개는 계약 밖이다 — 개인화할 근거가 없다. 콜드스타트는 호출자가 별도 경로로
    처리해야 한다(Steam 이 같은 이유로 `ValueError` 를 던진다).
    """
    if not len(list(seed_rows)):
        raise ValueError("seed_rows 가 비어 있다 — 콜드스타트는 next_page 의 계약 밖이다")
    seen = {int(x) for x in (seen_rows or ())}
    return recommend(
        list(seed_rows),
        components=components,
        strategy=strategy or PRODUCTION["strategy"],
        top_n=page_size,
        postprocess_on=True,
        postprocess_kwargs=postprocess_kwargs,
        exclude_rows=seen,
        **comp_kwargs,
    )
