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
                     # **제품 결정 (2026-09-12)**: 원어 줄거리도 서빙한다.
                     # 켜 두면 한국어 번역이 없는 26,051건이 통째로 빠지는데, 그 비용이
                     # 취향마다 다르다 — lowvote_romance 60% · longtail_family 42% ·
                     # coh_marvel 0%. 사실상 "유명작만 서빙" 필터였다.
                     # 빠진 작품들은 줄거리가 **있고** 임베딩도 돼 있다. 없는 건 한글 표시뿐이다.
                     require_korean: bool = False,
                     media_w: float = 0.0, genre_w: float = 0.0, vote_w: float = 0.0,
                     kw_w: float = 0.0, director_w: float = 0.0):
    ret = CandidateRetriever(artifacts, min_overview_len=min_overview_len,
                             require_korean=require_korean)
    if hub_lambda is not None: ret.hub_lambda = hub_lambda
    return (SeedLoader(artifacts), ret, ScoreAggregator(),
            PersonalizedRanker(artifacts, vote_boost=vote_boost,
                               rating_boost=rating_boost, align_w=align_w,
                               media_w=media_w, genre_w=genre_w, vote_w=vote_w, kw_w=kw_w,
                               director_w=director_w))


POOL_FLOOR = 400   # = 평가 경로의 풀 깊이(k=50 × 8)


def recommend(seed_rows, components=None, strategy: str = "top2_mean", top_n: int = 50,
              postprocess_on: bool = True, postprocess_kwargs: dict | None = None,
              exclude_rows=None, media: str | None = None, **comp_kwargs) -> pd.DataFrame:
    """`media` — "movie" | "tv" 면 그 매체만 후보로 남긴다(영화 탭 / 드라마 탭). `None` 은 혼합(현행).

    D-42 가 기각한 하드 필터는 **혼합 목록 안에서** 매체를 자르는 것이었다. 여기는 제품이
    탭을 나눈 뒤의 후보 범위라 다른 문제다. 시드는 매체와 무관하게 그대로 쓴다 —
    영화 시드로 드라마 탭을 채울 수 있어야 한다(시드 40/52 가 영화뿐).
    후보가 전부 한 매체면 `media_w` 페널티는 모든 후보에 같게 걸려 순서를 바꾸지 않는다.
    """
    if media not in (None, "movie", "tv"):
        raise ValueError(f"media 는 None | 'movie' | 'tv' — {media!r}")
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
    # 시드 메타는 **열에서 위치로** 꺼낸다. 예전에는 시드마다 `dataset.iloc[r]` 로 15열짜리
    # 혼합 dtype 행을 통째로 조립했다 — 시드 50개면 요청마다 100번이다 (2026-09-19 서빙 지연).
    # `_media`·`column()` 은 같은 열을 numpy 로 들고 있는 것이라 원소가 같은 객체다.
    seed_medias = {ranker._media[int(r)] for r in seed_rows}
    seed_genres = None
    if ranker.genre_w:
        gcol = ranker.column("genres")
        seed_genres = []
        for rr in seed_rows:
            g = gcol[int(rr)]
            seed_genres.append(frozenset(g.tolist() if hasattr(g, "tolist") else (g or [])))
    seed_kws = None
    if ranker.kw_w:
        kcol = ranker.column("keywords")
        seed_kws = []
        for rr in seed_rows:
            kk = kcol[int(rr)]
            seed_kws.append(frozenset(kk.tolist() if hasattr(kk, "tolist") else (kk or [])))
    seed_dirs = None
    if ranker.director_w:
        # 감독이 없는 시드(드라마 created_by 결측 등)는 빈 집합 → 그 시드 기여는 0 이다.
        seed_dirs = [ranker._dirs[int(rr)] for rr in seed_rows]
    servable = retriever.servable
    if media is not None:   # 풀 자르기(head) 전에 거르므로 탭마다 풀 깊이 rank_n 이 유지된다
        # `media_mask` 는 같은 `==` 비교를 매체값마다 한 번만 한다. `&` 가 새 배열을 만들어
        # 캐시는 그대로다(요청이 `servable` 을 고치지 않는다).
        servable = servable & ranker.media_mask(media)
    ranked = ranker.rank(scored, exclude_rows=excl, top_n=rank_n, seed_kws=seed_kws,
                         seed_dirs=seed_dirs,
                         servable_mask=servable, seed_pct=seed_pct,
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
              media: str | None = None, **comp_kwargs):
    """TMDB 새로고침 한 페이지. `seen_rows` 아래를 잇는다.

    반환은 `recommend` 와 같은 프레임이고 `row`(코퍼스 행)가 아이템 키다.
    `media` — "movie" | "tv" 탭 분리 (`recommend` 참고). 탭마다 `seen_rows` 를 따로 누적한다.

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
        media=media,
        **comp_kwargs,
    )
