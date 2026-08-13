import argparse
from pathlib import Path

import numpy as np
import yaml

from src.config import PROJECT_ROOT, ensure_artifacts_dir
from src.personalization.seed_loader import SeedLoader
from src.personalization.candidate_retriever import CandidateRetriever
from src.personalization.score_aggregator import ScoreAggregator
from src.personalization.personalized_ranker import PersonalizedRanker


def build_components(rec_boost: float = 0.03, artifacts=None, trend_weight: float = 0.0):
    """코퍼스 임베딩(76MB)과 dataset 을 읽는 무거운 생성자들을 한 번만 만든다.

    LOO 평가처럼 수십 번 호출하는 경우 `run_multi(..., components=...)` 로 재사용한다.
    """
    return (
        SeedLoader(artifacts),
        CandidateRetriever(artifacts),
        ScoreAggregator(),
        PersonalizedRanker(rec_boost=rec_boost, artifacts=artifacts, trend_weight=trend_weight),
    )


def run_multi(
    liked_appids: list[int],
    strategies: list[str] | None = None,
    top_n: int = 300,
    rec_boost: float = 0.03,
    output_dir: str | None = None,
    components: tuple | None = None,
    postprocess: bool = False,
    postprocess_kwargs: dict | None = None,
    exclude_appids: set[int] | list[int] | None = None,
) -> dict[str, dict]:
    """`postprocess=True` 면 랭킹 뒤에 다양성 후처리(시드 인터리빙·시리즈 상한·hard filter)를 건다.

    후처리는 상위를 걸러내므로 랭커에서 넉넉히(top_n × 5) 뽑은 뒤 잘라야 한다.

    `exclude_appids` — **새로고침 제품의 필수 입력.** 이게 없으면 이 함수는 순수 함수라
    같은 입력에 같은 목록을 낸다(= 새로고침해도 화면이 안 바뀐다). 여기에 넣을 것:
      · 이미 보여준 것    (`aod_ai.rec_impression`)
      · 이미 아는 것      (LIKE / DISLIKE / bookmark / 리뷰 작성한 콘텐츠)
    시드(`liked_appids`)는 자동으로 합쳐지므로 따로 넣지 않아도 된다.
    """
    if strategies is None:
        strategies = ["max", "mean", "top2_mean"]

    loader, retriever, aggregator, ranker = components or build_components(rec_boost)

    seed_embs = loader.load(liked_appids)
    sim_matrix = retriever.compute_similarity_matrix(seed_embs)
    corpus_df = retriever.full_corpus_frame()
    aggregated = aggregator.aggregate_all(sim_matrix, seed_embs, corpus_df, strategies=strategies)

    excluded = set(liked_appids) | set(exclude_appids or ())
    rank_n = top_n * 5 if postprocess else top_n
    results = {}
    for strategy in strategies:
        ranked = ranker.rank(
            aggregated[strategy],
            exclude_appids=excluded,
            top_n=rank_n,
        )
        if postprocess:
            from src.postprocess import postprocess as apply_postprocess

            ranked = apply_postprocess(
                ranked, ranker.dataset.reset_index(), top_n=top_n,
                **(postprocess_kwargs or {}),
            )
        results[strategy] = ranked
        if output_dir:
            out_path = Path(output_dir) / f"ranked_{strategy}.parquet"
            ranked.to_parquet(out_path, index=False)
            print(f"  Saved: {out_path}")

    return results


# 새로고침 3페이지까지 품질이 유지되도록 맞춘 값. 근거는 next_page docstring 참고.
REFRESH_REC_BOOST = 0.15

# 전체 코퍼스(173,691)로 넓히면서 필요해진 품질 하한. 근거는 apply_hard_filters docstring.
# 2026-08-12: 300 → 0. 하한은 얕은 페이지에서만 도움이 됐고 깊이에서는 오히려 해로웠다.
# 35프로필 · 미판정 0 · 페어드 부트스트랩 (top2_mean + 단일시드 필터 기준):
#
#   깊이    하한 300 (풀 12,852)   하한 없음 (풀 173,691)   Δ (95%CI)
#   k=10   0.9429                0.9343                 -0.009 [-0.034,+0.014]
#   k=20   0.9214                0.9257                 +0.004 [-0.013,+0.021]
#   k=30   0.9181                0.9162                 -0.002 [-0.016,+0.013]
#   k=50   0.9000 (미달 3)        0.9194 (미달 2)         +0.019 [+0.001,+0.039]  ← 유의
#
# 축별로 방향이 갈린다: 저리뷰 +0.080, 니치 +0.043, 롱테일 +0.010, 혼합 0.000, 대작 -0.002.
# 무명 게임을 좋아하는 사람의 이웃은 대부분 무명이라 좁은 풀 **밖**에 있고, 대작을 좋아하는
# 사람의 이웃은 이미 안에 다 있다. 하한은 후자에게 무해하고 전자에게만 손해였다.
#
# 이것이 전체 코퍼스 임베딩(173,691)이 값을 하는 지점이다. 얕은 페이지만 보면 차이가 없어
# "코퍼스 확대는 무의미"로 보였는데, k=50 에서 드러났다.
REFRESH_MIN_REVIEWS = 0

#: 합의 태그를 몇 개 맞혔는지로 재정렬하는 가중치. 0 이면 끔. 근거는 consensus_overlap_boost.
REFRESH_CONSENSUS_BOOST = 0.0

#: 시드 인기도에 비례하는 리뷰 하한 계수. 0 이면 끔.
#
# **왜 필요한가.** 꼬리의 질이 장르마다 다르다. 덱빌딩·메트로배니아의 무명작은 인디
# 열정작이라 판정 3점을 받지만(longtail_deckbuilder 는 리뷰 미보고 게임들로 P@100 1.000),
# FPS·ARPG 의 무명작은 에셋 플립이다. 공급 부족이 아니다 — 실측으로 coh_fps 취향에는
# 리뷰 1천+ 슈터가 551개, coh_arpg 취향에는 501개 있는데 100칸을 못 채우는 것이 아니라
# 랭커가 그것들 대신 양산품을 올린다.
#
# **왜 고정 임계값이 아닌가.** 인기시드(10만+) 조건부 하한 300 을 재봤더니 coh_arpg 는
# 0.73 → 0.83 으로 살았지만 coh_grand_strategy 가 0.84 → 0.76 으로 무너졌다. 무명이지만
# 잘 만든 니치 대전략이 잘리고 그 자리를 인기 있지만 장르가 다른 것(RimWorld·Steel Crew)이
# 채웠다. 계단식 임계값은 "대작 취향"과 "니치 장르 취향"을 같이 취급한다.
#
# 비례 규칙은 그 둘을 자동으로 갈라준다:
#   coh_fps 176만 → 하한 1,762   coh_arpg 27만 → 277   coh_grand_strategy 13만 → 127
#   niche/lowrev/longtail        → 100 미만이라 하한 없음 (축별 Δ 정확히 0.000)
#
# 35프로필 · 미판정 0 · 페어드 부트스트랩 (하한 없음 대비):
#   k=50   0.9331 (미달 2) → 0.9383 (미달 0)   Δ +0.005 [-0.005,+0.017]
#   k=100  0.9100 (미달 2) → 0.9180 (미달 0)   Δ +0.008 [+0.001,+0.016]  ← 유의
#   k=100 축별: 저리뷰 0.000 · 롱테일 0.000 · 니치 0.000 · 혼합 -0.002 · 대작 +0.030
#
# 계수는 {0.001, 0.0005} 두 개만 재고 0.001 을 골랐다. 0.0005 는 k=100 에서 미달 1개가
# 남았다. 값을 옮기려면 두 값을 다시 재야 한다.
REFRESH_SEED_SCALED_FLOOR = 0.001

#: 멀티 전용 + 플레이어 기반 미측정 후보 제거. 근거는 postprocess.drop_dead_multiplayer.
#
# 35프로필 · 미판정 0 · 페어드 부트스트랩:
#   k=50   0.9234 (미달 1) → 0.9223 (미달 1)   Δ -0.001 [-0.004,+0.001]
#   k=75   0.8990 (미달 2) → 0.9013 (미달 1)   Δ +0.002 [+0.000,+0.006]
#
# 축별 k=75: 저리뷰 0.000 · 롱테일 0.000 · 니치 0.000 · 혼합 +0.001 · 대작 +0.007.
# 35개 중 3개만 바뀌고 셋 다 올랐다: coh_classic_multi +0.053 · coh_fps +0.013 ·
# mix2_party_narrative +0.013. 평균 Δ 는 유의하지 않다(구조적으로 소수만 대상).
# 채택 근거는 평균이 아니라 **부작용 0 + 대상 개선 + 미달 감소**다.
REFRESH_DROP_DEAD_MP = True


def next_page(
    liked_appids: list[int],
    seen_appids: set[int] | list[int] | None = None,
    page_size: int = 10,
    strategy: str = "top2_mean",
    rec_boost: float = REFRESH_REC_BOOST,
    components: tuple | None = None,
    postprocess: bool = True,
    require_known_reviews: bool = False,
    min_reviews: int = REFRESH_MIN_REVIEWS,
    consensus_boost: float = REFRESH_CONSENSUS_BOOST,
    drop_dead_mp: bool = REFRESH_DROP_DEAD_MP,
    seed_scaled_floor: float = REFRESH_SEED_SCALED_FLOOR,
):
    """새로고침 한 번 = 이 함수 한 번. 서빙이 쓸 계약을 코드로 고정한다.

    호출자는 반환된 `steam_appid` 를 `seen_appids` 에 누적해서 다음 호출에 넘겨야 한다.
    그 누적을 어디에 저장할지가 서빙의 과제다(`aod_ai.rec_impression`).

    **기본값이 run_multi 와 다른 이유** — 깊은 페이지가 무너지는 것을 막기 위해서다.
    원인은 관련성 붕괴가 아니라 품질 붕괴였다: 유사도는 1→5페이지에서 0.75→0.71 로 거의
    안 변하는데 리뷰 수 중앙값이 2,831 → 434 로 무너진다. 유사도가 평평한 구간에서는
    미세한 유사도 차이보다 인기도가 훨씬 나은 정렬 기준이다.

      · `require_known_reviews=True` — Steam 이 리뷰 수를 보고하는 게임만(코퍼스의 39%).
        결측 = "리뷰가 거의 없음"이라는 이진 신호다.
      · `rec_boost=0.15` — 기본 0.03 은 유사도 스프레드보다 작아 정렬을 거의 못 바꾼다.
      · `strategy="top2_mean"` — **`max` 는 시드 3개 중 하나만 닮아도 상위에 올린다.**
        `coh_grand_strategy` 가 그 피해자였다: Civ V 만 닮은 캐주얼 전략이 HOI4·EU4 를
        무시하고 1페이지에 올라와 P@10 0.60 이었다. `top2_mean` 은 상위 2개 시드의 평균이라
        최소 2개와 맞아야 올라온다. 시드가 1개면 `max` 와 수학적으로 같다.

        35프로필 · 미판정 0 · 페어드 부트스트랩:

          깊이    max                 top2_mean            Δ (95%CI)
          k=10   0.8829 (미달 4)      0.9400 (미달 1)      +0.057 [+0.020,+0.097]
          k=20   0.8700 (미달 3)      0.9471 (미달 1)      +0.077 [+0.050,+0.106]

        깊어질수록 격차가 커진다 — `max` 는 깊은 페이지에서 "시드 하나만 닮은" 후보를
        더 많이 끌어오기 때문이다.

    판정 결과 (4개 프로필 × 페이지당 10개, Claude 판정, 각 페이지 40쌍 전수):

      페이지   현재(0.03,하한X)      개선(0.15,하한O)
        1      평균 1.79 / P@10 .64   평균 1.85 / P@10 .68
        2      평균 1.44 / P@10 .49   평균 1.53 / P@10 .53
        3      평균 1.06 / P@10 .32   평균 1.63 / P@10 .55

    현재 설정은 3페이지에서 41% 무너지지만(1.79→1.06) 개선 설정은 12% 하락에 그치고
    2→3페이지에서는 오히려 오른다. `run_multi` 기본값은 기존 실험 재현성 때문에 건드리지
    않는다 — 제품 경로인 이 함수에서만 바꾼다.

    `min_reviews=300` 은 코퍼스를 19,476 → 173,691 로 넓히면서 추가됐다. **코퍼스만 키우면
    오히려 나빠진다**는 것이 8개 프로필 × Top-10 전수 판정(미판정 0칸)으로 확인됐다:

      설정                    P@10    NDCG@10   0점/프로필   리뷰중앙값
      구 코퍼스 19,476        0.625    0.544      1.12        7,742
      신 코퍼스 173,691       0.562    0.566      1.25        2,210   ← 하락
      신 코퍼스 + 하한 300    0.713    0.631      0.88        6,572

    코퍼스가 9배가 되면 이웃도 9배가 되지만 늘어난 것의 대부분은 리뷰 수백 개짜리다.
    유사도가 평평한 구간에서 그것들이 유명작을 밀어낸다. 하한은 그 구간만 잘라낸다.
    """
    if not liked_appids:
        # 시드 0개는 이 함수의 계약 밖이다(개인화할 근거가 없다). 가드 없이는 numpy
        # 브로드캐스트 에러로 죽어 서빙이 원인을 알 수 없다. 콜드스타트는 호출자가
        # 인기순 등 별도 경로로 처리해야 한다.
        raise ValueError("liked_appids 가 비어 있다 — 콜드스타트는 next_page 의 계약 밖이다")

    seen = set(seen_appids or ())

    # 시드가 1개면 top2_mean 이 max 와 같아져 집계 레버가 안 듣는다 — 그 구멍만 태그로 메운다.
    # 다시드에는 절대 걸지 않는다(희귀 태그 합집합이 "공통점"이 아니라 "한 시드의 특이점"이 된다).
    from src.postprocess import (POPULAR_SEED_REVIEWS, consensus_seed_tags, load_dataset,
                                 rare_seed_tags)

    # **반드시 랭킹에 쓰는 데이터셋을 그대로 쓴다.** 예전에는 `load_dataset()` 를 인자 없이
    # 불러서 기본 아티팩트(s1_v2)를 읽었다 — components 가 tags_full 이어도 태그 필터만
    # 다른 코퍼스에서 계산됐다. max_df 는 코퍼스 크기에 정의되므로(21,883 vs 173,691)
    # "희귀"·"합의"의 기준 자체가 달라졌고, 작은 쪽에 없는 시드에서는 그냥 터졌다.
    ds = (
        components[3].dataset.reset_index()
        if components is not None
        else load_dataset()
    )

    # reindex 로 뽑는다: `.get()` 은 없는 시드에 None 을 줘서 nanmedian 이 터진다.
    rt = ds.set_index("steam_appid")["recommendations_total"]
    med = float(np.nanmedian(rt.reindex([int(a) for a in liked_appids]).astype("float")))

    solo_tags = cons_tags = None
    if len(liked_appids) == 1:
        solo_tags = rare_seed_tags(liked_appids, ds) or None
    else:
        # 인기 시드(리뷰 중앙 10만+) 취향에만 합의 태그를 요구한다. 니치·롱테일은 건드리지
        # 않는다 — 무조건 걸면 그쪽이 깎여 이득이 상쇄된다(postprocess docstring 참고).
        # "합의"는 시드 2개 이상에서만 정의되므로 이 분기에만 남는다.
        if med >= POPULAR_SEED_REVIEWS:
            cons_tags = consensus_seed_tags(liked_appids, ds) or None

    # 시드 인기도에 비례하는 하한. 임계값을 손으로 고르지 않으려는 연속 규칙이다.
    # 대작 취향(시드 중앙 176만)에는 큰 하한이, 니치 장르에는 약한 하한이 걸린다.
    #
    # **시드 개수와 무관하게 건다.** 처음에는 다시드 분기 안에 있었는데, 시드 중앙값은
    # 시드가 1개여도 정의되므로 그 제약엔 근거가 없었다. 평가 셋의 솔로 프로필 2개는
    # 시드가 니치라(19,987 → 19 · 5,856 → 5, 둘 다 100 미만) 차이가 안 드러났지만,
    # CS2(리뷰 513만) 하나로 추천받는 사용자는 coh_fps 를 0.77 로 끌어내렸던 에셋 플립
    # 꼬리를 그대로 받게 된다.
    #
    # **100 미만이면 아예 안 건다.** Steam 은 리뷰 100개 미만을 아예 보고하지 않아서
    # (관측된 최소값 101) 하한 1 은 "하한 없음"이 아니라 "리뷰 미보고 151,799개를
    # 전부 배제"가 된다. lowrev_detective(시드 중앙 1,238 → 하한 1)에서 100칸 중 24칸이
    # 통째로 바뀌는 것을 보고 잡았다. 저리뷰·롱테일 축을 죽이는 경로가 정확히 이것이다.
    if seed_scaled_floor and med == med:
        scaled = int(med * seed_scaled_floor)
        if scaled >= 100:
            min_reviews = max(min_reviews, scaled)

    # 후처리가 상위를 걸러내므로 넉넉히 뽑는다
    ranked = run_multi(
        liked_appids,
        strategies=[strategy],
        top_n=page_size * 30,
        rec_boost=rec_boost,
        components=components,
        postprocess=postprocess,
        postprocess_kwargs={"require_known_reviews": require_known_reviews,
                            "min_reviews": min_reviews,
                            "solo_seed_tags": solo_tags,
                            "consensus_tags": cons_tags,
                            "consensus_boost": consensus_boost,
                            "drop_dead_mp": drop_dead_mp,
                            # 이미 본 개수 = 페이지 번호 x page_size. 버킷을 그만큼 회전시켜
                            # 시드가 page_size 보다 많아도 ceil(N/page_size) 페이지 안에
                            # 모든 시드가 한 번은 나오게 한다. 회전이 없으면 약한 시드는
                            # 영영 안 나온다(실측: 시드 15개 중 11~15번째가 20페이지 내 0회).
                            "bucket_offset": len(seen)},
        exclude_appids=seen,
    )[strategy]
    return ranked.head(page_size).reset_index(drop=True)


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
