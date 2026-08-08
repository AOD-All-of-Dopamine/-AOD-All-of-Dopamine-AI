import argparse
from pathlib import Path

import numpy as np
import yaml

from src.config import PROJECT_ROOT, ensure_artifacts_dir
from src.personalization.seed_loader import SeedLoader
from src.personalization.candidate_retriever import CandidateRetriever
from src.personalization.score_aggregator import ScoreAggregator
from src.personalization.personalized_ranker import PersonalizedRanker


def build_components(rec_boost: float = 0.03, artifacts=None):
    """코퍼스 임베딩(76MB)과 dataset 을 읽는 무거운 생성자들을 한 번만 만든다.

    LOO 평가처럼 수십 번 호출하는 경우 `run_multi(..., components=...)` 로 재사용한다.
    """
    return (
        SeedLoader(artifacts),
        CandidateRetriever(artifacts),
        ScoreAggregator(),
        PersonalizedRanker(rec_boost=rec_boost, artifacts=artifacts),
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
REFRESH_MIN_REVIEWS = 300


def next_page(
    liked_appids: list[int],
    seen_appids: set[int] | list[int] | None = None,
    page_size: int = 10,
    strategy: str = "max",
    rec_boost: float = REFRESH_REC_BOOST,
    components: tuple | None = None,
    postprocess: bool = True,
    require_known_reviews: bool = True,
    min_reviews: int = REFRESH_MIN_REVIEWS,
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
    seen = set(seen_appids or ())
    # 후처리가 상위를 걸러내므로 넉넉히 뽑는다
    ranked = run_multi(
        liked_appids,
        strategies=[strategy],
        top_n=page_size * 30,
        rec_boost=rec_boost,
        components=components,
        postprocess=postprocess,
        postprocess_kwargs={"require_known_reviews": require_known_reviews,
                            "min_reviews": min_reviews},
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
