import argparse
from pathlib import Path

import numpy as np
import yaml

from src.config import PROJECT_ROOT, ensure_artifacts_dir
from src.personalization.seed_loader import SeedLoader
from src.personalization.candidate_retriever import CandidateRetriever
from src.personalization.score_aggregator import ScoreAggregator
from src.personalization.personalized_ranker import PersonalizedRanker


def build_components(pop_boost: float = 0.03, artifacts=None):
    """코퍼스 임베딩(76MB)과 dataset 을 읽는 무거운 생성자들을 한 번만 만든다.

    LOO 평가처럼 수십 번 호출하는 경우 `run_multi(..., components=...)` 로 재사용한다.
    """
    return (
        SeedLoader(artifacts),
        CandidateRetriever(artifacts),
        ScoreAggregator(),
        PersonalizedRanker(pop_boost=pop_boost, artifacts=artifacts),
    )


def run_multi(
    liked_ids: list[int],
    strategies: list[str] | None = None,
    top_n: int = 300,
    pop_boost: float = 0.03,
    output_dir: str | None = None,
    components: tuple | None = None,
    postprocess: bool = False,
    postprocess_kwargs: dict | None = None,
    exclude_ids: set[int] | list[int] | None = None,
) -> dict[str, dict]:
    """`postprocess=True` 면 랭킹 뒤에 다양성 후처리(시드 인터리빙·시리즈 상한·hard filter)를 건다.

    후처리는 상위를 걸러내므로 랭커에서 넉넉히(top_n × 5) 뽑은 뒤 잘라야 한다.

    `exclude_ids` — **새로고침 제품의 필수 입력.** 이게 없으면 이 함수는 순수 함수라
    같은 입력에 같은 목록을 낸다(= 새로고침해도 화면이 안 바뀐다). 여기에 넣을 것:
      · 이미 보여준 것    (`aod_ai.rec_impression`)
      · 이미 아는 것      (LIKE / DISLIKE / bookmark / 리뷰 작성한 콘텐츠)
    시드(`liked_ids`)는 자동으로 합쳐지므로 따로 넣지 않아도 된다.
    """
    if strategies is None:
        strategies = ["max", "mean", "top2_mean"]

    loader, retriever, aggregator, ranker = components or build_components(pop_boost)

    seed_embs = loader.load(liked_ids)
    sim_matrix = retriever.compute_similarity_matrix(seed_embs)
    corpus_df = retriever.full_corpus_frame()
    aggregated = aggregator.aggregate_all(sim_matrix, seed_embs, corpus_df, strategies=strategies)

    excluded = set(liked_ids) | set(exclude_ids or ())
    rank_n = top_n * 5 if postprocess else top_n
    results = {}
    for strategy in strategies:
        ranked = ranker.rank(
            aggregated[strategy],
            exclude_ids=excluded,
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


# 새로고침 깊은 페이지에서 품질이 유지되도록 맞춘 값. 근거는 next_page docstring 참고.
REFRESH_POP_BOOST = 0.15

# 품질 하한 기본값. `dataset_profile.json` 의 interest_count 분포를 보고 정한다.
# Steam 은 결측이 곧 "리뷰 거의 없음"이라 이진 판정이 가능했지만, 관심 수는 연속값이라
# 임계를 명시해야 한다. 코퍼스가 바뀌면 이 값도 다시 봐야 한다.
REFRESH_MIN_INTEREST = 100


def next_page(
    liked_ids: list[int],
    seen_ids: set[int] | list[int] | None = None,
    page_size: int = 10,
    strategy: str = "max",
    pop_boost: float = REFRESH_POP_BOOST,
    components: tuple | None = None,
    postprocess: bool = True,
    min_interest_count: int | None = REFRESH_MIN_INTEREST,
):
    """새로고침 한 번 = 이 함수 한 번. 서빙이 쓸 계약을 코드로 고정한다.

    호출자는 반환된 `item_id` 를 `seen_ids` 에 누적해서 다음 호출에 넘겨야 한다.
    그 누적을 어디에 저장할지가 서빙의 과제다(`aod_ai.rec_impression`).

    **기본값이 run_multi 와 다른 이유** — 깊은 페이지가 무너지는 것을 막기 위해서다.
    Steam 에서 측정된 원인은 관련성 붕괴가 아니라 **품질 붕괴**였다: 유사도는 1→5페이지에서
    0.75→0.71 로 거의 안 변하는데 리뷰 수 중앙값이 2,831→434 로 무너졌다. 유사도가 평평한
    구간에서는 미세한 유사도 차이보다 인기도가 훨씬 나은 정렬 기준이다. 그 판정 결과:

      페이지   현재(부스트 0.03, 하한X)   개선(0.15, 하한O)
        1      평균 1.79 / P@10 .64      1.85 / .68
        2      평균 1.44 / P@10 .49      1.53 / .53
        3      평균 1.06 / P@10 .32      1.63 / .55

    웹소설에서 같은 병이 더 심할 것으로 본다 — 관심 수 2짜리 작품이 평점 10.0 을 달고
    코퍼스에 대량으로 깔려 있기 때문이다. 다만 **숫자는 Steam 것이고 이 도메인에서 다시
    판정해야 한다.** 그 전까지 이 기본값은 "합리적 출발점"이지 검증된 값이 아니다.

      · `min_interest_count` — 품질 하한(연속값 임계). None 이면 끈다.
      · `pop_boost=0.15` — 0.03 은 유사도 스프레드보다 작아 정렬을 거의 못 바꾼다.

    `run_multi` 기본값은 실험 재현성 때문에 건드리지 않는다 — 제품 경로인 이 함수에서만 바꾼다.
    """
    seen = set(seen_ids or ())
    # 후처리가 상위를 걸러내므로 넉넉히 뽑는다
    ranked = run_multi(
        liked_ids,
        strategies=[strategy],
        top_n=page_size * 30,
        pop_boost=pop_boost,
        components=components,
        postprocess=postprocess,
        postprocess_kwargs={"min_interest_count": min_interest_count},
        exclude_ids=seen,
    )[strategy]
    return ranked.head(page_size).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="P1 Multi-Seed Personalized Retrieval (Full Corpus)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--liked-ids", type=int, nargs="+", help="시드 작품 product_no")
    parser.add_argument("--strategies", type=str, nargs="+", default=["max", "mean", "top2_mean"])
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--pop-boost", type=float, default=0.03)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        liked = cfg.get("liked_ids", args.liked_ids)
        strategies = cfg.get("aggregation", {}).get("strategies", args.strategies)
        top_n = cfg.get("aggregation", {}).get("top_n", args.top_n)
        pop_boost = cfg.get("aggregation", {}).get("pop_boost", args.pop_boost)
        output_dir = cfg.get("paths", {}).get("ranked_dir", args.output_dir)
    else:
        liked = args.liked_ids
        strategies = args.strategies
        top_n = args.top_n
        pop_boost = args.pop_boost
        output_dir = args.output_dir

    if not liked:
        parser.error("--liked-ids is required (or set liked_ids in config)")

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_dir = str(ensure_artifacts_dir())

    results = run_multi(
        liked_ids=liked,
        strategies=strategies,
        top_n=top_n,
        pop_boost=pop_boost,
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
