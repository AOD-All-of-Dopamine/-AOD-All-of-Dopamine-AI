import argparse
from pathlib import Path

import numpy as np
import yaml

from src.config import PROJECT_ROOT, ensure_artifacts_dir, PRODUCTION
from src.personalization.seed_loader import SeedLoader
from src.personalization.candidate_retriever import CandidateRetriever
from src.personalization.score_aggregator import ScoreAggregator
from src.personalization.personalized_ranker import PersonalizedRanker


def build_components(pop_boost: float | None = None, artifacts=None):
    """코퍼스 임베딩(76MB)과 dataset 을 읽는 무거운 생성자들을 한 번만 만든다.

    LOO 평가처럼 수십 번 호출하는 경우 `run_multi(..., components=...)` 로 재사용한다.
    `pop_boost=None` 이면 `config.PRODUCTION` 의 확정값을 쓴다 (D-66).
    """
    pop_boost = PRODUCTION["pop_boost"] if pop_boost is None else pop_boost
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
    pop_boost: float | None = None,      # None = config.PRODUCTION (D-66)
    output_dir: str | None = None,
    components: tuple | None = None,
    postprocess: bool = False,
    postprocess_kwargs: dict | None = None,
    exclude_ids: set[int] | list[int] | None = None,
    drop_excluded_series: bool | None = None,   # None = PRODUCTION (W-6)
    blocked_rows=None,
) -> dict[str, dict]:
    """`postprocess=True` 면 랭킹 뒤에 다양성 후처리(시드 인터리빙·시리즈 상한·hard filter)를 건다.

    후처리는 상위를 걸러내므로 랭커에서 넉넉히(top_n × 5) 뽑은 뒤 잘라야 한다.

    `exclude_ids` — **새로고침 제품의 필수 입력.** 이게 없으면 이 함수는 순수 함수라
    같은 입력에 같은 목록을 낸다(= 새로고침해도 화면이 안 바뀐다). 여기에 넣을 것:
      · 이미 보여준 것    (`aod_ai.rec_impression`)
      · 이미 아는 것      (LIKE / DISLIKE / bookmark / 리뷰 작성한 콘텐츠)
    시드(`liked_ids`)는 자동으로 합쳐지므로 따로 넣지 않아도 된다.

    `blocked_rows` — **서빙 가능 목록**(2026-09-19). 코퍼스 행 순서(`full_corpus_frame()` 의 행
    순서)에 맞춘 불리언 배열로, True 인 행은 후보에서 뺀다. 백엔드 카탈로그에 없는 작품을 추천해
    봐야 카드로 못 만드는 것을 막는 **제품이 정한 후보 범위**이고 랭킹 공식은 그대로다. `None`
    이면 아무것도 하지 않는다 — 기존 호출부·평가 하네스는 전부 이 경로라 결과가 비트 단위로 같다.

    **`exclude_ids` 와는 다른 자리에 건다.** `exclude_ids` 는 아래에서 `drop_seed_series`(W-6)
    에도 넘어가 "같은 작품의 다른 판본"까지 함께 지우는데, 서빙 가능 목록을 거기 넣으면 카탈로그에
    **있는** 판본까지 자기 자신의 다른 판본 때문에 사라진다(코퍼스 29,494행 중 4,164행이 중복 키
    그룹이다). 그래서 목록은 랭커에 넘기는 후보 프레임에서만 뺀다. 점수는 행마다 독립이고 정렬은
    걸러낸 뒤이므로, 거르는 시점이 점수 계산 앞이든 뒤든 남는 행과 값이 같다.
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
        candidates = aggregated[strategy]
        if blocked_rows is not None:
            # 서빙 가능 목록 — 랭커에 넘기기 전에만 뺀다. 아래 `drop_seed_series` 가 받는
            # `excluded` 에는 들어가지 않는다(위 docstring).
            candidates = candidates[~blocked_rows]
        ranked = ranker.rank(
            candidates,
            exclude_ids=excluded,
            top_n=rank_n,
        )
        if postprocess:
            from src.postprocess import postprocess as apply_postprocess

            dx = PRODUCTION["drop_excluded_series"] if drop_excluded_series is None else drop_excluded_series
            if dx and excluded:
                # 제외는 id 로만 된다. 시드·이미 본 작품의 **다른 판본**(작가|제목 키가 같은 것)도 뺀다 —
                # 없으면 1페이지에 시드 판본이, 2·3페이지에 앞 페이지 작품의 판본이 다시 뜬다(W-6).
                from src.postprocess import drop_seed_series
                # 랭커가 이미 `item_id` 로 인덱싱해 둔 dataset 을 그대로 넘긴다. 예전에는
                # 단계마다 `reset_index()` → `set_index()` 로 29,494행 인덱스를 다시 세웠다
                # (2026-09-19 서빙 지연). 후처리는 이 프레임을 읽기만 한다.
                ranked = drop_seed_series(ranked, ranker.dataset, list(excluded))
            ranked = apply_postprocess(
                ranked, ranker.dataset, top_n=top_n,
                **(postprocess_kwargs or {}),
            )
        results[strategy] = ranked
        if output_dir:
            out_path = Path(output_dir) / f"ranked_{strategy}.parquet"
            ranked.to_parquet(out_path, index=False)
            print(f"  Saved: {out_path}")

    return results


# **D-66 으로 확정값에 맞췄다.** 0.15 는 Steam 의 깊은-페이지 진단(리뷰 중앙 2,831→434)을
# 보고 고른 값이었고, 코드 자신이 *"숫자는 Steam 것이고 이 도메인에서 다시 판정해야 한다 …
# 검증된 값이 아니다"* 라고 적어 뒀다. 웹소설에서 실제로 재니 **0.15 는 미달이다**:
#     k=20 적합률 0.9452 (최고 대비 −0.0221) · 개별 최대하락 −0.350 → (A)(C) 위반
# 확정값 0.03 은 k=20·k=50 양쪽에서 통과했다. 이제 여기도 `PRODUCTION` 을 읽는다.
REFRESH_POP_BOOST = PRODUCTION["pop_boost"]

# 품질 하한 기본값. `dataset_profile.json` 의 interest_count 분포를 보고 정한다.
# Steam 은 결측이 곧 "리뷰 거의 없음"이라 이진 판정이 가능했지만, 관심 수는 연속값이라
# 임계를 명시해야 한다. 코퍼스가 바뀌면 이 값도 다시 봐야 한다.
# **끈다 (D-62).** `configs/wn_v1.yaml` 이 100 으로 적어 두었고 이 경로만 적용하고 있었다
# (평가 하네스는 안 걸었다 — 갈라짐 자체가 D-55 와 같은 종류다). 그런데 실측하면
# **걸면 손해다.** 52프로필 top-50 2,600슬롯에서:
#     관심수 0        607슬롯 적합률 **0.959**
#     관심수 1~99     172슬롯 적합률 **0.971**
#     관심수 1천~1만  516슬롯 적합률 0.940
#     관심수 10만+    822슬롯 적합률 0.960
# 100 미만(779슬롯·전체의 30%) 0.961 vs 100 이상 0.958 — **차이 −0.003.**
# 신호 측정에서도 log 관심수 잔차상관 **−0.006 (24/52)** 로 동전던지기다.
# 웹소설 코퍼스(7,062)는 이미 출판사가 큐레이션한 목록이라 관심수 0 이
# "쓰레기"가 아니라 "신작·니치"다. Steam(173,691, 리뷰 0 이 87.4%)과 다르다.
REFRESH_MIN_INTEREST = None


def next_page(
    liked_ids: list[int],
    seen_ids: set[int] | list[int] | None = None,
    page_size: int = 10,
    strategy: str = PRODUCTION["strategy"],   # X-20a. 예전에는 "max" 로 갈라져 있었다 —
                                             # 확정값은 top2_mean 이고 D-55 와 같은 종류의 미적용이었다.
    pop_boost: float = REFRESH_POP_BOOST,
    components: tuple | None = None,
    postprocess: bool = True,
    min_interest_count: int | None = REFRESH_MIN_INTEREST,
    drop_excluded_series: bool | None = None,   # None = PRODUCTION (W-6)
    blocked_rows=None,
):
    """새로고침 한 번 = 이 함수 한 번. 서빙이 쓸 계약을 코드로 고정한다.

    `blocked_rows` — 서빙 가능 목록(후보 제외 전용). `run_multi` 로 그대로 넘어간다.
    `None`(기본값)이면 예전과 결과가 완전히 같다. 자세한 근거는 `run_multi` docstring.

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

    **정정 (2026-09-04 플랫폼 검수).** 이 문단은 원래 `pop_boost=0.15` 를 쓰면서
    "0.03 은 유사도 스프레드보다 작아 정렬을 거의 못 바꾼다"고 적어 두었다. **틀렸다.**
    92프로필 실측: `pop_boost` 0.03 만으로 추천 top-50 의 관심 수 중앙이
    **16,000 → 40,500 (2.5배)** 로 뛴다(코퍼스 중앙 7,300). 유사도 낙차가 10위→50위에서
    0.049 뿐이라 3% 배율이면 그 구간을 다시 정렬하기에 충분하다.
    값 자체는 이미 D-66 에서 `REFRESH_POP_BOOST = PRODUCTION["pop_boost"]` 로 통일됐고
    (제품 경로도 0.03 이다) 여기 남아 있던 것은 **근거가 반증된 설명**뿐이었다.

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
        drop_excluded_series=drop_excluded_series,
        # 서빙 가능 목록은 `exclude_ids` 와 합치지 않는다 — 합치면 `drop_excluded_series` 가
        # 목록 안에 있는 판본까지 지운다(위 docstring).
        blocked_rows=blocked_rows,
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
