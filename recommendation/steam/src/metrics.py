# src/metrics.py
"""평가 지표 단일 소스.

이 프로젝트는 한동안 세 곳에서 NDCG를 각자 구현했다 — `evaluate.py`는 linear gain,
`eval_ranking.py`·`eval_personalization.py`는 exponential gain. 그래서 S1 숫자와 P1 숫자를
나란히 놓을 수 없었다. gain 함수는 이 모듈에서만 정의하고, 나머지는 전부 여기서 가져다 쓴다.

gain 모드는 설정(`evaluation.ndcg_gain`)에서 오며 기본값은 exponential이다.
0~3 판정 척도에서 "매우 타당(3)"과 "보통(2)"의 차이를 벌려주는 편이 추천 품질 판단에 맞다.
"""
import math
from collections.abc import Iterable, Sequence

import pandas as pd

LINEAR = "linear"
EXPONENTIAL = "exponential"
DEFAULT_GAIN = EXPONENTIAL
POSITIVE_THRESHOLD = 2


def gain(relevance: float, mode: str = DEFAULT_GAIN) -> float:
    r = float(relevance)
    if mode == LINEAR:
        return r
    if mode == EXPONENTIAL:
        return 2.0**r - 1.0
    raise ValueError(f"알 수 없는 ndcg_gain: {mode!r} (가능: {LINEAR!r}, {EXPONENTIAL!r})")


def dcg(rels: Iterable[float], mode: str = DEFAULT_GAIN) -> float:
    return sum(gain(r, mode) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(
    rels: Sequence[float],
    k: int = 10,
    ideal_pool: Iterable[float] | None = None,
    mode: str = DEFAULT_GAIN,
) -> float:
    """상위 k개의 NDCG.

    `ideal_pool`은 IDCG를 계산할 판정 모집단이다. None이면 `rels` 자신을 쓴다.

    **`ideal_pool` 을 "그 프로필에서 판정된 것 전부"로 넘기면 안 된다.** 프로필마다
    판정량이 다르면 분모가 달라져 NDCG 가 랭킹 품질이 아니라 판정량을 재게 된다.
    실측(26개 프로필, 판정 514쌍):

        pool_size ↔ NDCG 상관  피어슨 -0.671 / 스피어만 -0.713
        P@10 이 똑같이 0.600 인 프로필끼리:  pool<=10 → NDCG 0.806
                                            pool>=30 → NDCG 0.524

    같은 정확도인데 많이 채점한 프로필이 1.5배 벌을 받는다. 그래서 프로필 간 평균을 내면
    무의미한 숫자가 나온다(실측 '전체 NDCG 0.766' 은 pool 10짜리 18개와 30+ 짜리 8개를
    섞은 값이었다).

    비교 대상 전 변형의 Top-k union 을 **모든 프로필에 대해 균일하게** 판정했을 때만
    pool 을 넘겨도 된다. 그 조건이 아니면 `ideal_pool=None`(= 페이지 자기 기준)을 쓴다 —
    그때 NDCG 는 "보여준 k개를 올바른 순서로 놓았는가"를 재고, 판정량과 무관해진다.
    """
    rels = [float(r) for r in rels[:k]]
    pool = rels if ideal_pool is None else [float(r) for r in ideal_pool]
    idcg = dcg(sorted(pool, reverse=True)[:k], mode)
    if not idcg:
        return 0.0
    actual = dcg(rels, mode)
    if actual > idcg + 1e-9:
        # 수학적으로 불가능하다 — pool ⊇ rels 이면 DCG ≤ IDCG 가 항상 성립한다.
        # 따라서 이 예외는 "판정 풀이 랭킹 상위 k개를 다 담고 있지 않다"는 뜻이고,
        # 그대로 두면 NDCG 가 1을 넘는다. 실측으로 P07 1.160 / P02 1.021 이 나온 적이 있다.
        raise ValueError(
            f"NDCG > 1 ({actual / idcg:.3f}) — ideal_pool 이 랭킹 상위 {k}개를 덮지 못합니다. "
            f"pool 크기 {len(pool)}, 랭킹 {len(rels)}개. "
            "비교 대상 전 변형의 Top-k union 을 전부 판정한 뒤 다시 계산하세요 "
            "(assert_pool_coverage 참고)."
        )
    return actual / idcg


JUDGED_DENOM = "judged"
SLOT_DENOM = "k"


def precision_at_k(
    rels: Sequence[float],
    k: int = 10,
    threshold: int = POSITIVE_THRESHOLD,
    denominator: str = JUDGED_DENOM,
) -> float:
    """상위 k개 중 `threshold` 이상의 비율.

    **분모를 명시해야 한다.** 둘은 다른 질문에 답한다:

      · `"judged"` — 판정된 개수로 나눈다. "채점한 것 중 몇 %가 좋았나".
        `assert_pool_coverage` 로 Top-k 가 전부 판정된 것이 보장될 때만 안전하다.
        보장이 없으면 **미판정 칸이 분모에서도 빠져 정밀도가 부풀려진다** —
        후보를 많이 갈아치우는 설정일수록 유리해지는 방향이라 특히 위험하다.
      · `"k"` — 페이지 칸 수로 나눈다. "10칸 중 몇 칸이 쓸만했나".
        제품이 항상 10칸을 채워 보여주므로 제품 경로(`eval_product`)의 기본값이다.
        미판정/빈 칸은 실패로 계산된다.

    기존 호출부(S1 `eval_ranking`, P1 `eval_personalization`)는 `"judged"` 를 유지한다 —
    frozen summary.json 재현성이 걸려 있다. `eval_ranking` 은 pool 커버리지를 강제하므로
    두 분모가 어차피 같다.
    """
    rels = [float(r) for r in rels[:k]]
    hits = sum(1 for r in rels if r >= threshold)
    if denominator == SLOT_DENOM:
        return hits / k if k else 0.0
    if denominator == JUDGED_DENOM:
        return hits / len(rels) if rels else 0.0
    raise ValueError(f"알 수 없는 denominator: {denominator!r} (가능: {JUDGED_DENOM!r}, {SLOT_DENOM!r})")


def paired_bootstrap_ci(
    baseline: Sequence[float],
    variant: Sequence[float],
    iterations: int = 20000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """같은 프로필 집합에서 잰 두 설정의 차이에 신뢰구간을 붙인다.

    **왜 짝지어야 하나** — 프로필마다 난이도가 크게 다르다(P07 0.30 ~ P05 0.80). 짝을 풀면
    그 분산이 전부 잡음으로 들어와 어떤 차이도 유의하지 않게 나온다. 프로필별 차이를
    재표집하면 프로필 난이도가 상쇄된다.

    **왜 필수인가** — 프로필 8개 × 10칸이면 1칸이 0.0125다. 이 하네스로 잰 값들을
    실측으로 검정해보면:

        리뷰 하한 300 추가      Δ +0.138  95%CI [+0.013, +0.237]  유의
        코퍼스 확대 단독        Δ -0.063  95%CI [-0.250, +0.137]  판정 불가
        인기도 부스트 상향      Δ +0.013  95%CI [-0.125, +0.113]  판정 불가

    즉 지금 표본으로 잡히는 것은 0.15 이상의 효과뿐이다. 그보다 작은 차이를 "개선"이라고
    부르면 안 된다. 표본을 늘리기 전까지는 이 함수가 그것을 매번 상기시킨다.
    """
    import numpy as np

    a = np.asarray(baseline, dtype=float)
    b = np.asarray(variant, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"짝이 맞지 않습니다: baseline {a.shape} vs variant {b.shape}")
    if a.size < 2:
        raise ValueError("짝지은 표본이 2개 미만입니다 — 신뢰구간을 낼 수 없습니다.")

    diff = b - a
    rng = np.random.default_rng(seed)
    draws = rng.choice(diff, size=(iterations, diff.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {
        "n": int(diff.size),
        "delta": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


def format_ci(result: dict, label: str = "") -> str:
    verdict = "유의" if result["significant"] else "판정 불가"
    return (f"{label:<28s} Δ={result['delta']:+.4f}  "
            f"95%CI [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]  {verdict} "
            f"(n={result['n']})")


def _norm(v):
    """int64/float 혼재로 키가 어긋나는 것을 막는다 (19476.0 != 19476)."""
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _key_set(df: pd.DataFrame, cols: Sequence[str]) -> set[tuple]:
    return {tuple(_norm(v) for v in row) for row in df[list(cols)].to_numpy().tolist()}


def assert_pool_coverage(
    judged: pd.DataFrame,
    ranked: pd.DataFrame,
    key_cols: Sequence[str],
    k: int = 10,
    label: str = "variant",
    rank_col: str = "rank",
) -> int:
    """랭킹 상위 k개가 판정 풀에 100% 들어있는지 검사하고, 검사한 행 수를 반환한다.

    계약: **pool = 비교 대상 전 변형의 Top-k union이고, pool 안은 전부 판정한다.**
    이 계약이 지켜지면 "미판정을 0점으로 채울지 제외할지" 문제가 애초에 생기지 않는다.
    새 변형을 추가하면 여기서 실패하는 것이 정상이다 — 조용히 0점을 주는 것보다 낫다.
    """
    top = ranked[ranked[rank_col] <= k]
    judged_keys = _key_set(judged, key_cols)
    missing = sorted(_key_set(top, key_cols) - judged_keys)
    if missing:
        preview = ", ".join(str(m) for m in missing[:5])
        raise ValueError(
            f"{label}: Top-{k} 중 {len(missing)}쌍이 판정 풀에 없습니다 — pool 재생성이 필요합니다. "
            f"({', '.join(key_cols)}) 예시: {preview}"
            + (" ..." if len(missing) > 5 else "")
        )
    return len(top)
