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

    `ideal_pool`은 IDCG를 계산할 판정 모집단이다. pooled 평가에서는 **해당 앵커/프로필의
    판정 풀 전체**를 넘겨야 한다 — 그래야 여러 변형을 같은 분모로 비교할 수 있다.
    None이면 `rels` 자신을 모집단으로 쓴다(단일 랭킹의 자기 기준 NDCG).
    """
    rels = [float(r) for r in rels[:k]]
    pool = rels if ideal_pool is None else [float(r) for r in ideal_pool]
    idcg = dcg(sorted(pool, reverse=True)[:k], mode)
    return dcg(rels, mode) / idcg if idcg else 0.0


def precision_at_k(
    rels: Sequence[float], k: int = 10, threshold: int = POSITIVE_THRESHOLD
) -> float:
    rels = [float(r) for r in rels[:k]]
    return sum(1 for r in rels if r >= threshold) / len(rels) if rels else 0.0


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
