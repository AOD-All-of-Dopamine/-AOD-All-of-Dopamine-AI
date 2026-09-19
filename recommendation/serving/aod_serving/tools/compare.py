"""기준 목록 비교 규칙. 같은 머신에서는 완전 일치, 다른 머신에서는 **기준 점수가 동점인 자리**의 순서 차이만 허용한다 —
동점의 순서는 pandas 기본 정렬(불안정)이 정하고 CPU 의 SIMD 경로에 따라 달라진다."""
from __future__ import annotations


def tied_positions(scores: list[float]) -> set[int]:
    """이웃과 점수가 같은 자리. 마지막 자리는 페이지 밖 항목과 동점일 수 있어 항상 포함한다."""
    n = len(scores)
    out = {n - 1} if n else set()
    for j in range(n - 1):
        if scores[j] == scores[j + 1]:
            out |= {j, j + 1}
    return out


def pages_equal(want_ids, got_ids, want_scores=None, allow_ties: bool = False) -> bool:
    want_ids, got_ids = list(want_ids), list(got_ids)
    if want_ids == got_ids:
        return True
    if not (allow_ties and want_scores) or len(want_ids) != len(got_ids):
        return False
    ok = tied_positions(list(want_scores))
    return all(w == g or j in ok for j, (w, g) in enumerate(zip(want_ids, got_ids)))
