"""페이지 구성 — **점수를 섞지 않고 자리를 나눈다.**

배율형 인기 반영은 네 번 재서 네 번 기각됐다(TMDB D-60 · 웹툰 T-2/T-6 · 웹소설 W-1).
매번 같은 이유였다 — **하나의 상수로 대작 취향과 롱테일 취향을 동시에 만족시킬 수 없다.**
평균은 조금 오르는데 롱테일 프로필이 −0.10 씩 무너진다.

여기서는 점수를 건드리지 않는다. 유사도 순위는 그대로 두고 **10칸짜리 페이지의 구성비**만 정한다.

    인기 칸 수 = clamp(round(10 × 시드 인기 백분위), 하한, 상한)

시드가 이미 취향을 말해 준다 — 인터스텔라·인셉션을 준 사람(백분위 0.97)은 유명작을 원하고,
투표 33개짜리 남아공 스릴러를 준 사람(0.05)은 그렇지 않다. 같은 6칸을 둘 다에게 주면 후자가 깨진다.

**하한 3 은 제품 결정이다** (2026-09-13). 백분위가 아무리 낮아도 10칸 중 3칸은 인기작을 준다 —
시드가 무명이어도 아는 작품이 하나도 없으면 목록 전체를 신뢰하기 어렵다는 판단이다. 측정으로 정한 값이 아니다.

**인기 칸은 상위 `pop_window` 안에서만 고른다.** 이게 없으면 꼬리 끝에서 유명한 것이 끌려 올라온다 —
실측: top-100 전체에서 고르면 98위 `스콜피온`(천재 해커물, 투표 4,172)이 1페이지에 들어왔다.
브레이킹 배드 시드와 거리가 먼데 투표수만 많아서다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

#: 인기 칸의 하한·상한 (10칸 기준). 하한은 제품 결정, 상한은 유사도 칸을 최소 3 남기려는 것.
POP_MIN, POP_MAX = 3, 7
#: 인기 칸을 고를 범위. 유사도 상위 이 순위 안에서만 고른다.
POP_WINDOW = 30


def seed_popularity_pct(vote_pct: dict, seed_ids) -> float:
    """시드의 인기 백분위 중앙. 0(무명) ~ 1(대작)."""
    v = [vote_pct.get(i) for i in seed_ids]
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else 0.5


def pop_quota(seed_pct: float, page: int = 10,
              lo: int = POP_MIN, hi: int = POP_MAX) -> int:
    """시드 백분위 → 인기 칸 수. 하한·상한으로 자른다."""
    return int(np.clip(round(page * seed_pct), lo, hi))


def compose(ranked_ids, vote_of: dict, seed_pct: float, page: int = 10,
            window: int = POP_WINDOW, lo: int = POP_MIN, hi: int = POP_MAX):
    """유사도 순 후보 → 한 페이지.

    반환은 `(item_id, "pop"|"sim")` 목록이다. 어느 칸에서 왔는지 남겨 두면
    나중에 사람 채점·클릭 로그로 두 칸의 성적을 따로 잴 수 있다.
    """
    ids = list(ranked_ids)
    q = pop_quota(seed_pct, page, lo, hi)
    head = ids[:window]
    pop = sorted(head, key=lambda i: -vote_of.get(i, 0))[:q]
    used = set(pop)
    sim = [i for i in ids if i not in used][:page - q]
    # 유사도 칸이 먼저, 인기 칸이 뒤 — 상위는 취향 적합을 우선한다
    out = [(i, "sim") for i in sim] + [(i, "pop") for i in pop]
    return out, q
