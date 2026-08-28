"""믹싱 규칙 M0 · M1 · M2 — 순위만 쓴다. 점수는 받지 않는다.

입력:  lists  = {platform: [item_id, ...]}   (각 플랫폼 랭커의 top-N, 순위순)
       seeds  = {platform: n_seeds}           (시드 없는 플랫폼은 키 자체가 없음)
       coh    = {platform: seed_cohesion}     (M2 만 쓴다)
출력:  [(platform, item_id, platform_rank), ...]  길이 ≤ k

**빈 자리를 다른 플랫폼으로 채우지 않는다** (D-23 · §48). 목록이 짧아진다.
"""
import math

ORDER = ("steam", "tmdb", "wn")   # 동수일 때의 고정 순서


def _platform_order(seeds):
    return sorted(seeds, key=lambda p: (-seeds[p], ORDER.index(p)))


def _quota(seeds, k, weights=None):
    w = weights or {p: seeds[p] for p in seeds}
    tot = sum(w.values())
    q = {p: max(1, round(k * w[p] / tot)) for p in w}
    # 반올림 합이 k 를 넘으면 가장 큰 쿼터부터 1씩 줄인다
    while sum(q.values()) > k:
        p = max(q, key=lambda x: (q[x], -ORDER.index(x)))
        q[p] -= 1
    return q


def _take(lists, quota, k):
    """쿼터 순서대로 라운드로빈. 플랫폼이 바닥나면 그 자리는 **비운다**."""
    ptr = {p: 0 for p in quota}
    left = dict(quota)
    out = []
    plats = list(quota)
    while len(out) < k and any(left[p] > 0 and ptr[p] < len(lists.get(p, [])) for p in plats):
        for p in plats:
            if len(out) >= k: break
            if left[p] <= 0 or ptr[p] >= len(lists.get(p, [])): continue
            out.append((p, lists[p][ptr[p]], ptr[p] + 1))
            ptr[p] += 1; left[p] -= 1
    return out


def M0(lists, seeds, coh=None, k=10):
    """기준선 — 라운드로빈. 플랫폼당 ⌈k/P⌉."""
    plats = _platform_order(seeds)
    per = math.ceil(k / len(plats))
    return _take(lists, {p: per for p in plats}, k)


def M1(lists, seeds, coh=None, k=10):
    """시드 비례 쿼터. 최소 1."""
    plats = _platform_order(seeds)
    q = _quota({p: seeds[p] for p in plats}, k)
    return _take(lists, {p: q[p] for p in plats}, k)


def M2(lists, seeds, coh, k=10):
    """M1 + 저응집(<0.5) 플랫폼은 쿼터 절반. D-22."""
    plats = _platform_order(seeds)
    w = {p: seeds[p] * (0.5 if coh.get(p, 1.0) < 0.5 else 1.0) for p in plats}
    q = _quota({p: seeds[p] for p in plats}, k, weights=w)
    return _take(lists, {p: q[p] for p in plats}, k)


def M3(lists, seeds, coh=None, k=10, wn_min_episodes=20, episodes=None):
    """M0 + 웹소설 후보에서 화수 < wn_min_episodes 제외 (X-2).

    플랫폼 랭커가 아니라 **통합 층**에서 거른다 — 화수는 플랫폼 내 등급 신호가 아니지만
    (D-62 잔차 −0.038) "게임·영화 옆에 3화짜리를 내미는 게 맞나"라는 통합 물음에서는 신호다.
    `episodes`: {item_id: episode_count}. 없으면 M0 과 같다.
    """
    if episodes and "wn" in lists:
        lists = dict(lists)
        lists["wn"] = [i for i in lists["wn"] if episodes.get(i, episodes.get(str(i), 0)) >= wn_min_episodes]
    return M0(lists, seeds, coh, k)


def _wnfilter(lists, episodes, th=20):
    if episodes and "wn" in lists:
        lists = dict(lists); lists["wn"] = [i for i in lists["wn"] if episodes.get(i, episodes.get(str(i), 0)) >= th]
    return lists

def M4(lists, seeds, coh=None, k=10, wn_min_episodes=20, episodes=None):
    """M1 + 웹소설 화수 필터 (X-5)."""
    return M1(_wnfilter(lists, episodes, wn_min_episodes), seeds, coh, k)

def M5(lists, seeds, coh, k=10, wn_min_episodes=20, episodes=None):
    """M2 + 웹소설 화수 필터 (X-5)."""
    return M2(_wnfilter(lists, episodes, wn_min_episodes), seeds, coh, k)


def M6(lists, seeds, coh=None, k=10, wn_min_episodes=20, episodes=None):
    """M3 + **시드가 1개뿐인 플랫폼의 쿼터를 절반**으로 (X-10).

    독립 3인 잣대(§59) 600 슬롯에서 플랫폼 시드 1개인 슬롯 P 0.42 vs 나머지 0.84.
    좋아한 작품 하나는 약한 증거다. 그 플랫폼 쿼터를 반으로 줄이고 나머지는 시드 있는
    다른 플랫폼에 라운드로빈으로 돌린다 (뺏은 자리를 비우지 않는다 — D-23 의 "바닥남"과 다르다).
    응집도 ≥0.7 인 다중 시드는 P 가 0.25~1.00 으로 갈려 신호가 아니라 제외.
    """
    lists = _wnfilter(lists, episodes, wn_min_episodes)
    plats = _platform_order(seeds)
    w = {p: (0.5 if seeds[p] == 1 else 1.0) for p in plats}
    q = _quota({p: 1 for p in plats}, k, weights=w)
    return _take(lists, q, k)


def _tmdbfilter(lists, votes, th=100):
    if votes and "tmdb" in lists:
        lists = dict(lists); lists["tmdb"] = [i for i in lists["tmdb"] if votes.get(i, votes.get(str(i), 0)) >= th]
    return lists


def M7(lists, seeds, coh=None, k=10, wn_min_episodes=20, episodes=None, tmdb_min_votes=100, votes=None):
    """M6 + **TMDB 후보 투표수 < tmdb_min_votes 제외** (X-11).

    독립 3인 잣대 TMDB 282 슬롯에서 순위 잔차 corr(log투표) +0.258, 투표 <100 슬롯 P 0.54.
    `votes`: {tmdb_id: vote_count}. 없으면 M6 과 같다.
    """
    return M6(_tmdbfilter(lists, votes, tmdb_min_votes), seeds, coh, k, wn_min_episodes, episodes)


RULES = {"M0": M0, "M1": M1, "M2": M2, "M3": M3, "M4": M4, "M5": M5, "M6": M6, "M7": M7}
