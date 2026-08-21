"""후처리 — 프랜차이즈 상한 · 시드 인터리빙 · 하드 필터.

랭킹은 "가장 비슷한 것"을 뽑는다. 그런데 가장 비슷한 것은 **같은 시리즈의 다른 편**이다.
크로스도메인 h67 에서 실측된 사례: `레고 닌자고 무비` → `레고 닌자고`,
`에버 에프터` → `해피 에버 에프터`. 3점을 받긴 하지만 이건 추천이 아니라 **검색**이다.

TMDB 코퍼스 실태: 같은 기본 제목 3건 이상인 그룹 1,358개 · 5,948건(9.9%).
최대 그룹은 극장판 도라에몽(40) · 명탐정 코난(33) · 루팡 3세(27) · 배트맨(25).
semantic_text 에서 제목을 이미 뺐는데도 줄거리가 비슷해서 뭉친다.
"""
import re
import numpy as np, pandas as pd

# 시리즈 기본 제목 추출용. 부제·시즌·회차를 떼어낸다.
_MARKS = re.compile(r"[®™©]")
_SEASON = re.compile(r"\s*(시즌|시리즈|Season|Series|Part|파트)\s*\d+.*$", re.I)
_TRAIL_NUM = re.compile(r"\s*(\d+|[IVX]{1,4})\s*$")
_PAREN = re.compile(r"\s*[\(（][^)）]*[\)）]\s*$")

# 관사/접두 — `더 배트맨` 과 `배트맨` 을 같은 시리즈로 묶기 위해 떼어낸다.
_ARTICLE = re.compile(r"^(the|a|an|더|디)\s+", re.I)

def base_title(name: str) -> str:
    """`극장판 도라에몽: 진구의 우주표류기` → `극장판도라에몽`.

    **공백을 전부 제거한다.** TMDB 는 같은 작품을 표기만 다르게 두 번 담는다 —
    실측: `마녀배달부 키키`(4,674표)와 `마녀 배달부 키키`(83표)가 별개 행이다.
    공백을 남기면 시리즈 상한이 이걸 못 막고 목록에 같은 영화가 두 번 들어간다.

    **관사도 뗀다** — `배트맨`(8,708표)과 `더 배트맨`(12,339표)이 함께 올라왔다.
    """
    s = _MARKS.sub("", str(name)).strip()
    s = _PAREN.sub("", s)
    s = re.split(r"\s*[:：]\s*|\s+[-–—]\s+", s)[0]
    s = _SEASON.sub("", s)
    s = _TRAIL_NUM.sub("", s)
    s = _ARTICLE.sub("", s.strip())
    s = re.sub(r"\s+", "", s).lower()          # 공백 전부 제거
    return s or _MARKS.sub("", str(name)).strip().lower()

_PUNCT = re.compile(r"[:：,·\-–—!?\.]+$")

def _prefix2(name: str) -> str:
    """앞 2어절. 어절 끝의 구두점은 뗀다 — `도라에몽:` 과 `도라에몽` 이 갈리면 안 된다."""
    toks = [_PUNCT.sub("", t) for t in _MARKS.sub("", str(name)).strip().split()]
    toks = [t for t in toks if t]
    return " ".join(toks[:2]).lower() if len(toks) >= 3 else ""

def franchise_prefixes(dataset: pd.DataFrame, min_count: int = 3) -> set[str]:
    """코퍼스에서 **실제로 확인되는** 시리즈 접두사만 모은다.

    `극장판 도라에몽: 진구의 우주표류기` 는 콜론으로 잘리지만
    `극장판 도라에몽 진구와 이상한 바람` 은 콜론이 없어 안 잘린다. 그렇다고 무조건
    앞 2어절로 자르면 `그 남자의 기억법` 과 `그 남자 그 여자` 가 뭉친다.

    ⇒ **앞 2어절이 3건 이상에서 반복되고, 그 뒤가 서로 다를 때만** 시리즈로 인정한다.
    규칙이 아니라 관측이 정한다.
    """
    cache = getattr(franchise_prefixes, "_cache", None)
    if cache is not None: return cache
    from collections import defaultdict
    seen = defaultdict(set)
    for n in dataset["name"]:
        p2 = _prefix2(n)
        if p2: seen[p2].add(str(n))
    out = {k for k, v in seen.items() if len(v) >= min_count}
    franchise_prefixes._cache = out
    return out

def franchise_key(name: str, prefixes: set[str] | None = None) -> str:
    """시리즈 키 — 확인된 접두사가 있으면 그것을, 없으면 `base_title` 을 쓴다."""
    if prefixes:
        p = _prefix2(name)
        if p and p in prefixes: return re.sub(r"\s+", "", p)
    return base_title(name)

def cap_franchise(df: pd.DataFrame, dataset: pd.DataFrame, franchise_max: int = 1,
                  seed_rows=None) -> pd.DataFrame:
    """같은 기본 제목을 `franchise_max` 개까지만 남긴다.

    **시드가 속한 시리즈는 더 강하게 막는다** — 사용자가 이미 아는 시리즈다.
    다만 완전히 막지는 않는다(`극장판 도라에몽`을 좋아하면 다른 편도 유효한 추천일 수 있다).
    """
    if df.empty: return df
    pref = franchise_prefixes(dataset)
    base = dataset.set_index("row")["name"].map(lambda n: franchise_key(n, pref))
    df = df.copy()
    df["fbase"] = df["row"].map(base)
    seed_bases = set(base.loc[list(seed_rows)]) if seed_rows else set()
    keep, cnt = [], {}
    for t in df.itertuples(index=False):
        b = t.fbase
        lim = franchise_max if b not in seed_bases else max(1, franchise_max)
        if cnt.get(b, 0) < lim:
            cnt[b] = cnt.get(b, 0) + 1
            keep.append(True)
        else:
            keep.append(False)
    return df[np.array(keep)].drop(columns=["fbase"]).reset_index(drop=True)

def interleave_by_seed(ranked: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """시드별로 라운드로빈. 강한 시드가 목록을 독점하는 것을 막는다.

    크로스도메인에서 실측된 병리 — 시드 2649 는 3칸 전부 `헬블레이드` 하나에서 나왔고
    나머지 두 시드는 추천에 아무 영향도 못 줬다. 사용자는 셋을 줬는데 하나만 반영된다.
    Steam 에서도 인터리빙을 끄면 `twenty_broad` 가 20개 시드 중 4개에서만 추천을 받았다.

    **앞 단계의 순서를 버리지 않는다** — 각 시드 안의 순서는 그대로 두고 뽑기만 번갈아 한다.
    """
    if ranked.empty or "dominant_seed" not in ranked: return ranked.head(top_n)
    groups = {s: list(g.index) for s, g in ranked.groupby("dominant_seed", sort=False)}
    order = sorted(groups, key=lambda s: ranked.loc[groups[s][0], "rank"])
    out, i = [], 0
    while len(out) < top_n and any(groups.values()):
        moved = False
        for s in order:
            if groups[s]:
                out.append(groups[s].pop(0)); moved = True
                if len(out) >= top_n: break
        if not moved: break
        i += 1
    res = ranked.loc[out].reset_index(drop=True)
    res["rank"] = np.arange(1, len(res) + 1)
    return res

def cap_media(df: pd.DataFrame, dataset: pd.DataFrame, tv_max_ratio: float | None = None) -> pd.DataFrame:
    """TV 비율 상한. `None` 이면 끔 (기본값 — 근거 없이 켜지 않는다)."""
    if tv_max_ratio is None or df.empty: return df
    media = dataset.set_index("row")["media"]
    is_tv = df["row"].map(media).eq("tv").to_numpy()
    lim = int(len(df) * tv_max_ratio)
    keep, n = [], 0
    for t in is_tv:
        if t:
            keep.append(n < lim); n += int(n < lim)
        else: keep.append(True)
    return df[np.array(keep)].reset_index(drop=True)

def postprocess(ranked: pd.DataFrame, dataset: pd.DataFrame, top_n: int = 50,
                franchise_max: int = 1, interleave: bool = True,
                tv_max_ratio: float | None = None, seed_rows=None) -> pd.DataFrame:
    df = cap_franchise(ranked, dataset, franchise_max=franchise_max, seed_rows=seed_rows)
    df = cap_media(df, dataset, tv_max_ratio)
    if interleave: df = interleave_by_seed(df, top_n=top_n)
    else: df = df.head(top_n).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df
