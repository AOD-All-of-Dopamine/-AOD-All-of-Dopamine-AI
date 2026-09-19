"""후처리 — 프랜차이즈 상한 · 시드 인터리빙 · 하드 필터.

랭킹은 "가장 비슷한 것"을 뽑는다. 그런데 가장 비슷한 것은 **같은 시리즈의 다른 편**이다.
크로스도메인 h67 에서 실측된 사례: `레고 닌자고 무비` → `레고 닌자고`,
`에버 에프터` → `해피 에버 에프터`. 3점을 받긴 하지만 이건 추천이 아니라 **검색**이다.

TMDB 코퍼스 실태: 같은 기본 제목 3건 이상인 그룹 1,358개 · 5,948건(9.9%).
최대 그룹은 극장판 도라에몽(40) · 명탐정 코난(33) · 루팡 3세(27) · 배트맨(25).
semantic_text 에서 제목을 이미 뺐는데도 줄거리가 비슷해서 뭉친다.
"""
import re
import weakref
from functools import lru_cache

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
    titles = set()
    for n in dataset["name"]:
        p2 = _prefix2(n)
        if p2: seen[p2].add(str(n))
        toks = [_PUNCT.sub("", t) for t in _MARKS.sub("", str(n)).strip().split()]
        toks = [t for t in toks if t]
        if len(toks) == 2: titles.add(" ".join(toks).lower())
    # 근거 둘 중 하나면 시리즈로 인정한다:
    #   (1) 같은 앞 2어절이 서로 다른 제목 `min_count` 건 이상에서 반복된다
    #   (2) **그 2어절 자체가 코퍼스의 작품 제목이다** — `다크 나이트` 가 존재하는데
    #       `다크 나이트 라이즈` 가 있으면 속편이다. (1)만으로는 라이즈 하나뿐이라 못 잡는다.
    out = {k for k, v in seen.items() if len(v) >= min_count} | (set(seen) & titles)
    franchise_prefixes._cache = out
    return out

def franchise_key(name: str, prefixes: set[str] | None = None) -> str:
    """시리즈 키 — 확인된 접두사가 있으면 그것을, 없으면 `base_title` 을 쓴다."""
    if prefixes:
        p = _prefix2(name)
        if p and p in prefixes: return re.sub(r"\s+", "", p)
    return base_title(name)


_BASE_CACHE: dict = {}
_NAMES_CACHE: dict = {}


def _by_dataset(cache: dict, dataset: pd.DataFrame, build):
    """코퍼스마다 1회만 계산해 돌려준다 (2026-09-19 서빙 지연).

    캐시 키는 `id(dataset)` 이지만 **약한 참조로 동일 객체인지 확인**한다 — dataset 은
    오래 사는 `ranker.dataset` 이라, 랭커가 버려진 뒤 다른 코퍼스가 같은 주소에 들어오면
    앞 코퍼스의 값을 돌려줄 수 있다 (Steam `_real_reviews` 와 같은 꼴).
    죽은 항목은 캐시가 커질 때 치운다 — 평가 스크립트가 코퍼스를 여러 번 만드는 경우 때문이다.
    두 번 만들어도 같은 값이라 경합에 안전하고, 돌려주는 것은 **읽기 전용**이다.
    """
    key = id(dataset)
    hit = cache.get(key)
    if hit is not None and hit[0]() is dataset:
        return hit[1]
    if len(cache) > 8:
        for k in [k for k, v in cache.items() if v[0]() is None]:
            del cache[k]
    out = build()
    cache[key] = (weakref.ref(dataset), out)
    return out


def franchise_base(dataset: pd.DataFrame) -> pd.Series:
    """row → 시리즈 키. **요청과 무관한 값**이다.

    예전에는 `cap_franchise` 가 호출마다 코퍼스 59,780개 이름에 `franchise_key`(정규식 6종)를
    다시 돌렸다 — 실측 호출당 580 ms 로 TMDB 지연의 최대 항목이었다. 식은 그대로 옮겼다.

    **전제 둘.** (1) `dataset` 의 `name`·`row` 는 기동 뒤 바뀌지 않는다(랭커가 만든 뒤 읽기만 한다).
    (2) `franchise_prefixes` 는 예전부터 **첫 dataset 의 접두사를 전역 캐시**로 들고 있었고,
    여기서도 그 함수를 그대로 부르므로 접두사 집합은 예전과 같은 것이 쓰인다.
    """
    def build():
        pref = franchise_prefixes(dataset)
        return dataset.set_index("row")["name"].map(lambda n: franchise_key(n, pref))
    return _by_dataset(_BASE_CACHE, dataset, build)


def name_by_row(dataset: pd.DataFrame) -> pd.Series:
    """row → name. `set_index("row")` 는 59,780행 인덱스 재구성이라 단계마다 반복하지 않는다.

    `row` 열이 없는 구 호출부(평가 하네스의 평탄한 프레임)는 예전과 똑같이 `dataset["name"]` 이다.
    """
    return _by_dataset(_NAMES_CACHE, dataset,
                       lambda: dataset.set_index("row")["name"]
                       if "row" in dataset.columns else dataset["name"])

# ── 시드 반복작 (D-46) ──────────────────────────────────────────────────────
# `franchise_key` 는 접두어/기본제목 기반이라 **시드의 다른 편을 하나도 못 잡았다.**
# 실측(52프로필 top-50, 2026-08-26): 할로윈→할로윈 엔드·킬즈·H20·데스데이,
# 28일 후→28주 후·28년 후, 라이온 킹→무파사: 라이온 킹, 데드풀→데드풀 2,
# 워킹 데드→피어 더 워킹 데드 — 15건 전부 통과했다.
#   할로윈 → 할로윈엔드   (공백 제거가 오히려 갈라놓는다)
#   28일 후 → 28일후 vs 28주후
#   데드풀 → 데드풀2      (접두어 경로가 base_title 의 끝숫자 제거를 건너뛴다)
#
# Steam 의 `_is_iteration()` 을 그대로 옮기면 **데드풀 2 하나만** 잡힌다 —
# 그쪽은 이름이 숫자/에디션으로 끝나는 것만 보기 때문이다. 이름 붙은 속편
# (할로윈 엔드)은 영화 쪽이 훨씬 흔하다. 그래서 토큰열 포함으로 다시 짠다.
_PERIOD = re.compile(r"^\d+\s*(일|주|개월|월|년|시간|분)$")
_NUMBERED = re.compile(r"^(\d+|[ivx]{1,4})$")

#: 이름 → 토큰열은 순수 함수이고 이름은 코퍼스(59,780편)로 한정된다. `drop_seed_iterations`
#: 가 호출마다 후보 × 시드 만큼 같은 정규식을 다시 돌리던 것을 메모이즈로 없앤다
#: (2026-09-19 서빙 지연 — 시드 50개 · k=50 에서 호출당 148 ms).
#: **전제: 위 정규식 상수(`_MARKS`·`_PERIOD`)를 런타임에 바꾸지 않는다.** 바꾸려면
#: `_tok.cache_clear()` 를 함께 불러야 한다.
_NAME_CACHE_MAX = 1 << 17

@lru_cache(maxsize=_NAME_CACHE_MAX)
def _tok(s: str) -> tuple[str, ...]:
    """캐시 본체. **튜플**을 돌려준다 — 캐시가 넘긴 객체를 호출부가 고칠 수 없게."""
    x = _MARKS.sub("", s)
    x = re.sub(r"[^0-9A-Za-z가-힣 ]", " ", x)
    return tuple("#기간" if _PERIOD.match(t) else t for t in x.lower().split() if t)

def _tokseq(name: str) -> list[str]:
    """제목을 토큰열로. `28일`·`20년` 은 **기간 슬롯**으로 뭉갠다 —
    `28일 후`와 `28주 후`를 갈라놓는 것이 오직 그 한 토큰이기 때문이다."""
    return list(_tok(str(name)))

def _is_iter(s: tuple[str, ...], c: tuple[str, ...], numbered_only: bool) -> bool:
    """토큰열끼리의 판정 — `is_seed_iteration` 의 본체를 그대로 떼어낸 것이다.

    시드 토큰열을 후보마다 다시 만들지 않으려고 이름이 아니라 토큰열을 받는다.
    리스트가 튜플로 바뀌었을 뿐 비교(`!=`·슬라이스·`rest[0]`)의 값은 같다.
    """
    if not s or len(c) < len(s):
        return False
    for i in range(len(c) - len(s) + 1):
        if c[i:i + len(s)] != s:
            continue
        if len(s) < 2 and i != 0:
            return False
        rest = c[i + len(s):]
        if numbered_only:
            return bool(rest) and _NUMBERED.match(rest[0]) is not None
        return True
    return False

def is_seed_iteration(seed_name: str, cand_name: str, numbered_only: bool = False) -> bool:
    """후보가 **시드 제목을 통째로 품고 더 뻗은** 것인가.

    방향이 핵심이다. `시드 ⊂ 후보` 만 본다 — 반대는 속편이 아니다:
        시드 `드라큐라 백작부인` → 후보 `드라큐라`      (다른 영화)
        시드 `다크 나이트`      → 후보 `다크`          (독일 드라마)
    한 토큰짜리 시드는 **후보의 맨 앞**에서만 인정한다. 안 그러면
        시드 `대부` → 후보 `리오네 사니타의 대부`(g=3) 를 잘못 지운다.

    `numbered_only=True` 면 뻗은 부분이 숫자/로마숫자일 때만 — Steam 규칙에 해당한다.
    """
    return _is_iter(_tok(str(seed_name)), _tok(str(cand_name)), numbered_only)

def drop_seed_iterations(df: pd.DataFrame, dataset: pd.DataFrame, seed_rows,
                         numbered_only: bool = False) -> pd.DataFrame:
    """시드의 다른 편을 뺀다. 컷 전에 돌려 빈칸이 뒤에서 채워지게 한다.

    사용자가 Steam 에서 직접 지적한 결함과 같은 종류다(문명 VI 시드 → VII 추천) —
    사용자는 자기가 적은 작품의 속편 존재를 이미 안다. **점수를 사는 변경이 아니다.**
    실제로 지워지는 것 중 절반 가까이가 3등급이라 P@k 는 내려갈 수 있다(D-34 긴장).
    """
    if df.empty or seed_rows is None or len(list(seed_rows)) == 0:
        return df
    names = name_by_row(dataset)
    seed_names = [str(names.get(int(r), "")) for r in seed_rows]
    seed_names = [n for n in seed_names if n]
    if not seed_names:
        return df
    # 시드 토큰열은 후보와 무관하다 — 후보마다 다시 만들지 않는다. 같은 이름의 시드가
    # 겹쳐도 `any` 의 값은 같으므로 중복은 한 번만 본다(시드 50개 경로).
    seed_toks = list(dict.fromkeys(_tok(n) for n in seed_names))
    def bad(r):
        c = _tok(str(names.get(int(r), "")))
        return any(_is_iter(s, c, numbered_only) for s in seed_toks)
    return df[~df["row"].map(bad)].reset_index(drop=True)


def cap_franchise(df: pd.DataFrame, dataset: pd.DataFrame, franchise_max: int = 1,
                  seed_rows=None, seed_franchise_max: int = 0) -> pd.DataFrame:
    """같은 기본 제목을 `franchise_max` 개까지만 남긴다.

    **시드가 속한 시리즈는 기본적으로 통째로 뺀다**(`seed_franchise_max=0`).

    행 단위 제외만으로는 부족하다 — 채점 중 실측된 사고 2건:
        시드 `아기가 생겼어요`(movie)  →  추천 `아기가 생겼어요`(tv)      같은 작품, 다른 매체
        시드 `라이온 킹`(1994 애니)    →  추천 `라이온 킹`(2019 실사)     같은 작품, 다른 버전
    TMDB 는 리메이크·실사판·TV판을 **같은 이름의 다른 행**으로 담는다. 사용자가 이미
    본 작품을 다시 추천하는 것은 추천이 아니다.

    `seed_franchise_max` 를 올리면 시드 시리즈의 다른 편도 허용된다
    (`토이 스토리` → `토이 스토리 3`). **어느 쪽이 나은지는 측정 대상이다.**
    """
    if df.empty: return df
    base = franchise_base(dataset)      # 코퍼스마다 1회 — 예전에는 호출마다 59,780행을 다시 훑었다
    df = df.copy()
    df["fbase"] = df["row"].map(base)
    seed_bases = set(base.loc[list(seed_rows)]) if seed_rows else set()
    keep, cnt = [], {}
    for t in df.itertuples(index=False):
        b = t.fbase
        lim = franchise_max if b not in seed_bases else seed_franchise_max
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
                tv_max_ratio: float | None = None, seed_rows=None,
                seed_franchise_max: int = 0,
                drop_seed_iter: str | None = None) -> pd.DataFrame:
    df = cap_franchise(ranked, dataset, franchise_max=franchise_max, seed_rows=seed_rows,
                       seed_franchise_max=seed_franchise_max)
    if drop_seed_iter:                       # D-46. "numbered" | "any"
        df = drop_seed_iterations(df, dataset, seed_rows,
                                  numbered_only=(drop_seed_iter == "numbered"))
    df = cap_media(df, dataset, tv_max_ratio)
    if interleave: df = interleave_by_seed(df, top_n=top_n)
    else: df = df.head(top_n).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df
