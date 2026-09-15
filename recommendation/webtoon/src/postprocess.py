"""후처리 — 하드 필터 → 시리즈 상한 → 작가 상한 → 시드 교차 → Top-N.
순서는 세 플랫폼과 동일하다(먼저 지우고, 그다음 다양성, 마지막에 자른다)."""
from __future__ import annotations
import re
import pandas as pd

_BRACKET = re.compile(r"[\[\(<][^\]\)>]*[\]\)>]")
_NOISE = re.compile(r"(시즌\s*\d+|시즌|\d+\s*부|외전|번외|리메이크|리마스터|재연재|단편|프리퀄|스핀오프)")


def series_key(name: str) -> str:
    """`화산귀환 시즌2` · `[리메이크] 나 혼자 만렙` 같은 꼬리를 떼고 앞 두 토큰."""
    s = _BRACKET.sub(" ", str(name)).lower()
    s = _NOISE.sub(" ", s)
    s = re.sub(r"[^0-9a-z가-힣\s]", " ", s)
    toks = [t for t in s.split() if t]
    return " ".join(toks[:2]) if toks else str(name).lower()


def apply_hard_filters(df: pd.DataFrame, dataset: pd.DataFrame, drop_adult: bool = True):
    """성인물 제거. 수집 단계에서 대부분 빠지지만 여기서도 막는다."""
    if not drop_adult or "item_id" not in df.columns:
        return df.reset_index(drop=True)
    ad = dataset.set_index("item_id")
    keep = []
    for i in df["item_id"]:
        try:
            r = ad.loc[int(i)]; r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            keep.append(not (bool(r.get("adult", False)) or str(r.get("age_type", "")) == "RATE_19"))
        except KeyError:
            keep.append(True)
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def _cap(df, dataset, col_fn, cap: int):
    seen: dict[str, int] = {}; keep = []
    ad = dataset.set_index("item_id")
    for i in df["item_id"]:
        try:
            r = ad.loc[int(i)]; r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            k = col_fn(r)
        except KeyError:
            k = None
        if k is None or k == "":
            keep.append(True); continue
        n = seen.get(k, 0)
        keep.append(n < cap); seen[k] = n + 1
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def cap_series(df, dataset, series_max: int = 1):
    return _cap(df, dataset, lambda r: series_key(r["name"]), series_max)


def cap_artist(df, dataset, artist_max: int = 2):
    def key(r):
        a = r.get("artists")
        if a is None or (hasattr(a, "__len__") and len(a) == 0): return str(r.get("author", ""))
        return str(list(a)[0])
    return _cap(df, dataset, key, artist_max)


def drop_seed_series(df, dataset, seed_ids):
    """시드와 같은 시리즈는 뺀다 — 이미 본 작품의 시즌2를 추천하지 않는다."""
    ad = dataset.set_index("item_id")
    keys = set()
    for s in seed_ids:
        try:
            r = ad.loc[int(s)]; r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            keys.add(series_key(r["name"]))
        except KeyError:
            pass
    keep = []
    for i in df["item_id"]:
        try:
            r = ad.loc[int(i)]; r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            keep.append(series_key(r["name"]) not in keys)
        except KeyError:
            keep.append(True)
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def interleave_by_seed(ranked: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """dominant_seed 버킷 라운드로빈 — 시드 하나가 목록을 독점하지 못하게."""
    if "dominant_seed" not in ranked.columns:
        return ranked.head(top_n).reset_index(drop=True)
    buckets: dict = {}
    for _, r in ranked.iterrows():
        buckets.setdefault(r["dominant_seed"], []).append(r)
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    out, i = [], 0
    while len(out) < top_n and any(buckets[k] for k in order):
        k = order[i % len(order)]
        if buckets[k]: out.append(buckets[k].pop(0))
        i += 1
    return pd.DataFrame(out).head(top_n).reset_index(drop=True)


def postprocess(df, dataset, top_n=50, seed_ids=None, series_max=1, artist_max=2,
                drop_adult=True, interleave=True, drop_series_ids=None):
    df = apply_hard_filters(df, dataset, drop_adult)
    if seed_ids: df = drop_seed_series(df, dataset, seed_ids)
    # 싫어요한 작품의 다른 시즌·외전도 뺀다. series_key 한계: 제목 끝에 숫자가 붙은 속편은 못 묶는다.
    if drop_series_ids: df = drop_seed_series(df, dataset, drop_series_ids)
    if series_max: df = cap_series(df, dataset, series_max)
    if artist_max: df = cap_artist(df, dataset, artist_max)
    if interleave: df = interleave_by_seed(df, top_n)
    return df.head(top_n).reset_index(drop=True)
