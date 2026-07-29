# src/postprocess.py
"""랭킹 **뒤**에 붙는 후처리.

랭커를 건드리지 않는 독립 단계라 S1/P1 양쪽에 재사용되고, 끄고 켜서 비교할 수 있다.
설계 스펙 §5.4 의 순서를 따른다: hard filter → 시리즈 상한 → 다양성(인터리빙) → Top-N.

**왜 필요한가** (20개 프로필 실측):
  - 한 시드가 Top-100 의 73.4% 를 차지한다. 최악은 99/100.
    깊이를 늘려도 다른 취향이 나오지 않고 같은 시드의 것이 더 나올 뿐이다.
  - 10개씩 새로고침하는 제품에서 이건 "2페이지가 1페이지랑 똑같다"로 나타난다.
    새로고침 제품의 병목은 정확도가 아니라 변화량이다.
"""
import re
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, artifact_dir

# 장르 태그로 걸러낼 성인/고어 콘텐츠. anchor_builder.NON_CORE_GENRE_LABELS 의 부분집합이다
# (그쪽은 '인디'·'앞서 해보기'처럼 장르가 아닌 것도 포함하므로 그대로 쓰면 안 된다).
ADULT_GENRES = frozenset({"신체 노출", "선정적 콘텐츠"})
GORE_GENRES = frozenset({"고어", "폭력적"})
VR_ONLY_CATEGORY = "VR 전용"

TREND_FEATURES = PROJECT_ROOT / "artifacts" / "trend_v1" / "trend_features.parquet"

_SERIES_STRIP = re.compile(r"[®™©]")
_SERIES_NOISE = re.compile(
    r"\b(\d+|i{1,3}|iv|v|vi|vii|viii|ix|x|remastered|enhanced|definitive|special|"
    r"complete|collection|edition|goty|hd|vr|classic|source|deluxe|ultimate)\b"
)


def series_key(name: str) -> str:
    """시리즈 근사. 개발사/퍼블리셔 필드가 dataset 에 없어 이름으로 추정한다.

    'Counter-Strike 2' 와 'Counter-Strike: Source' → 'counter strike'
    'MadOut' 과 'MadOut Ice Storm' → 'madout ice'  (완벽하지 않다 — 앞 2단어 근사)
    """
    s = _SERIES_STRIP.sub("", str(name)).lower()
    s = re.sub(r"[^a-z0-9가-힣 ]", " ", s)
    s = _SERIES_NOISE.sub(" ", s)
    toks = s.split()
    return " ".join(toks[:2]) if toks else str(name).lower()


def _load_release_dates() -> pd.Series:
    """미출시 판정용. trend_features 의 parsed_date 를 재사용한다.

    trend_features.parse_release_date() 는 원본 jsonl 을 읽어야 해서 이 환경에서 못 돈다.
    커밋된 산출물에 이미 파싱 결과가 들어 있으므로 그것을 쓴다.
    """
    if not TREND_FEATURES.exists():
        return pd.Series(dtype="object")
    t = pd.read_parquet(TREND_FEATURES)
    return t.set_index("steam_appid")["parsed_date"]


def apply_hard_filters(
    ranked: pd.DataFrame,
    dataset: pd.DataFrame,
    drop_unreleased: bool = True,
    drop_adult: bool = True,
    drop_vr_only: bool = True,
    require_known_reviews: bool = False,
) -> pd.DataFrame:
    """추천으로 내보내면 안 되는 것을 제거한다.

    고어/폭력은 거르지 않는다 — 그건 취향이지 결격 사유가 아니다.

    `require_known_reviews` — 품질 하한. Steam 은 리뷰 수가 일정 이상일 때만
    `recommendations.total` 을 보고한다. 실측: 코퍼스 19,476개 중 값이 있는 것은
    7,558개(39%)뿐이고 **있는 값은 전부 100 이상**이다. 즉 결측 = "리뷰가 거의 없는 게임"
    이라는 깨끗한 이진 신호다.

    깊은 페이지가 무너지는 원인이 이것이다 — 유사도는 거의 안 변하는데(1페이지 0.75 →
    5페이지 0.71) 리뷰 수 중앙값이 2,831 → 434 로 붕괴한다. 관련성이 떨어지는 게 아니라
    아무도 안 해본 게임으로 채워진다.
    """
    df = ranked.copy()
    meta = dataset.set_index("steam_appid") if "steam_appid" in dataset.columns else dataset
    keep = pd.Series(True, index=df.index)

    if drop_adult:
        genres = df["steam_appid"].map(lambda a: set(meta["genres"].get(a, [])))
        keep &= ~genres.map(lambda g: bool(g & ADULT_GENRES))

    if drop_vr_only and "categories" in meta.columns:
        cats = df["steam_appid"].map(lambda a: set(meta["categories"].get(a, [])))
        keep &= ~cats.map(lambda c: VR_ONLY_CATEGORY in c)

    if require_known_reviews and "has_recommendations" in meta.columns:
        keep &= df["steam_appid"].map(lambda a: bool(meta["has_recommendations"].get(a, False)))

    if drop_unreleased:
        dates = _load_release_dates()
        if len(dates):
            # "표에 없다"(모름)와 "표에 있는데 날짜가 없다"(미출시/파싱실패)를 구분한다.
            # 모르는 것을 미출시로 취급하면 전부 걸러진다.
            known = set(dates.index)
            keep &= df["steam_appid"].map(
                lambda a: a not in known or pd.notna(dates.get(a))
            )

    return df[keep].reset_index(drop=True)


def cap_series(df: pd.DataFrame, dataset: pd.DataFrame, series_max: int = 1) -> pd.DataFrame:
    """같은 시리즈를 최대 series_max 개만 남긴다.

    인터리빙은 각 시드의 최근접을 끌어올리는데, 시드의 최근접은 종종 자기 프랜차이즈다
    (Skyrim → Skyrim VR, Witcher 3 → Witcher Adventure Game). 그래서 인터리빙과
    **함께** 걸어야 한다.
    """
    if df.empty:
        return df.reset_index(drop=True)
    meta = dataset.set_index("steam_appid") if "steam_appid" in dataset.columns else dataset
    names = df["steam_appid"].map(lambda a: meta["name"].get(a, ""))
    keys = names.map(series_key)
    seen: dict[str, int] = {}
    keep = []
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
        keep.append(seen[k] <= series_max)
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def interleave_by_seed(ranked: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """dominant_seed 버킷을 라운드로빈으로 배치한다.

    **전역 상한이 아니라 라운드로빈이어야 한다.** 상한만 걸면 "1~50위는 농장, 51~100위는
    FPS"가 되어 1~5페이지가 전부 농장이다. 라운드로빈은 매 페이지에 시드가 섞이게 한다.

    버킷 순서는 각 버킷 1등의 final_score 내림차순 — 가장 강한 시드가 1위 자리를 갖는다.
    빈 버킷은 자동으로 건너뛰므로, 코퍼스에 이웃이 없는 시드가 있어도 목록이 짧아지지 않는다.
    """
    if ranked.empty or "dominant_seed" not in ranked.columns or ranked["dominant_seed"].isna().all():
        out = ranked.head(top_n).reset_index(drop=True)
        if len(out):
            out["rank"] = range(1, len(out) + 1)
        return out

    buckets = [
        g.sort_values("final_score", ascending=False)
        for _, g in ranked.groupby("dominant_seed", sort=False)
    ]
    buckets.sort(key=lambda b: -b["final_score"].iloc[0])

    picked, depth = [], 0
    while len(picked) < top_n and any(depth < len(b) for b in buckets):
        for b in buckets:
            if depth < len(b):
                picked.append(b.iloc[depth])
                if len(picked) >= top_n:
                    break
        depth += 1

    out = pd.DataFrame(picked).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out


def postprocess(
    ranked: pd.DataFrame,
    dataset: pd.DataFrame,
    top_n: int = 100,
    seed_interleave: bool = True,
    series_max: int = 1,
    hard_filters: bool = True,
    require_known_reviews: bool = False,
) -> pd.DataFrame:
    """스펙 §5.4 순서: hard filter → 시리즈 상한 → 다양성 → Top-N."""
    df = ranked
    if hard_filters:
        df = apply_hard_filters(df, dataset, require_known_reviews=require_known_reviews)
    if series_max:
        df = cap_series(df, dataset, series_max)
    if seed_interleave:
        df = interleave_by_seed(df, top_n)
    else:
        df = df.head(top_n).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    return df


def load_dataset(artifacts: str | Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(artifact_dir(artifacts) / "dataset.parquet")
