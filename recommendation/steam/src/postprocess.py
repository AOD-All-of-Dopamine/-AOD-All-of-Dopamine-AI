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
    min_reviews: int = 0,
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

    `min_reviews` — 이진 신호로는 부족하다는 것이 전체 코퍼스에서 드러났다. 코퍼스를
    19,476 → 173,691 로 키우자 `has_recommendations` 를 통과하는 게임이 7,558 → 21,892 로
    늘면서 Top-10 리뷰 중앙값이 7,742 → 2,210 으로 무너졌다. 판정 258쌍으로 잰 리뷰 수 대
    적합률은 단조 증가한다:

        리뷰 <500   적합 29%   무관 26%
        리뷰 500-2k  적합 57%
        리뷰 2k-10k  적합 57%
        리뷰 10k+    적합 78%   무관  5%

    하한 300 을 걸면 P@10 0.562 → 0.713 (구 코퍼스 0.625 도 상회). 하한을 10,000 까지
    올리면 다시 떨어진다 — 유명작만 남아 발견의 가치가 사라지기 때문이다.
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

    if min_reviews > 0 and "recommendations_total" in meta.columns:
        # Int64 결측을 0 으로 눕혀야 한다 — pd.NA 는 비교에서 불리언이 되지 않는다.
        totals = meta["recommendations_total"].astype("float").fillna(0.0)
        keep &= df["steam_appid"].map(lambda a: totals.get(a, 0.0) >= min_reviews)

    if drop_unreleased and "coming_soon" in meta.columns:
        # dataset 이 직접 들고 있으면 그것을 쓴다 — 전체 코퍼스에 적용된다
        keep &= ~df["steam_appid"].map(lambda a: bool(meta["coming_soon"].get(a, False)))
    elif drop_unreleased:
        # 구 dataset 에는 컬럼이 없다 → trend_features 로 폴백(커버리지 11%)
        dates = _load_release_dates()
        if len(dates):
            # "표에 없다"(모름)와 "표에 있는데 날짜가 없다"(미출시/파싱실패)를 구분한다.
            # 모르는 것을 미출시로 취급하면 전부 걸러진다.
            known = set(dates.index)
            keep &= df["steam_appid"].map(
                lambda a: a not in known or pd.notna(dates.get(a))
            )

    return df[keep].reset_index(drop=True)


def series_group(name: str, publisher: str = "") -> str:
    """시리즈 식별자. 퍼블리셔가 있으면 `퍼블리셔|이름앞부분` 으로 더 정확해진다.

    실데이터 검증 결과(2026-08, 전체 코퍼스):
      'MadOut' vs 'MadOut Ice Storm'    이름기반 놓침 → 퍼블리셔로 **잡음**
      'War Trigger 3' vs 'WT2'          퍼블리셔가 같아도 **여전히 놓침**
                                        (`rocketeer|war` vs `rocketeer|wt2`)
      'Skyrim' vs 'Elder Scrolls Online' 둘 다 잡지만 **잡으면 안 된다**
                                        (ESO 는 MMO — 판정에서 2점을 준 정당한 추천)

    즉 이 함수는 놓치기도 하고 과하게 잡기도 한다. 이름 첫 단어를 붙이는 설계가 양방향
    실패의 원인인데, 빼면 대형 퍼블리셔의 무관한 작품까지 한 시리즈로 묶인다.
    그래서 이 함수는 '확실한 동일 시리즈'만 담당하고, 나머지는 `cap_publisher` 가 맡는다.

    구 데이터처럼 퍼블리셔가 없으면 이름 기반으로 자동 폴백한다.
    """
    key = series_key(name)
    pub = str(publisher or "").strip().lower()
    if not pub:
        return key
    head = key.split()[0] if key.split() else key
    return f"{pub}|{head}"


def cap_publisher(df: pd.DataFrame, dataset: pd.DataFrame, publisher_max: int = 2) -> pd.DataFrame:
    """한 퍼블리셔가 목록을 점유하는 것을 막는다. 시리즈 상한이 못 잡는 것을 덮는다.

    왜 시리즈 상한만으로 부족한가 — 실측 사례:
      · 'War Trigger 3' / 'WT2' 는 이름이 달라 시리즈 판정이 못 잡지만 퍼블리셔가 같다.
      · P06(CS2/PUBG/Stardew) Top-10 에 Valve 게임이 4개 들어왔다
        (CS:Source 1점, Day of Defeat 3점, Half-Life 2점, Alien Swarm 2점).

    왜 상한 1 이 아니라 2 인가 — 리뷰 있는 21,892개의 퍼블리셔가 9,378곳이고 71% 가
    1개만 냈다. 상한 1 은 대형 퍼블리셔의 **다른 시리즈**까지 잘라낸다
    (Skyrim 과 Elder Scrolls Online 은 같은 Bethesda 지만 다른 경험이다).
    """
    if df.empty or publisher_max <= 0:
        return df.reset_index(drop=True)
    meta = dataset.set_index("steam_appid") if "steam_appid" in dataset.columns else dataset
    if "publisher" not in meta.columns:
        return df.reset_index(drop=True)

    seen: dict[str, int] = {}
    keep = []
    for a in df["steam_appid"]:
        pub = str(meta["publisher"].get(a, "") or "").strip().lower()
        if not pub:  # 퍼블리셔 미상은 제한하지 않는다
            keep.append(True)
            continue
        seen[pub] = seen.get(pub, 0) + 1
        keep.append(seen[pub] <= publisher_max)
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def cap_series(df: pd.DataFrame, dataset: pd.DataFrame, series_max: int = 1) -> pd.DataFrame:
    """같은 시리즈를 최대 series_max 개만 남긴다.

    인터리빙은 각 시드의 최근접을 끌어올리는데, 시드의 최근접은 종종 자기 프랜차이즈다
    (Skyrim → Skyrim VR, Witcher 3 → Witcher Adventure Game). 그래서 인터리빙과
    **함께** 걸어야 한다.
    """
    if df.empty:
        return df.reset_index(drop=True)
    meta = dataset.set_index("steam_appid") if "steam_appid" in dataset.columns else dataset
    has_pub = "publisher" in meta.columns
    keys = df["steam_appid"].map(
        lambda a: series_group(
            meta["name"].get(a, ""),
            meta["publisher"].get(a, "") if has_pub else "",
        )
    )
    seen: dict[str, int] = {}
    keep = []
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
        keep.append(seen[k] <= series_max)
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def interleave_by_seed(ranked: pd.DataFrame, top_n: int = 100,
                       bucket_offset: int = 0) -> pd.DataFrame:
    """dominant_seed 버킷을 라운드로빈으로 배치한다.

    **전역 상한이 아니라 라운드로빈이어야 한다.** 상한만 걸면 "1~50위는 농장, 51~100위는
    FPS"가 되어 1~5페이지가 전부 농장이다. 라운드로빈은 매 페이지에 시드가 섞이게 한다.

    버킷 순서는 각 버킷 1등의 final_score 내림차순 — 가장 강한 시드가 1위 자리를 갖는다.
    빈 버킷은 자동으로 건너뛰므로, 코퍼스에 이웃이 없는 시드가 있어도 목록이 짧아지지 않는다.

    **시드 개수에 따른 동작** (실측, page_size=10 기준):

        시드  1개 → 한 시드가 10/10. 인터리빙은 아무 일도 하지 않는다(정상)
        시드  3개 → 최다 점유 4/10
        시드 10개 → 각 시드 1칸씩
        시드 15개 → 한 페이지에 10개만 담긴다 → `bucket_offset` 으로 돌려가며 태운다

    `bucket_offset` — **시드가 page_size 보다 많을 때 약한 시드가 굶어 죽는 것을 막는다.**
    이게 없으면 매 페이지마다 버킷 정렬이 처음부터 다시 시작해 항상 같은 상위 10개 버킷이
    이긴다. 실측(시드 15개, 버킷당 후보 30개):

        offset 없음 → 시드별 첫 등장 페이지 {0~9: 1, 10: 11, 11~14: 20페이지 내 없음}
        offset 있음 → {0~9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6}   (회전 폭 1)
        offset = 본 개수 → 2페이지 안에 15개 전부                      (회전 폭 page_size)

    `next_page` 는 `len(seen_appids)` 를 그대로 넘긴다. 페이지마다 page_size 씩 늘어나므로
    회전 폭이 page_size 가 되고, 시드 N개가 `ceil(N / page_size)` 페이지 안에 전부 등장한다.
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
    if bucket_offset and len(buckets) > 1:
        k = bucket_offset % len(buckets)
        buckets = buckets[k:] + buckets[:k]

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
    publisher_max: int = 2,
    hard_filters: bool = True,
    require_known_reviews: bool = False,
    min_reviews: int = 0,
    bucket_offset: int = 0,
) -> pd.DataFrame:
    """스펙 §5.4 순서: hard filter → 시리즈/퍼블리셔 상한 → 다양성 → Top-N.

    `bucket_offset` 은 인터리빙에 그대로 넘어간다 — 시드가 page_size 보다 많을 때
    페이지마다 버킷 순서를 회전시켜 약한 시드가 굶지 않게 한다.
    """
    df = ranked
    if hard_filters:
        df = apply_hard_filters(df, dataset, require_known_reviews=require_known_reviews,
                                min_reviews=min_reviews)
    if series_max:
        df = cap_series(df, dataset, series_max)
    if publisher_max:
        df = cap_publisher(df, dataset, publisher_max)
    if seed_interleave:
        df = interleave_by_seed(df, top_n, bucket_offset=bucket_offset)
    else:
        df = df.head(top_n).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    return df


def load_dataset(artifacts: str | Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(artifact_dir(artifacts) / "dataset.parquet")
