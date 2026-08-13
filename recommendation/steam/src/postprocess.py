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

# Steam 이 개발사에게 받는 **공식** 성인 콘텐츠 등급. 3 = 노골적 성적 묘사, 4 = 성인 전용.
#
# 왜 사용자 태그가 아니라 이것인가 — 태그로 성인물을 판정하려고 규칙을 다섯 번 만들었고
# 다섯 번 다 양방향으로 틀렸다:
#
#     태그 존재      Witcher 3 · Cyberpunk 2077 · Baldur's Gate 3 차단
#     성인태그 상위3  Cyberpunk 2077 차단
#     성인태그 상위2  GTA V · Fear & Hunger 차단, Tricolour Lovestory 통과
#     Hentai 만      PAYDAY 3 · Age of History II 차단, House Party 통과
#
# 원인은 태그가 **사용자 투표**라는 것이다. 밈으로도 붙고(PAYDAY 3 에 Hentai), 표 수가
# 인기도와 교란된다. descriptor 는 지정값이라 그 오염이 없다. 실측:
#
#     3/4 있음  FlipWitch · Carnal Instinct · Acting Lessons · Subverse · HuniePop
#     3/4 없음  PAYDAY 3 · Age of History II · Hatred · BG3 · Rust · RDR2 · GTA V ·
#               Bayonetta · 스텔라 블레이드 · Doki Doki Literature Club
#
# **완벽하지는 않다** — House Party · NEKOPARA · Summer Memories 는 개발사가 지정을 안 해
# 통과한다. 놓치는 쪽(false negative)이지 정상 게임을 잃는 쪽은 아니라서 이 방향을 택했다.
ADULT_DESCRIPTOR_IDS = frozenset({3, 4})
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


def _has_adult_descriptor(ids) -> bool:
    """content_descriptorids 에 성인 등급이 있는가.

    parquet 왕복 후에는 list 가 아니라 numpy 배열로 돌아오고, 결측은 None 이다.
    `set(x or ())` 로 쓰면 배열에서 ValueError 가 난다.
    """
    if ids is None:
        return False
    try:
        return bool(ADULT_DESCRIPTOR_IDS & {int(i) for i in ids})
    except TypeError:
        return False


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
        if "content_descriptorids" in meta.columns:
            # parquet 는 리스트를 numpy 배열로 돌려준다 — `x or ()` 는 배열에서
            # "truth value is ambiguous" 로 터진다. None 검사를 명시적으로 한다.
            cd = meta["content_descriptorids"]
            keep &= ~df["steam_appid"].map(
                lambda a: _has_adult_descriptor(cd.get(a)))

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


def series_counts(appids, dataset: pd.DataFrame) -> dict[str, int]:
    """appid 목록의 시리즈 키별 등장 횟수. cap_series 의 세션 사전값으로 쓴다."""
    meta = dataset.set_index("steam_appid") if "steam_appid" in dataset.columns else dataset
    has_pub = "publisher" in meta.columns
    out: dict[str, int] = {}
    for a in appids:
        a = int(a)
        k = series_group(
            meta["name"].get(a, ""),
            meta["publisher"].get(a, "") if has_pub else "",
        )
        out[k] = out.get(k, 0) + 1
    return out


def cap_series(df: pd.DataFrame, dataset: pd.DataFrame, series_max: int = 1,
               prior_counts: dict[str, int] | None = None,
               session_max: int | None = None) -> pd.DataFrame:
    """같은 시리즈를 페이지당 최대 series_max 개, **세션 전체로 session_max 개**만 남긴다.

    인터리빙은 각 시드의 최근접을 끌어올리는데, 시드의 최근접은 종종 자기 프랜차이즈다
    (Skyrim → Skyrim VR, Witcher 3 → Witcher Adventure Game). 그래서 인터리빙과
    **함께** 걸어야 한다.

    **왜 세션 상한이 따로 필요한가 (2026-08-13 홀드아웃에서 발견).** 페이지 내 상한은
    호출 한 번 안에서만 작동한다. 새로고침은 페이지마다 별도 호출이므로, 프랜차이즈가
    밀집한 축에서는 "페이지마다 정확히 1개"가 규칙적으로 반복됐다 — ho_sports 100칸에
    Axis Football 이 7개(순위 19·25·34·46·78·83·91, 페이지당 1개씩). `prior_counts` 에
    이미 보여준 칸들의 시리즈 수를 넘기면 세션 누계로 자른다.

    session_max 를 1로 두면 안 된다: Trails 시리즈처럼 각 편이 전부 명작인 경우
    (ho_jrpg 에서 5편 모두 판정 3) 반복이 오히려 옳다. 기본값 근거는
    REFRESH_SERIES_SESSION_MAX 주석 참고.
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
    prior = dict(prior_counts or {})
    seen: dict[str, int] = {}
    keep = []
    for k in keys:
        page_n = seen.get(k, 0) + 1
        total_n = prior.get(k, 0) + page_n
        ok = page_n <= series_max and (session_max is None or total_n <= session_max)
        if ok:
            seen[k] = page_n
        keep.append(ok)
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


#: 이 비율보다 흔한 태그는 "정의적"이지 않다고 본다. Singleplayer(68%)·Indie(55%)·
#: Casual(44%)·Action(43%) 처럼 절반 가까운 게임이 달고 있는 태그로는 취향이 안 좁혀진다.
RARE_TAG_MAX_DF = 0.05
SEED_TAG_TOPN = 15


def rare_seed_tags(seed_appids, dataset: pd.DataFrame,
                   max_df: float = RARE_TAG_MAX_DF, topn: int = SEED_TAG_TOPN) -> set[str]:
    """시드의 상위 태그 중 코퍼스에서 드문 것들 — 그 취향을 정의하는 태그."""
    tags = dataset.set_index("steam_appid")["tags"] if "tags" in dataset.columns else None
    if tags is None:
        return set()

    def names(x):
        if x is None or isinstance(x, float):
            return []
        return [t["name"] if isinstance(t, dict) else str(t) for t in x]

    df_count: dict[str, int] = {}
    for x in tags:
        for t in set(names(x)):
            df_count[t] = df_count.get(t, 0) + 1
    n = len(tags)
    out = set()
    for a in seed_appids:
        for t in names(tags.get(int(a))) [:topn]:
            if df_count.get(t, 0) / n < max_df:
                out.add(t)
    return out


#: 시드 리뷰 중앙값이 이 이상이면 "대작 취향" — 합의 태그 필터를 건다.
POPULAR_SEED_REVIEWS = 100_000
#: 합의 태그로 인정할 최대 문서빈도. 이보다 흔하면 취향을 안 좁힌다.
CONSENSUS_TAG_MAX_DF = 0.15


def consensus_seed_tags(seed_appids, dataset: pd.DataFrame, min_seeds: int = 2,
                        max_df: float = CONSENSUS_TAG_MAX_DF, topn: int = SEED_TAG_TOPN) -> set[str]:
    """시드 `min_seeds` 개 이상이 **함께** 가진 태그 중 흔하지 않은 것.

    `rare_seed_tags` 와 다른 점: 저쪽은 합집합이라 "한 시드의 특이점"까지 통과시킨다.
    다시드 프로필에 그걸 걸었더니 `lowrev_cozy_narrative` 가 0.90 → 0.75 로 무너졌다
    (Visual Novel 이 정의적이라고 오인돼 일본식 연애 VN 이 대거 통과). 합의를 요구하면
    시드들이 실제로 공유하는 축만 남는다:

        coh_arpg  → Action RPG · Fantasy · Open World · Third Person
        coh_cozy  → Building · Crafting · Sandbox · Open World
    """
    tags = dataset.set_index("steam_appid")["tags"] if "tags" in dataset.columns else None
    if tags is None:
        return set()

    def names(x):
        if x is None or isinstance(x, float):
            return []
        return [t["name"] if isinstance(t, dict) else str(t) for t in x]

    df_count: dict[str, int] = {}
    for x in tags:
        for t in set(names(x)):
            df_count[t] = df_count.get(t, 0) + 1
    n = len(tags)

    shared: dict[str, int] = {}
    for a in seed_appids:
        for t in set(names(tags.get(int(a)))[:topn]):
            shared[t] = shared.get(t, 0) + 1
    return {t for t, c in shared.items() if c >= min_seeds and df_count.get(t, 0) / n < max_df}


def keep_sharing_tags(df: pd.DataFrame, dataset: pd.DataFrame, wanted: set[str],
                      topn: int = SEED_TAG_TOPN) -> pd.DataFrame:
    """후보의 상위 태그가 `wanted` 와 하나라도 겹치는 것만 남긴다."""
    if not wanted:
        return df
    tags = dataset.set_index("steam_appid")["tags"]

    def hit(a):
        x = tags.get(int(a))
        if x is None or isinstance(x, float):
            return False
        got = {t["name"] if isinstance(t, dict) else str(t) for t in x[:topn]}
        return bool(wanted & got)

    keep = df["steam_appid"].map(hit)
    return df[keep] if keep.any() else df


#: 싱글플레이 카테고리 표기(스팀이 로케일별로 다르게 준다).
SINGLEPLAYER_MARKERS = ("Single-player", "싱글 플레이어", "シングルプレイヤー", "单人")
#: 온라인 대전/협동 표기. 대전은 사람이 없으면 성립 자체가 안 되고, 협동은 친구를 데려오면 된다.
ONLINE_PVP_MARKERS = ("Online PvP", "온라인 PvP", "オンラインPvP", "在线 PvP")
ONLINE_COOP_MARKERS = ("Online Co-op", "온라인 협동", "オンライン協力プレイ", "在线合作")


def drop_dead_multiplayer(df: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """**멀티 전용인데 플레이어 기반이 측정조차 안 되는** 후보를 뺀다.

    인기도 하한이 아니다. 멀티플레이 전용 게임은 사람이 없으면 실행이 안 되는 것과 같다 —
    싱글 카테고리가 없고 Steam 이 리뷰 수를 보고하지도 않는다면(대략 100개 미만) 매칭이
    잡히지 않는다. 추천으로서 성립하지 않는 것이지 무명이라 나쁜 것이 아니다.

    왜 리뷰 하한과 다른가: 하한 300 을 인기 시드에만 걸어봤더니 `coh_grand_strategy` 가
    0.84 → 0.76 으로 무너졌다. 무명이지만 잘 만든 니치 대전략이 잘려나가고 그 자리를
    인기 있지만 장르가 다른 것(RimWorld·Steel Crew)이 채웠다. 리뷰 수는 "니치인가"와
    "쓰레기인가"를 구분하지 못한다. 이 규칙은 그 구분을 하지 않는다 — 싱글플레이가 되는
    게임은 리뷰가 0이어도 남는다. 저리뷰·롱테일·니치 축은 구조적으로 영향을 받지 않는다.

    `coh_classic_multi` (Garry's Mod / TF2 / L4D2) 의 50~75 구간 실패 칸:
      KUBOOM · Camp Wars · Guns N Stuff 2 · Project Teddy · Shards Online · BLOCKPOST
      전부 멀티 전용 + 리뷰 미보고다. 태그는 시드와 완벽히 겹친다.

    **2차 보강 (k=100 에서 드러남).** 위 규칙만으로는 `coh_fps` 의 실패 8칸이 다 빠져나갔다:
      Multiplayer Shooter FPS · Battle Room · Polygon Bit Battle Royale · Pixel Strike 3D
      · Anti Terrorist Shooting Game · Tactical Vengeance · Chapter Wars · The Last War
    전부 `['싱글 플레이어', '멀티플레이어', 'PvP', '온라인 PvP']` 를 선언한다. PvP 슈터의
    "싱글 플레이어"는 대개 봇이라 살려줄 근거가 못 된다. 그래서 **온라인 PvP 의존**은
    싱글 표기와 무관하게 자른다. 협동은 자르지 않는다 — 친구 3명을 데려오면 성립하지만
    5대5 랭크 로비는 혼자 만들 수 없다. Wolfenstein: Enemy Territory(101) · Task Force(304)
    처럼 리뷰가 보고되는 PvP 게임은 이 규칙에 걸리지 않는다.
    """
    if "categories" not in dataset.columns:
        return df
    cats = dataset.set_index("steam_appid")["categories"]
    known = dataset.set_index("steam_appid")["has_recommendations"]

    def dead(a):
        a = int(a)
        if bool(known.get(a, False)):
            return False
        c = cats.get(a)
        if c is None or isinstance(c, float) or len(c) == 0:
            return False
        names = [x.get("description", "") if isinstance(x, dict) else str(x) for x in c]
        has_solo = any(any(m in n for m in SINGLEPLAYER_MARKERS) for n in names)
        has_pvp = any(any(m in n for m in ONLINE_PVP_MARKERS) for n in names)
        has_coop = any(any(m in n for m in ONLINE_COOP_MARKERS) for n in names)
        # PvP 의존은 싱글 표기가 있어도 살려주지 않는다 — 5대5 로비는 혼자 못 만든다.
        # 협동은 다르다: 친구를 데려오면 성립하므로 그쪽은 싱글 표기가 없어도 안 자른다.
        if has_pvp and not has_coop:
            return True
        return not has_solo

    keep = ~df["steam_appid"].map(dead)
    return df[keep] if keep.any() else df


def consensus_overlap_boost(df: pd.DataFrame, dataset: pd.DataFrame, wanted: set[str],
                            weight: float, topn: int = SEED_TAG_TOPN) -> pd.DataFrame:
    """합의 태그를 **몇 개** 맞혔는지로 재정렬한다. 거르지 않고 순서만 바꾼다.

    `keep_sharing_tags` 는 "하나라도 겹치면 통과"라 임계값을 못 만든다. 실측(k=75):

        coh_grand_strategy (합의태그 12개)
          Europa Universalis V  7개    Supreme Ruler 1936  5개   ← 판정 3
          RimWorld              2개    Steel Crew          2개   ← 판정 1
        coh_classic_multi (합의태그 8개)
          Killing Floor 2       7개                              ← 판정 3
          NYZD                  2개    KUBOOM              3개   ← 판정 1

    임계값을 2로 올려도 안 되는 이유: Witcher 2(coh_arpg 판정 3)도 2개다. 절대 개수가
    아니라 **그 프로필에서 가능한 최대치 대비 비율**이 신호라서, 필터가 아니라 가중치로
    쓴다. 리뷰 하한과 달리 무명 게임을 원천 배제하지 않으므로 저리뷰 축이 안 다친다.
    """
    if not wanted or weight <= 0:
        return df
    tags = dataset.set_index("steam_appid")["tags"]

    def hits(a):
        x = tags.get(int(a))
        if x is None or isinstance(x, float):
            return 0
        got = {t["name"] if isinstance(t, dict) else str(t) for t in x[:topn]}
        return len(wanted & got)

    out = df.copy()
    frac = out["steam_appid"].map(hits) / max(len(wanted), 1)
    score_col = "final_score" if "final_score" in out.columns else out.columns[-1]
    out[score_col] = out[score_col] * (1.0 + frac * weight)
    return out.sort_values(score_col, ascending=False).reset_index(drop=True)


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
    solo_seed_tags: set[str] | None = None,
    consensus_tags: set[str] | None = None,
    consensus_boost: float = 0.0,
    drop_dead_mp: bool = False,
    seen_appids=None,
    series_session_max: int | None = None,
) -> pd.DataFrame:
    """스펙 §5.4 순서: hard filter → 시리즈/퍼블리셔 상한 → 다양성 → Top-N.

    `bucket_offset` 은 인터리빙에 그대로 넘어간다 — 시드가 page_size 보다 많을 때
    페이지마다 버킷 순서를 회전시켜 약한 시드가 굶지 않게 한다.

    `solo_seed_tags` — **시드가 1개일 때만** 넘어오는 희귀 태그 집합. 후보가 그중 하나라도
    가져야 통과한다.

    왜 단일 시드에만 거는가: 시드가 하나면 `top2_mean` 이 `max` 와 수학적으로 같아 집계
    레버가 아예 안 듣는다. 실제로 Salt and Sanctuary 한 개로 랭킹하면 Souls-like +
    Metroidvania 를 **모두** 가진 76개 후보의 순위 중앙값이 526위였다 — Hollow Knight 가
    149위다. 임베딩이 정의적 태그를 못 쓰는 것이고, 태그 IDF 재순위로는 안 고쳐졌다
    (태그가 이미 임베딩 안에 있어 상관이 높다). 하드 조건으로 걸어야 움직인다.

    왜 다시드에는 안 거는가: 시드가 여럿이면 희귀 태그의 합집합이 "시드 간 공통점"이 아니라
    "한 시드의 특이점"이 된다. 무조건 걸었더니 `lowrev_cozy_narrative` 가 0.90 → 0.75 로
    떨어졌다 — Visual Novel 태그가 정의적이라고 오인돼 일본식 연애 VN 이 대거 통과했다.

    35프로필 · 미판정 0 (top2_mean 기준):

      깊이   미적용            단일시드 조건부      바뀐 프로필
      k=10  0.9314 (미달 2)   0.9429 (미달 1)     1개  niche_soulslike_solo 0.50→0.90
      k=20  0.9114 (미달 1)   0.9214 (미달 0)     1개  niche_soulslike_solo 0.55→0.90

    평균 Δ 는 +0.01 로 유의하지 않다 — 35개 중 1개만 변하므로 구조적으로 그렇다. 채택 근거는
    평균이 아니라 **단일 시드 구제 + 다시드 무영향**이다.

    `consensus_tags` — **시드 리뷰 중앙값이 10만 이상인 다시드 프로필에만** 넘어온다.
    시드들이 함께 가진 흔하지 않은 태그를 요구한다.

    왜 인기 시드에만 거는가: 대작 취향의 실패는 저품질 무명작에 몰려 있는데(실패 칸 리뷰
    중앙값 778 vs 성공 2,774) 리뷰 하한으로 자르면 좋은 저리뷰 추천까지 날아간다. 태그
    합의로 자르면 "장르는 겹치는데 경험이 다른" 것만 걸린다.

    무조건 걸면 안 된다 — 전체 적용 시 니치 -0.027 · 롱테일 -0.015 로 상쇄돼 Δ +0.001
    (유의하지 않음)이 된다. 조건부로 좁히면 그 축들이 정의상 영향을 안 받는다:

      k=50 축별 Δ:  니치 0.000 · 롱테일 0.000 · 저리뷰 0.000 · 혼합 +0.002 · 대작 +0.010
      k=50 전체:    0.9194 (미달 2) → 0.9229 (미달 1)

    평균 Δ 는 +0.003 으로 유의하지 않다(17/35 만 대상이고 그중에도 순위가 바뀌는 것은
    소수다). 채택 근거는 **부작용 0 + 대상 축 개선 + 미달 감소**다.
    """
    df = ranked
    if drop_dead_mp:
        df = drop_dead_multiplayer(df, dataset)
    if solo_seed_tags:
        df = keep_sharing_tags(df, dataset, solo_seed_tags)
    if consensus_tags:
        df = keep_sharing_tags(df, dataset, consensus_tags)
        df = consensus_overlap_boost(df, dataset, consensus_tags, consensus_boost)
    if hard_filters:
        df = apply_hard_filters(df, dataset, require_known_reviews=require_known_reviews,
                                min_reviews=min_reviews)
    if series_max:
        prior = series_counts(seen_appids, dataset) if (seen_appids and series_session_max) else None
        df = cap_series(df, dataset, series_max,
                        prior_counts=prior, session_max=series_session_max)
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
