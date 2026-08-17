# src/postprocess.py
"""랭킹 **뒤**에 붙는 후처리.

Steam 판의 구조(hard filter → 시리즈 상한 → 다양성 인터리빙 → Top-N)를 그대로 가져오고
술어만 웹소설로 바꿨다. 설계 스펙 §5.4 의 순서를 따른다.

**왜 필요한가** — Steam 20개 프로필 실측에서 한 시드가 Top-100 의 73.4%(최악 99/100)를
차지했다. 10개씩 새로고침하는 제품에서 이건 "2페이지가 1페이지랑 똑같다"로 나타난다.
새로고침 제품의 병목은 정확도가 아니라 변화량이다. 이 성질은 도메인과 무관하다.
"""
import re

import pandas as pd

from src.config import artifact_dir

# 성인물은 크롤 단계에서 이미 빠지지만(19금은 상세 진입 자체가 막힘) 목록 경로가 바뀌면
# 샐 수 있어 여기서도 막는다. Steam 의 ADULT_GENRES 자리.
MAX_AGE_LIMIT = 15  # 19세 이용가 제외

# 한국어 웹소설 시리즈 마커. Steam 의 로마숫자/edition/GOTY 규칙은 이 도메인에 없다.
#   '나 혼자만 레벨업 2부' · '전지적 독자 시점 외전' · '재벌집 막내아들 [단행본]'
# 판본 표기([독점]/[단행본]/개정판)는 같은 작품의 다른 상품이라 반드시 묶어야 한다.
_SERIES_BRACKET = re.compile(r"\[[^\]]*\]")
_SERIES_NOISE = re.compile(
    r"(\d+\s*부(?:작)?|시즌\s*\d+|외전|번외|후속편|속편|개정판|완전판|합본|세트|"
    r"단행본|독점|리마스터|재편집|\d+\s*권|전편|후편)"
)
_NON_WORD = re.compile(r"[^0-9a-z가-힣 ]")


def series_key(name: str) -> str:
    """시리즈 근사. 판본/부수 표기를 걷어내고 앞 2어절을 쓴다.

    '나 혼자만 레벨업 2부'     → '나 혼자만'
    '전지적 독자 시점 [단행본]' → '전지적 독자'
    Steam 판과 같은 한계가 있다 — 제목이 전혀 다른 같은 시리즈는 못 잡는다.
    그래서 `series_group` 에서 출판사를 함께 쓴다.
    """
    s = _SERIES_BRACKET.sub(" ", str(name)).lower()
    s = _SERIES_NOISE.sub(" ", s)
    s = _NON_WORD.sub(" ", s)
    toks = s.split()
    return " ".join(toks[:2]) if toks else str(name).lower()


def series_group(name: str, publisher: str = "") -> str:
    """시리즈 식별자. 출판사가 있으면 `출판사|이름앞부분` 으로 더 정확해진다.

    웹소설은 Steam 보다 출판사 신호가 강하다 — 같은 작품의 [독점]/[단행본]/개정판이
    같은 출판사에서 나오기 때문이다. 다만 대형 출판사가 다작을 하므로 이름 첫 어절을
    함께 남겨서 한 출판사 전체가 한 그룹으로 뭉치는 것을 막는다.
    """
    key = series_key(name)
    pub = str(publisher or "").strip().lower()
    if not pub:
        return key
    head = key.split()[0] if key.split() else key
    return f"{pub}|{head}"


def apply_hard_filters(
    ranked: pd.DataFrame,
    dataset: pd.DataFrame,
    drop_adult: bool = True,
    min_interest_count: int | None = None,
) -> pd.DataFrame:
    """추천으로 내보내면 안 되는 것을 제거한다.

    `min_interest_count` — 품질 하한. Steam 은 리뷰 수 결측이 "리뷰가 거의 없음"이라는
    깨끗한 이진 신호였지만(있는 값은 전부 100 이상), 네이버 시리즈 관심 수는 **연속값**이라
    임계값으로 다뤄야 한다. 관심 2짜리 작품이 평점 10.0 을 달고 있는 게 흔하다.

    Steam 에서 깊은 페이지가 무너진 원인이 관련성이 아니라 품질이었다(유사도는 0.75→0.71로
    거의 안 변하는데 리뷰 수 중앙값이 2,831→434 로 붕괴). 임계값은 `dataset_profile.json`
    의 interest_count 분포를 보고 정한다.
    """
    df = ranked.copy()
    if df.empty:
        return df.reset_index(drop=True)
    meta = dataset.set_index("item_id") if "item_id" in dataset.columns else dataset
    keep = pd.Series(True, index=df.index)

    if drop_adult and "age_limit" in meta.columns:
        ages = df["item_id"].map(lambda i: meta["age_limit"].get(i, 0))
        keep &= ages.fillna(0) <= MAX_AGE_LIMIT

    if min_interest_count is not None and "interest_count" in meta.columns:
        # 결측(관심 수 미표기)도 하한 미달로 본다 — Steam 과 같은 취급이다.
        counts = df["item_id"].map(lambda i: meta["interest_count"].get(i))
        keep &= counts.fillna(0) >= min_interest_count

    return df[keep].reset_index(drop=True)


def cap_series(df: pd.DataFrame, dataset: pd.DataFrame, series_max: int = 1) -> pd.DataFrame:
    """같은 시리즈를 최대 series_max 개만 남긴다.

    인터리빙은 각 시드의 최근접을 끌어올리는데, 웹소설에서 시드의 최근접은 거의 항상
    같은 작품의 다른 판본이거나 같은 작가의 다음 작품이다. 그래서 인터리빙과 **함께** 건다.
    """
    if df.empty:
        return df.reset_index(drop=True)
    meta = dataset.set_index("item_id") if "item_id" in dataset.columns else dataset
    has_pub = "publisher" in meta.columns
    keys = df["item_id"].map(
        lambda i: series_group(
            meta["name"].get(i, ""),
            meta["publisher"].get(i, "") if has_pub else "",
        )
    )
    seen: dict[str, int] = {}
    keep = []
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
        keep.append(seen[k] <= series_max)
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def cap_author(df: pd.DataFrame, dataset: pd.DataFrame, author_max: int = 2) -> pd.DataFrame:
    """같은 작가를 최대 author_max 개만 남긴다.

    시리즈 상한만으로는 부족하다 — 웹소설 작가는 제목이 전혀 다른 작품을 여러 개 내고,
    같은 작가의 문체/소재가 임베딩상 매우 가까워서 한 작가가 페이지를 점령한다.
    Steam 에는 없던 축인데(게임은 개발사 다작이 덜 유사하다) 이 도메인에서는 필요하다.
    """
    if df.empty or "author" not in dataset.columns:
        return df.reset_index(drop=True)
    meta = dataset.set_index("item_id") if "item_id" in dataset.columns else dataset
    authors = df["item_id"].map(lambda i: str(meta["author"].get(i, "") or "").strip())
    seen: dict[str, int] = {}
    keep = []
    for a in authors:
        if not a:  # 작가 미상은 묶지 않는다
            keep.append(True)
            continue
        seen[a] = seen.get(a, 0) + 1
        keep.append(seen[a] <= author_max)
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def interleave_by_seed(ranked: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """dominant_seed 버킷을 라운드로빈으로 배치한다.

    **전역 상한이 아니라 라운드로빈이어야 한다.** 상한만 걸면 "1~50위는 로판, 51~100위는
    무협"이 되어 1~5페이지가 전부 로판이다. 라운드로빈은 매 페이지에 시드가 섞이게 한다.

    **입력 순서가 우선순위다 — 이 함수는 재정렬하지 않고 보존한다.** 호출자가 넘기는
    `ranked` 가 원하는 순서로 정렬돼 있어야 한다. 서빙에서는 `ranker.rank()` 가
    final_score 내림차순으로 정렬하고 그 뒤 후처리는 걸러내기만 하므로, 결과적으로
    "가장 강한 시드가 1위 자리를 갖고 버킷 안도 점수순"이 성립한다.
    빈 버킷은 자동으로 건너뛰므로 코퍼스에 이웃이 없는 시드가 있어도 목록이 짧아지지 않는다.
    """
    if ranked.empty or "dominant_seed" not in ranked.columns or ranked["dominant_seed"].isna().all():
        out = ranked.head(top_n).reset_index(drop=True)
        if len(out):
            out["rank"] = range(1, len(out) + 1)
        return out

    # `sort=False` 가 이미 (a) 버킷을 첫 등장 순서로, (b) 버킷 안을 원래 순서로 유지한다.
    # 입력이 final_score 내림차순이면 추가 정렬은 중복이고, 아니면 앞 단계가 세운 순서를
    # 통째로 버린다. Steam 에서 이 지뢰가 실제로 터졌다 — 순서를 세우는 후처리 실험
    # 세 개가 조용히 무효가 됐고, 측정값이 소수점까지 동일해서 겨우 발견했다.
    buckets = [g for _, g in ranked.groupby("dominant_seed", sort=False)]

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
    author_max: int = 2,
    hard_filters: bool = True,
    min_interest_count: int | None = None,
) -> pd.DataFrame:
    """스펙 §5.4 순서: hard filter → 시리즈/작가 상한 → 다양성 → Top-N."""
    df = ranked
    if hard_filters:
        df = apply_hard_filters(df, dataset, min_interest_count=min_interest_count)
    if series_max:
        df = cap_series(df, dataset, series_max)
    if author_max:
        df = cap_author(df, dataset, author_max)
    if seed_interleave:
        df = interleave_by_seed(df, top_n)
    else:
        df = df.head(top_n).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    return df


def load_dataset(artifacts=None) -> pd.DataFrame:
    return pd.read_parquet(artifact_dir(artifacts) / "dataset.parquet")
