# src/postprocess.py
"""랭킹 **뒤**에 붙는 후처리.

Steam 판의 구조(hard filter → 시리즈 상한 → 다양성 인터리빙 → Top-N)를 그대로 가져오고
술어만 웹소설로 바꿨다. 설계 스펙 §5.4 의 순서를 따른다.

**왜 필요한가** — Steam 20개 프로필 실측에서 한 시드가 Top-100 의 73.4%(최악 99/100)를
차지했다. 10개씩 새로고침하는 제품에서 이건 "2페이지가 1페이지랑 똑같다"로 나타난다.
새로고침 제품의 병목은 정확도가 아니라 변화량이다. 이 성질은 도메인과 무관하다.
"""
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import artifact_dir, PRODUCTION

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


#: 이름 → 키 변환은 순수 함수이고 이름은 코퍼스(29,494편)로 한정된다. 후처리가 호출마다
#: 후보 수천 개에 같은 정규식을 세 번씩 다시 돌리던 것을 메모이즈로 없앤다 (2026-09-19 서빙 지연).
#: **전제: 위 정규식 상수(`_SERIES_BRACKET`·`_SERIES_NOISE`·`_NON_WORD`)를 런타임에 바꾸지
#: 않는다.** 바꾸려면 `_series_key.cache_clear()` 를 함께 불러야 한다.
_NAME_CACHE_MAX = 1 << 17


@lru_cache(maxsize=_NAME_CACHE_MAX)
def _series_key(s: str) -> str:
    x = _SERIES_BRACKET.sub(" ", s).lower()
    x = _SERIES_NOISE.sub(" ", x)
    x = _NON_WORD.sub(" ", x)
    toks = x.split()
    return " ".join(toks[:2]) if toks else s.lower()


def series_key(name: str) -> str:
    """시리즈 근사. 판본/부수 표기를 걷어내고 앞 2어절을 쓴다.

    '나 혼자만 레벨업 2부'     → '나 혼자만'
    '전지적 독자 시점 [단행본]' → '전지적 독자'
    Steam 판과 같은 한계가 있다 — 제목이 전혀 다른 같은 시리즈는 못 잡는다.
    그래서 `series_group` 에서 출판사를 함께 쓴다.
    """
    return _series_key(str(name))


def meta_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    """`item_id` 를 인덱스로 가진 메타 프레임. 이미 인덱스면 그대로 돌려준다.

    후처리 단계마다 따로 적혀 있던 `set_index("item_id")` 를 한 함수로 모은다. 서빙은
    랭커가 이미 인덱싱해 둔 `ranker.dataset` 을 그대로 넘겨 그 반복 자체를 없앤다.
    **읽기 전용으로만 쓴다** — 이 프레임을 고치는 후처리는 없다. 구 호출부(`item_id` 가
    열인 평탄한 프레임, 예: `wn_eval.Engine`)는 예전과 똑같이 돈다.
    """
    return dataset.set_index("item_id") if "item_id" in dataset.columns else dataset


def _lookup(meta: pd.DataFrame, cols: tuple[str, ...], ids, default=""):
    """`[meta[c].get(i, default) for i in ids]` 를 열마다 한 번에 구한다.

    후처리 지연의 대부분이 여기였다(2026-09-19 실측, k=50 에서 후보 7,500행):
    행마다 `Series.get` 을 부르거나(`cap_series`·`cap_author`) `meta.loc[i]` 로 18열짜리
    혼합 dtype 행을 통째로 조립했다(`drop_seed_series`). 값은 그대로다 — 같은 열의 같은
    원소를 위치로 꺼내 올 뿐이다.

    돌려주는 `known` 은 id 가 인덱스에 있었는지다. 예전에 모르는 id 에서 KeyError 를 내던
    호출부가 그 조건을 그대로 판단할 수 있게 한다.

    인덱스가 중복인 경우는 정상 코퍼스에 없다(랭커의 `map` 이 먼저 터진다). 그래도 값이
    조용히 달라지지 않게, 그때는 예전처럼 행마다 `Series.get` 을 부른다.
    """
    ids = [int(i) for i in ids]
    if meta.index.is_unique:
        pos = meta.index.get_indexer(ids)
        known = pos >= 0
        safe = np.where(known, pos, 0)
        out = []
        for c in cols:
            if c not in meta.columns:
                out.append([default] * len(ids))
                continue
            v = meta[c].take(safe).tolist()
            out.append(v if known.all() else [x if k else default for x, k in zip(v, known)])
        return known, out
    known = np.asarray(pd.Index(ids).isin(meta.index), dtype=bool)
    return known, [[default] * len(ids) if c not in meta.columns
                   else [meta[c].get(i, default) for i in ids] for c in cols]


def series_group(name: str, publisher: str = "", author: str = "", by: str | None = None) -> str:
    """시리즈 식별자.

    `by="publisher"` (현행) — `출판사|이름 첫 어절`. 전제는 "같은 작품의 [독점]/[단행본]/개정판이
    같은 출판사에서 나온다"였다. 대형 출판사 다작이 한 그룹으로 뭉치지 않게 첫 어절을 남긴다.

    `by="author"` (W-5 후보) — `작가|series_key`. **wn_v6 에서 위 전제가 깨진다**: 같은 작품 판본이
    `제이플미디어`/`제이플러스` 처럼 출판사 표기가 달라 서로 다른 시리즈로 갈라지고,
    시드와 똑같은 작품이 추천되거나 한 목록에 두 번 나온다(52프로필 top-10 에코 11 · 중복 5).
    작가가 없으면 출판사 키로 돌아간다.
    """
    by = PRODUCTION["series_by"] if by is None else by
    key = series_key(name)
    au = str(author or "").strip().lower()
    if by == "author" and au:
        return f"a:{au}|{key}"
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
    meta = meta_frame(dataset)
    keep = pd.Series(True, index=df.index)

    if drop_adult and "age_limit" in meta.columns:
        _, (a,) = _lookup(meta, ("age_limit",), df["item_id"], default=0)
        ages = pd.Series(a, index=df.index)
        keep &= ages.fillna(0) <= MAX_AGE_LIMIT

    if min_interest_count is not None and "interest_count" in meta.columns:
        # 결측(관심 수 미표기)도 하한 미달로 본다 — Steam 과 같은 취급이다.
        # 열 꺼내기는 람다 **밖**에서 한 번만 한다 — 안에 두면 후보 수만큼
        # DataFrame.__getitem__ 이 돈다. 여기는 관심 수가 nullable(Int64)이라
        # 값 경로를 바꾸지 않으려고 map 은 그대로 둔다.
        count_col = meta["interest_count"]
        counts = df["item_id"].map(lambda i: count_col.get(i))
        keep &= counts.fillna(0) >= min_interest_count

    return df[keep].reset_index(drop=True)


def drop_seed_series(df: pd.DataFrame, dataset: pd.DataFrame, seed_ids, series_by: str | None = None) -> pd.DataFrame:
    """시드 작품의 외전·후속부를 추천에서 뺀다.

    2026-08-22 사용자 피드백(Steam 문명 VI→VII 사례와 동일 규칙): 좋아한 작품의
    외전/2부 재추천은 발견이 아니라 중복이다 — 독자는 본편 페이지에서 이미 안다.
    실측: 52프로필 × k=50 에서 1건(`제가 죽었다고 각성하시다니요 외전`)뿐이지만
    규칙으로 막아 둔다.
    """
    if df.empty or not seed_ids:
        return df
    meta = meta_frame(dataset)

    def grp(i):
        r = meta.loc[int(i)]
        return series_group(str(r.get("name", "")), str(r.get("publisher", "")),
                            str(r.get("author", "") or ""), by=series_by)

    def groups(ids) -> list[str]:
        """위 `grp` 을 목록 전체에 대해 한 번에. 식은 그대로고 조회만 열 단위로 바뀐다."""
        known, (nm, pb, au) = _lookup(meta, ("name", "publisher", "author"), ids)
        if not known.all():   # 모르는 id 는 예전처럼 `meta.loc` 가 KeyError 를 내게 둔다
            return [grp(i) for i in ids]
        return [series_group(str(n), str(p), str(a or ""), by=series_by)
                for n, p, a in zip(nm, pb, au)]

    seed_groups = set(groups(list(seed_ids)))
    # 시드 자체는 상위에서 이미 제외되므로 여기서는 시리즈 동료만 거른다
    keep = [g not in seed_groups for g in groups(df["item_id"])]
    return df[pd.Series(keep, index=df.index)].reset_index(drop=True)


def cap_series(df: pd.DataFrame, dataset: pd.DataFrame, series_max: int = 1, series_by: str | None = None) -> pd.DataFrame:
    """같은 시리즈를 최대 series_max 개만 남긴다.

    인터리빙은 각 시드의 최근접을 끌어올리는데, 웹소설에서 시드의 최근접은 거의 항상
    같은 작품의 다른 판본이거나 같은 작가의 다음 작품이다. 그래서 인터리빙과 **함께** 건다.
    """
    if df.empty:
        return df.reset_index(drop=True)
    meta = meta_frame(dataset)
    # 세 열을 행마다 꺼내던 것(`meta["name"].get(i, "")` × 후보 수)을 한 번에 꺼낸다.
    # 없는 열·모르는 id 는 `_lookup` 이 예전과 같은 기본값("")을 준다.
    _, (names, pubs, authors) = _lookup(meta, ("name", "publisher", "author"), df["item_id"])
    keys = [series_group(n, p, a, by=series_by) for n, p, a in zip(names, pubs, authors)]
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
    meta = meta_frame(dataset)
    _, (au,) = _lookup(meta, ("author",), df["item_id"])
    authors = [str(a or "").strip() for a in au]
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
    series_by: str | None = None,
) -> pd.DataFrame:
    """스펙 §5.4 순서: hard filter → 시리즈/작가 상한 → 다양성 → Top-N."""
    df = ranked
    if hard_filters:
        df = apply_hard_filters(df, dataset, min_interest_count=min_interest_count)
    if series_max:
        df = cap_series(df, dataset, series_max, series_by=series_by)
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
