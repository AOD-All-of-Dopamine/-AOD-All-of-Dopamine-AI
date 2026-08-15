# tests/test_build_p1_profiles.py
"""프로필 정의의 불변식 — 어기면 판정 514쌍이 조용히 무효가 된다.

판정은 `(profile_id, candidate_appid)` 로만 키가 잡히는데 관련성은 시드에 의존한다.
그래서 기존 프로필의 시드를 바꾸면 조인은 성공하는데 의미가 달라진다. 여기서 잡는다.
"""
import numpy as np
import pandas as pd
import pytest

from src.build_p1_profiles import (
    ALL_PROFILES,
    DEV,
    LEGACY_PROFILES,
    LOWREV_PROFILES,
    LOWREV_REVIEW_BAND,
    NICHE_PROFILES,
    NICHE_REVIEW_BAND,
    VAL,
    derive_profile_type,
    seeds_fingerprint,
    validate_niche,
    validate_split,
)


def _p(pid, liked, split, declared="coherent"):
    return {"profile_id": pid, "liked": liked, "split": split, "declared": declared}


# --------------------------------------------------------- dev/val 누수 차단

def test_two_shared_seeds_is_a_leak():
    """3개 중 2개를 공유하면 사실상 같은 프로필이다.

    예전 기준은 `max_shared > 2` 일 때만 실패해서 이것을 통과시켰고, 그 결과
    val(누수) 0.800 / val(깨끗) 0.650 으로 홀드아웃이 오염됐다.
    """
    profiles = [_p("d", [1, 2, 3], DEV), _p("v", [1, 2, 9], VAL)]
    with pytest.raises(ValueError, match="누수"):
        validate_split(profiles)


def test_one_shared_seed_is_allowed():
    stats = validate_split([_p("d", [1, 2, 3], DEV), _p("v", [1, 8, 9], VAL)])
    assert stats["max_shared_dev_val"] == 1


def test_no_shared_seeds_is_ideal():
    stats = validate_split([_p("d", [1, 2, 3], DEV), _p("v", [7, 8, 9], VAL)])
    assert stats["max_shared_dev_val"] == 0
    assert stats["val_seeds_also_in_dev"] == 0


def test_split_must_have_both_sides():
    with pytest.raises(ValueError, match="쏠렸"):
        validate_split([_p("d", [1], DEV), _p("d2", [2], DEV)])


def test_duplicate_profile_id_is_rejected():
    """id 가 겹치면 판정이 엉뚱한 프로필에 붙는다."""
    with pytest.raises(ValueError, match="중복"):
        validate_split([_p("x", [1], DEV), _p("x", [2], VAL)])


def test_real_definition_has_no_leak():
    stats = validate_split(ALL_PROFILES)
    assert stats["max_shared_dev_val"] <= 1
    assert stats["dev"] > 0 and stats["val"] > 0


# ------------------------------------------------- 기존 프로필 시드 불변식

def test_legacy_seeds_are_frozen():
    """판정이 붙은 프로필의 시드는 바꿀 수 없다. 지문이 바뀌면 판정을 다시 해야 한다.

    2026-08-11 에 mix_* 10개를 은퇴시키고 mix2_/longtail_ 14개를 추가했다 — 프로필
    **집합**은 바뀌었지만, 판정이 붙어 있던 31개의 (profile_id, liked) 는 그대로여야 한다.
    """
    from src.build_p1_profiles import LOWREV_PROFILES, NICHE_PROFILES

    judged31 = LEGACY_PROFILES + NICHE_PROFILES + LOWREV_PROFILES
    assert seeds_fingerprint(judged31) == "eeaf06d349a3"
    # 현재 평가 집합(43개)의 지문. 바뀌면 프로필이 편집된 것이다 — 의도한 변경인지 확인할 것.
    # 2026-08-14 감사 후 8개 추가(2시드×3 · MMO/스포츠/JRPG · 6시드 · 20시드)로
    # cebf076832c9 → 888d9744061f. 기존 35개의 시드는 그대로다(위 judged31 지문이 증인).
    assert seeds_fingerprint(ALL_PROFILES) == "888d9744061f"


def test_fingerprint_detects_a_changed_seed():
    a = [_p("x", [1, 2, 3], DEV)]
    b = [_p("x", [1, 2, 4], DEV)]
    assert seeds_fingerprint(a) != seeds_fingerprint(b)


def test_fingerprint_ignores_definition_order():
    a = [_p("x", [1], DEV), _p("y", [2], VAL)]
    assert seeds_fingerprint(a) == seeds_fingerprint(list(reversed(a)))


def test_legacy_profile_count_is_twenty():
    assert len(LEGACY_PROFILES) == 20


# ------------------------------------------------------------------ 니치 축

def test_niche_seeds_stay_in_the_review_band():
    """니치 축을 추가했다고 적어놓고 또 대작이 들어가는 것을 막는다."""
    lo, hi = NICHE_REVIEW_BAND
    ds = pd.DataFrame({"steam_appid": [1, 2], "recommendations_total": [hi + 1, 5000]})
    with pytest.raises(ValueError, match="니치 구간"):
        validate_niche([_p("niche_x", [1, 2], DEV)], ds)


def test_niche_rejects_seed_with_no_reviews():
    ds = pd.DataFrame({"steam_appid": [1], "recommendations_total": [None]})
    with pytest.raises(ValueError, match="니치 구간"):
        validate_niche([_p("niche_x", [1], DEV)], ds)


def test_non_niche_profiles_are_not_band_checked():
    ds = pd.DataFrame({"steam_appid": [1], "recommendations_total": [5_000_000]})
    assert validate_niche([_p("coh_x", [1], DEV)], ds)["seed_reviews_max"] == 5_000_000


def test_niche_profiles_cover_varied_seed_counts():
    """고정 3개 가정을 깨는 것이 이 축의 목적 중 하나다."""
    counts = sorted({len(p["liked"]) for p in NICHE_PROFILES})
    assert counts == [1, 3, 5, 10]


def test_niche_profiles_do_not_share_seeds_with_each_other():
    seen = set()
    for p in NICHE_PROFILES:
        assert not (seen & set(p["liked"])), p["profile_id"]
        seen |= set(p["liked"])


def test_lowrev_seeds_stay_in_their_band():
    """저리뷰 축이 또 상위권으로 채워지는 것을 막는다 — niche 밴드가 구 코퍼스 기준이라
    전체 코퍼스에서는 그것도 상위 4% 였다."""
    lo, hi = LOWREV_REVIEW_BAND
    ds = pd.DataFrame({"steam_appid": [1], "recommendations_total": [hi + 1]})
    with pytest.raises(ValueError, match="lowrev_ 구간"):
        validate_niche([_p("lowrev_x", [1], DEV)], ds)


def test_longtail_seeds_stay_in_their_band():
    """롱테일 축(100~300)도 같은 방식으로 지킨다 — lowrev 가 이름과 달리 중견 히트로
    채워졌던 실수를 반복하지 않는다."""
    ds = pd.DataFrame({"steam_appid": [1], "recommendations_total": [301]})
    with pytest.raises(ValueError, match="longtail_ 구간"):
        validate_niche([_p("longtail_x", [1], DEV)], ds)


def test_lowrev_profiles_do_not_overlap_anything():
    other = {a for p in LEGACY_PROFILES + NICHE_PROFILES for a in p["liked"]}
    seen = set()
    for p in LOWREV_PROFILES:
        assert not (other & set(p["liked"])), p["profile_id"]
        assert not (seen & set(p["liked"])), p["profile_id"]
        seen |= set(p["liked"])


def test_seed_popularity_now_spans_a_real_range():
    """시드 57개가 전부 코퍼스 상위 0.22% 라서 인기도 관련 실험이 원천 불가였다."""
    assert len(LOWREV_PROFILES) >= 5


def test_niche_profiles_do_not_reuse_legacy_seeds():
    legacy = {a for p in LEGACY_PROFILES for a in p["liked"]}
    for p in NICHE_PROFILES:
        assert not (legacy & set(p["liked"])), p["profile_id"]


# ------------------------------------------------ 라벨을 측정으로 대체

def test_profile_type_is_derived_from_measured_cohesion():
    t = derive_profile_type({"a": 0.9, "b": 0.7, "c": 0.3, "d": 0.1})
    assert t == {"a": "coherent", "b": "coherent", "c": "mixed", "d": "mixed"}


def test_single_seed_profile_gets_its_own_label():
    """시드가 하나면 쌍이 없어 응집도를 정의할 수 없다 — mixed 로 뭉뚱그리면 안 된다."""
    t = derive_profile_type({"a": 0.6, "b": float("nan"), "c": 0.4})
    assert t["b"] == "single"


def test_derive_handles_all_nan():
    assert derive_profile_type({"a": float("nan")}) == {"a": "single"}


# ------------------------------------------------------- 산출물 계약

@pytest.fixture(scope="module")
def built():
    from src.config import PROJECT_ROOT

    return pd.read_parquet(PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet")


def test_artifact_has_the_new_columns(built):
    for col in ("n_seeds", "profile_type", "profile_type_declared", "seed_cohesion", "split"):
        assert col in built.columns


def test_artifact_matches_the_definition(built):
    assert len(built) == len(ALL_PROFILES)
    defined = {p["profile_id"]: list(p["liked"]) for p in ALL_PROFILES}
    for _, r in built.iterrows():
        assert list(r["liked_appids"]) == defined[r["profile_id"]]


def test_artifact_seed_counts_are_mixed(built):
    assert set(built["n_seeds"]) >= {1, 3, 5, 10}


def test_single_seed_rows_have_nan_cohesion(built):
    single = built[built["n_seeds"] == 1]
    assert len(single) and single["seed_cohesion"].isna().all()


def test_declared_label_disagrees_with_measurement(built):
    """손으로 붙인 라벨이 얼마나 틀렸는지를 산출물이 계속 들고 있어야 한다.

    응집도의 **절대값**은 단언하지 않는다 — 임베딩을 바꾸면 값이 통째로 달라지고,
    그때마다 테스트를 고치는 것은 아무것도 지켜주지 않는다(태그 도입 때 실제로 깨졌다).
    지켜야 할 것은 "선언 라벨이 실측과 다르다는 사실이 산출물에 남아 있는가"다.
    """
    paired = built[built["n_seeds"] > 1]
    flipped = (paired["profile_type"] != paired["profile_type_declared"]).sum()
    assert flipped > 0
    assert paired["seed_cohesion"].between(0, 1).all()
    assert paired["seed_cohesion"].std() > 0.01, "응집도가 전부 같으면 라벨이 무의미하다"


def test_seed_review_range_is_no_longer_degenerate(built):
    """시드 57개가 전부 코퍼스 상위 0.22% 이던 상태를 되돌리지 않는다."""
    assert built["profile_id"].str.startswith("lowrev_").sum() >= 5


def test_supply_limited_profiles_are_still_evaluated():
    """공급 제약으로 표시하더라도 평가 집합에는 남는다 — 빼면 한계가 안 보인다.

    2026-08-12 현재 비어 있다. niche_puzzle_solo 를 넣었다가 뺐는데, 공급이 모자란 것이
    아니라 리뷰 하한이 후보를 가리고 있었다(하한 개방 시 P@50 0.70 → 0.88).
    """
    from src.build_p1_profiles import SUPPLY_LIMITED

    ids = {p["profile_id"] for p in ALL_PROFILES}
    assert SUPPLY_LIMITED <= ids, SUPPLY_LIMITED - ids


def test_representation_limited_profiles_stay_in_the_set():
    """표현 한계 프로필도 평가에 남긴다 — 빼면 한계가 안 보이고, 표현을 고치면 여기서 먼저 드러난다."""
    from src.build_p1_profiles import REPRESENTATION_LIMITED

    ids = {p["profile_id"] for p in ALL_PROFILES}
    assert REPRESENTATION_LIMITED <= ids, REPRESENTATION_LIMITED - ids


def test_holdout_profiles_are_independent_of_the_dev_set():
    """홀드아웃 시드가 개발 셋과 겹치면 '한 번도 안 쓴 데이터' 라는 전제가 무너진다."""
    from src.build_p1_profiles import HOLDOUT_PROFILES

    dev_seeds = {a for p in ALL_PROFILES for a in p["liked"]}
    for p in HOLDOUT_PROFILES:
        assert not (dev_seeds & set(p["liked"])), p["profile_id"]


def test_holdout_profiles_never_enter_the_dev_set():
    """홀드아웃이 ALL_PROFILES 에 들어가는 순간 홀드아웃이 아니게 된다."""
    from src.build_p1_profiles import HOLDOUT_PROFILES

    dev_ids = {p["profile_id"] for p in ALL_PROFILES}
    assert not ({p["profile_id"] for p in HOLDOUT_PROFILES} & dev_ids)


def test_holdout_seeds_are_frozen():
    """판정이 붙은 뒤 시드가 바뀌면 판정이 조용히 무효가 된다 — 개발 셋과 같은 규칙."""
    from src.build_p1_profiles import HOLDOUT_PROFILES, seeds_fingerprint

    assert seeds_fingerprint(HOLDOUT_PROFILES) == "a80b30c8ffec"


def test_dyn_scenarios_are_frozen():
    """전이 평가의 판정도 (시나리오, 게임) 키로 동결된다 — 시드가 바뀌면 판정이 무효다."""
    from src.build_p1_profiles import DYN_SCENARIOS, seeds_fingerprint

    profiles = [{"profile_id": k, "liked": v["base"] + ([v["add"]] if "add" in v else [])}
                for k, v in sorted(DYN_SCENARIOS.items())]
    assert seeds_fingerprint(profiles) == "1c3e4e5afcf8"


def test_seed_count_distribution_covers_the_user_journey():
    """좋아요 개수 분포가 실사용자 여정을 덮어야 한다.

    2026-08-14 감사: 35개 중 31개가 3시드라 상시 측정 지점이 사실상 하나뿐이었다.
    특히 2시드는 전이 평가에서 확인된 **품질 저점**(0.80~0.90)이자 온보딩 사용자가
    전원 거쳐가는 구간인데 정적 프로필이 0개였다. 다시 3시드로 쏠리면 여기서 잡는다.
    """
    counts = [len(p["liked"]) for p in ALL_PROFILES]
    assert sum(1 for c in counts if c == 1) >= 2, "단일 시드(집계 레버가 안 듣는 구간)"
    assert sum(1 for c in counts if c == 2) >= 3, "2시드 — 확인된 품질 저점"
    assert sum(1 for c in counts if 4 <= c <= 9) >= 2, "활성 사용자 구간"
    assert max(counts) >= 20, "파워 유저 — 인터리빙 버킷 > 페이지 크기"


def test_gap_axes_do_not_reuse_holdout_seeds():
    """공백 축(MMO·스포츠·JRPG)은 홀드아웃과 시드가 겹치면 안 된다.

    홀드아웃은 1회 소진됐으므로, 같은 시드를 재사용하면 '한 번도 안 쓴 데이터'로
    측정한 0.61(MMO)·0.72(스포츠)와 앞으로의 개발 셋 수치를 비교할 수 없게 된다.
    """
    from src.build_p1_profiles import GAP_AXIS_PROFILES, HOLDOUT_PROFILES

    ho = {a for p in HOLDOUT_PROFILES for a in p["liked"]}
    for p in GAP_AXIS_PROFILES:
        assert not (ho & set(p["liked"])), p["profile_id"]
