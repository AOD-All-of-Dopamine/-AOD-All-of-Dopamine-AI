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
    """판정이 붙은 프로필의 시드는 바꿀 수 없다. 지문이 바뀌면 판정을 다시 해야 한다."""
    assert seeds_fingerprint(ALL_PROFILES) == "cdb4a1c22941"


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
    """손으로 붙인 라벨이 얼마나 틀렸는지를 산출물이 계속 들고 있어야 한다."""
    paired = built[built["n_seeds"] > 1]
    flipped = (paired["profile_type"] != paired["profile_type_declared"]).sum()
    assert flipped > 0
    assert np.isclose(built["seed_cohesion"].max(), 0.683, atol=0.01)
