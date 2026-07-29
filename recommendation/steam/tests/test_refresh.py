# tests/test_refresh.py
"""새로고침 계약 — 재조회할 때마다 새 콘텐츠가 나와야 한다.

이 파일이 지키는 것: `run_multi` 는 순수 함수라 제외 집합이 없으면 같은 목록을 낸다.
새로고침이 재조회 방식인 제품에서 이건 "화면이 안 바뀐다"로 나타난다.
"""
import pandas as pd
import pytest

from src.personalized_retrieve import next_page, run_multi

SEEDS = [413150, 105600, 648800]  # Stardew Valley / Terraria / Raft
ART = "artifacts/rep_v2"


@pytest.fixture(scope="module")
def comp():
    from src.personalized_retrieve import build_components

    return build_components(0.03, artifacts=ART)


def test_without_exclusion_refresh_returns_identical_list(comp):
    """제외 집합이 없으면 새로고침해도 똑같다 — 이게 고쳐야 했던 버그다."""
    a = run_multi(SEEDS, strategies=["max"], top_n=10, components=comp)["max"]
    b = run_multi(SEEDS, strategies=["max"], top_n=10, components=comp)["max"]
    assert a["steam_appid"].tolist() == b["steam_appid"].tolist()


def test_exclusion_produces_new_items(comp):
    a = run_multi(SEEDS, strategies=["max"], top_n=10, components=comp)["max"]
    b = run_multi(
        SEEDS, strategies=["max"], top_n=10, components=comp,
        exclude_appids=set(a["steam_appid"]),
    )["max"]
    assert not (set(a["steam_appid"]) & set(b["steam_appid"]))


def test_seeds_are_always_excluded_even_if_not_passed(comp):
    r = run_multi(SEEDS, strategies=["max"], top_n=50, components=comp)["max"]
    assert not (set(SEEDS) & set(r["steam_appid"]))


def test_next_page_never_repeats_across_refreshes(comp):
    """새로고침 5번 = 서로 겹치지 않는 50개."""
    seen, pages = set(), []
    for _ in range(5):
        page = next_page(SEEDS, seen_appids=seen, page_size=10, components=comp)
        assert len(page) == 10
        assert not (set(page["steam_appid"]) & seen)
        seen |= set(page["steam_appid"])
        pages.append(page)
    assert len(seen) == 50


def test_next_page_keeps_seed_diversity_at_depth(comp):
    """깊이 들어가도 한 시드가 페이지를 독식하면 안 된다."""
    seen = set()
    for _ in range(3):
        page = next_page(SEEDS, seen_appids=seen, page_size=10, components=comp)
        share = page["dominant_seed"].value_counts().iloc[0] / len(page)
        assert share <= 0.6, f"한 시드가 {share:.0%} 차지"
        seen |= set(page["steam_appid"])
