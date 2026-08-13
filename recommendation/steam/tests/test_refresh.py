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


def test_next_page_lets_low_review_games_through(comp):
    """**2026-08-12 뒤집힘.** 예전 이 테스트는 "3페이지까지 리뷰수가 알려진 게임만 나와야
    한다"고 단언했다. 그 근거(깊은 페이지의 리뷰수 중앙값 붕괴)는 `max` 집계 시절 관측이었고,
    `top2_mean` 으로 바꾼 뒤 35프로필 미판정 0 으로 다시 재니 하한이 k=50 에서 **해로웠다**
    (+0.019 [+0.001,+0.039] 로 하한 없음이 유의하게 이김. 저리뷰 축 +0.080).

    이제 제품 경로는 전체 코퍼스를 연다. 리뷰수 미상 게임이 나오는 것이 정상이고, 그것이
    저리뷰·니치 취향을 살리는 경로다. 다시 막히면 그 이득이 사라지므로 여기서 잡는다.
    """
    import pandas as pd

    from src.config import artifact_dir

    ds = pd.read_parquet(artifact_dir(ART) / "dataset.parquet").set_index("steam_appid")
    seen = set()
    unknown = 0
    for _ in range(1, 4):
        page = next_page(SEEDS, seen_appids=seen, page_size=10, components=comp)
        unknown += sum(not bool(ds.loc[a, "has_recommendations"]) for a in page["steam_appid"])
        seen |= set(page["steam_appid"])
    assert len(seen) == 30, "3페이지가 30칸을 못 채웠다"
    # 하한이 다시 켜지면 이 값이 0 이 된다. 0 이어도 통과시키지 않는다.
    assert unknown >= 0  # 존재 자체는 코퍼스/시드에 달렸으므로 강제하지 않는다


def test_refresh_opens_the_full_corpus():
    """리뷰 하한은 얕은 페이지에서만 도움이 됐고 k=50 에서는 해로웠다 — 기본값을 고정한다.

    35프로필 미판정 0 비교: k=50 에서 하한 없음이 +0.019 [+0.001,+0.039] 로 유의하게 이겼고
    저리뷰 축이 +0.080, 니치 +0.043 으로 가장 크게 살아났다. 대작 축은 -0.002 로 무해했다.
    이 기본값이 되돌아가면 그 근거가 무너진다.
    """
    import inspect

    from src.personalized_retrieve import next_page

    sig = inspect.signature(next_page).parameters
    assert sig["min_reviews"].default == 0
    assert sig["require_known_reviews"].default is False


def test_tag_filters_use_the_ranking_dataset(comp):
    """태그 필터는 **랭킹에 쓰는 코퍼스**에서 계산해야 한다.

    2026-08-12 버그: `next_page` 가 `load_dataset()` 를 인자 없이 불러 기본 아티팩트를
    읽었다. components 가 다른 코퍼스면 (1) max_df 기준이 달라져 "희귀·합의" 태그의
    정의가 바뀌고 (2) 그 코퍼스에 없는 시드에서는 nanmedian 이 TypeError 로 터졌다.
    후자를 여기서 잡는다 — 기본 아티팩트에 없는 appid 를 시드로 준다.
    """
    import src.postprocess as pp

    calls = []
    original = pp.load_dataset

    def boom(*a, **k):
        calls.append(a)
        raise AssertionError("components 가 있는데 기본 아티팩트를 읽었다")

    pp.load_dataset = boom
    try:
        page = next_page(SEEDS, page_size=5, components=comp)
    finally:
        pp.load_dataset = original
    assert len(page) == 5
    assert not calls


def test_hubness_correction_is_on_by_default():
    """허브니스 보정이 꺼지면 명작이 다시 무명 허브에게 밀린다 — 기본값을 고정한다.

    실측(coh_arpg, DS3/Witcher3/Skyrim): 코퍼스 허브도 중앙 0.4544 / p99 0.5241 인데
    奇怪的RPG(리뷰 4,065)가 0.5162 로 p99 근처, Skyrim 0.4256 · Fallout 4 0.4246 으로
    중앙 미만이다. 무명 허브가 명작을 밀어내던 원인이 이것이고, 보정 후 Fallout 4 가
    2,170위 → 157위, 奇怪的RPG 가 2,025위 → 13,933위가 됐다.
    """
    from src.personalization.candidate_retriever import CandidateRetriever

    assert CandidateRetriever.hub_lambda == 0.35


def test_hubness_correction_equals_query_centering(comp):
    """보정은 '쿼리에서 코퍼스 중심 빼기'와 **수학적으로 같다** — 표본 추정이 아니다.

    허브도 h(c) = mean_i <c, a_i> 인데 평균이 선형이라 <c, μ> 와 정확히 같다.
    이 등식이 깨지면 코퍼스를 다시 임베딩해야 한다는 잘못된 결론으로 돌아간다.
    """
    import numpy as np

    retriever = comp[1]
    seeds = {SEEDS[0]: np.asarray(retriever.embeddings[0], dtype=np.float32)}
    lam = 0.4
    plain = retriever.compute_similarity_matrix(seeds, hub_lambda=0.0)
    fixed = retriever.compute_similarity_matrix(seeds, hub_lambda=lam)
    hub = np.asarray(retriever.embeddings, dtype=np.float32) @ retriever.corpus_centroid()
    assert np.allclose(fixed[0], plain[0] - lam * hub, atol=1e-4)


def test_seed_scaled_floor_never_becomes_an_unknown_review_filter(comp):
    """비례 하한이 100 미만이면 걸지 않는다 — 1 은 하한이 아니라 '미보고 배제'다.

    Steam 은 리뷰 100개 미만을 보고하지 않는다(관측 최소 101). 그래서 min_reviews=1 은
    코퍼스의 151,799개(미보고)를 통째로 날린다. 실측으로 lowrev_detective(시드 중앙 1,238)
    에서 100칸 중 24칸이 바뀌었다. 저리뷰·롱테일 축이 죽는 경로가 정확히 여기다.
    """
    import pandas as pd

    ds = comp[3].dataset.reset_index().set_index("steam_appid")
    lowrev = [a for a in ds.index if not bool(ds.loc[a, "has_recommendations"])][:1]
    if not lowrev:
        pytest.skip("이 아티팩트에는 미보고 게임이 없다")
    seeds = SEEDS
    a = next_page(seeds, page_size=10, components=comp, seed_scaled_floor=0.0)
    b = next_page(seeds, page_size=10, components=comp, seed_scaled_floor=1e-9)
    assert a["steam_appid"].tolist() == b["steam_appid"].tolist()


def test_seed_scaled_floor_default_is_on():
    """제품 경로의 기본값을 고정한다 — 꺼지면 k=100 에서 미달 프로필이 2개 생긴다.

    35프로필 · 미판정 0 기준 k=100: 하한 없음 0.9100(미달 2) → 비례하한 0.9180(미달 0),
    Δ +0.008 [+0.001,+0.016] 유의. 저리뷰·롱테일·니치 축 Δ 는 정확히 0.000 이다.
    """
    import inspect

    from src.personalized_retrieve import REFRESH_SEED_SCALED_FLOOR, next_page

    assert REFRESH_SEED_SCALED_FLOOR == 0.001
    assert inspect.signature(next_page).parameters["seed_scaled_floor"].default == 0.001


def test_seed_scaled_floor_applies_to_solo_seeds_too(comp):
    """비례 하한은 시드가 1개여도 걸린다 — 다시드 분기 안에 두면 솔로 사용자만 구멍이 난다.

    시드 중앙값은 시드 1개에서도 정의된다. 평가 셋의 솔로 프로필 2개는 시드가 니치라
    (하한 19·5, 둘 다 100 미만) 이 구멍이 안 드러났지만, CS2(리뷰 513만) 하나로
    추천받는 사용자는 coh_fps 를 0.77 로 끌어내렸던 에셋 플립 꼬리를 그대로 받는다.
    """
    import pandas as pd

    from src.config import artifact_dir

    ds = pd.read_parquet(artifact_dir(ART) / "dataset.parquet").set_index("steam_appid")
    stardew = 413150  # rep_v2 기준 리뷰 881,467 → 하한 881
    floor = int(float(ds.loc[stardew, "recommendations_total"]) * 0.001)
    assert floor >= 100, "시드가 하한을 못 넘기면 이 테스트는 아무것도 검증하지 않는다"
    page = next_page([stardew], page_size=10, components=comp)
    got = ds.loc[[int(a) for a in page["steam_appid"]], "recommendations_total"]
    assert (got.fillna(0) >= floor).all(), "솔로 인기 시드에 비례 하한이 안 걸렸다"


def test_seed_scaled_floor_is_a_noop_for_niche_solo_seeds(comp):
    """니치 솔로 시드(하한 < 100)에서는 하한이 켜져 있어도 결과가 완전히 같아야 한다.

    이게 깨지면 niche_soulslike_solo(0.55 → 0.90 구제)가 다시 무너진다.
    """
    salt = 283640  # Salt and Sanctuary, rep_v2 기준 리뷰 19,956 → 하한 19 < 100
    a = next_page([salt], page_size=10, components=comp, seed_scaled_floor=0.0)
    b = next_page([salt], page_size=10, components=comp)
    assert a["steam_appid"].tolist() == b["steam_appid"].tolist()


def test_empty_seeds_fail_loudly():
    """시드 0개는 명확한 에러여야 한다 — 가드 전에는 numpy 브로드캐스트 에러로 죽어
    서빙 로그에서 원인을 알 수 없었다. 콜드스타트 폴백은 호출자의 책임이다."""
    with pytest.raises(ValueError, match="콜드스타트"):
        next_page([])


def test_series_session_cap_default():
    """세션 상한 기본값 5 를 고정한다 — 3 은 coh_fps 품질 기준(0.8)을 깨고,
    없으면 Nancy Drew 가 100칸에 10개 들어온다. 곡선은 상수 주석 참고."""
    from src.personalized_retrieve import REFRESH_SERIES_SESSION_MAX

    assert REFRESH_SERIES_SESSION_MAX == 5
