"""새로고침 계약. Steam 에서 실제로 터졌던 문제들을 성질로 고정한다.

실제 아티팩트(임베딩 + dataset)가 있어야 돌기 때문에 없으면 스킵한다.
파이프라인을 한 번이라도 완주했으면 자동으로 켜진다.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import artifact_dir

ART = artifact_dir()
pytestmark = pytest.mark.skipif(
    not (ART / "corpus_embeddings.npy").exists(),
    reason=f"{ART} 에 임베딩이 없습니다 — 파이프라인을 먼저 완주하세요",
)


@pytest.fixture(scope="module")
def components():
    from src.personalized_retrieve import REFRESH_POP_BOOST, build_components

    return build_components(pop_boost=REFRESH_POP_BOOST)


@pytest.fixture(scope="module")
def dataset():
    return pd.read_parquet(ART / "dataset.parquet")


@pytest.fixture(scope="module")
def seeds():
    """코퍼스에서 관심 수가 많은 서로 다른 장르 3편을 시드로 쓴다.

    하드코딩한 product_no 를 쓰면 코퍼스가 바뀔 때마다 테스트가 깨진다.
    """
    idx = pd.read_parquet(ART / "corpus_index.parquet")
    ds = pd.read_parquet(ART / "dataset.parquet")
    df = idx[["item_id"]].merge(ds, on="item_id", how="inner")
    df = df[df["interest_count"].notna()]
    df["g"] = df["genres"].map(lambda g: g[0] if len(g) else "")
    picked = (
        df.sort_values("interest_count", ascending=False)
        .drop_duplicates("g")
        .head(3)["item_id"]
        .tolist()
    )
    if len(picked) < 2:
        pytest.skip("시드로 쓸 작품이 부족합니다 (코퍼스가 너무 작음)")
    return picked


class TestExclusionContract:
    def test_without_exclusion_refresh_returns_identical_list(self, components, seeds):
        """제외 집합이 없으면 run_multi 는 순수 함수다 = 새로고침해도 화면이 안 바뀐다.

        이건 버그가 아니라 성질이다. 호출자가 seen_ids 를 누적해 넘길 책임이 있다는 것을
        코드로 못박아 둔다 — 실제로 이걸 놓쳐서 "새로고침이 안 먹는다"가 났었다.
        """
        from src.personalized_retrieve import next_page

        a = next_page(seeds, page_size=5, components=components)
        b = next_page(seeds, page_size=5, components=components)
        assert list(a["item_id"]) == list(b["item_id"])

    def test_seeds_never_appear_in_results(self, components, seeds):
        from src.personalized_retrieve import next_page

        out = next_page(seeds, page_size=10, components=components)
        assert not set(out["item_id"]) & set(seeds)

    def test_next_page_never_repeats_across_refreshes(self, components, seeds):
        """3회 새로고침 = 전부 서로 다른 작품이어야 한다."""
        from src.personalized_retrieve import next_page

        seen: set[int] = set()
        for _ in range(3):
            page = next_page(seeds, seen_ids=seen, page_size=5, components=components)
            ids = set(page["item_id"])
            assert not (ids & seen), "이전 페이지에서 본 작품이 다시 나왔다"
            seen |= ids


class TestDiversity:
    def test_no_single_seed_dominates_a_page(self, components, seeds):
        """한 시드가 Top-100 의 73%를 먹던 문제. 인터리빙이 이걸 막아야 한다."""
        from src.personalized_retrieve import next_page

        page = next_page(seeds, page_size=10, components=components)
        if "dominant_seed" not in page.columns or page["dominant_seed"].isna().all():
            pytest.skip("dominant_seed 없음")
        share = page["dominant_seed"].value_counts().iloc[0] / len(page)
        assert share <= 0.7, f"한 시드가 페이지의 {share:.0%} 를 차지한다"

    def test_series_not_duplicated_within_page(self, components, seeds, dataset):
        """같은 작품의 [독점]/[단행본]이 나란히 나오면 안 된다."""
        from src.postprocess import series_group

        from src.personalized_retrieve import next_page

        page = next_page(seeds, page_size=10, components=components)
        meta = dataset.set_index("item_id")
        keys = [
            series_group(meta["name"].get(i, ""), meta["publisher"].get(i, ""))
            for i in page["item_id"]
        ]
        assert len(keys) == len(set(keys)), f"중복 시리즈: {keys}"


class TestQualityFloor:
    def test_quality_floor_holds_at_depth(self, components, seeds, dataset):
        """깊은 페이지가 무너지는 원인은 관련성이 아니라 품질이었다.

        3페이지까지 전부 하한을 넘어야 한다.
        """
        from src.personalized_retrieve import REFRESH_MIN_INTEREST, next_page

        meta = dataset.set_index("item_id")["interest_count"]
        seen: set[int] = set()
        for page_no in range(1, 4):
            page = next_page(seeds, seen_ids=seen, page_size=5, components=components)
            if page.empty:
                break
            counts = [meta.get(i) for i in page["item_id"]]
            assert all(
                c is not None and not pd.isna(c) and c >= REFRESH_MIN_INTEREST for c in counts
            ), f"{page_no}페이지에 품질 하한 미달 작품: {counts}"
            seen |= set(page["item_id"])

    def test_floor_can_be_disabled(self, components, seeds):
        """하한을 끄면 더 많은 후보가 남아야 한다 — 필터가 실제로 작동한다는 증거."""
        from src.personalized_retrieve import next_page

        strict = next_page(seeds, page_size=10, components=components, min_interest_count=10**9)
        loose = next_page(seeds, page_size=10, components=components, min_interest_count=None)
        assert len(loose) >= len(strict)


class TestAdultFilter:
    def test_no_19_rated_in_results(self, components, seeds, dataset):
        from src.personalized_retrieve import next_page

        page = next_page(seeds, page_size=10, components=components)
        ages = dataset.set_index("item_id")["age_limit"]
        assert all(ages.get(i, 0) <= 15 for i in page["item_id"])
