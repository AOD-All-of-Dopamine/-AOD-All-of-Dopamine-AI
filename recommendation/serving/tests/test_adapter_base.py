import pandas as pd, pytest
from aod_serving.engine.adapters.base import AdapterItem, EngineAdapter, IntKeyMixin, frame_items
from aod_serving.engine.overrides import EffectiveConfig

CFG = EffectiveConfig(production={}, postprocess={}, hash="x" * 12, approved=False, verdict=None)


class Fake(IntKeyMixin, EngineAdapter):
    platform = "webtoon"; factor_schema = "webtoon.v1"

    def __init__(self, n_return=3):
        super().__init__("/nowhere", CFG); self._known = {1, 2, 3, 4, 5}; self.calls = []; self.n_return = n_return

    def load(self): pass
    def first_key(self): return "1"

    def _next_page(self, **kw):
        self.calls.append(kw)
        return pd.DataFrame({"item_id": [10, 11, 12][: self.n_return], "seed_similarity": [.9, .8, .7][: self.n_return],
                             "final_score": [.95, .85, .75][: self.n_return], "dominant_seed": [1, 1, 2][: self.n_return]})

    def _items(self, frame):
        return frame_items(frame, key_col="item_id", to_key=str, dominant_to_key=str, factor_cols={})


def test_unknown_and_malformed_seeds_are_dropped_not_raised():
    a = Fake(); r = a.recommend(k=3, seeds=["1", "999", "abc", "２", "-1", "2"])
    assert a.calls[0]["seeds"] == [1, 2] and r.dropped_seeds == ["999", "abc", "２", "-1"]


def test_duplicate_seeds_keep_first_position():
    a = Fake(); a.recommend(k=3, seeds=["2", "1", "2"]); assert a.calls[0]["seeds"] == [2, 1]


def test_seed_dedup_happens_on_parsed_key_not_raw_string():
    """"2"·"002" 는 같은 코퍼스 키다 — 원문 문자열이 달라도 엔진에는 한 번만 들어가야 한다."""
    a = Fake(); r = a.recommend(k=3, seeds=["2", "002"])
    assert a.calls[0]["seeds"] == [2] and r.dropped_seeds == []


def test_unknown_seed_dedup_is_on_raw_string():
    a = Fake(); r = a.recommend(k=3, seeds=["abc", "abc", "xyz"])
    assert r.dropped_seeds == ["abc", "xyz"] and a.calls == []


def test_absurdly_long_numeric_seed_is_dropped_not_raised():
    """`int(key)` 는 아주 긴 숫자 문자열에서 느려지거나(ValueError) 예외를 던질 수 있다 — 길이로 먼저 거른다."""
    a = Fake(); r = a.recommend(k=3, seeds=["9" * 5000, "1"])
    assert r.dropped_seeds == ["9" * 5000] and a.calls[0]["seeds"] == [1]


def test_no_valid_seed_means_empty_and_exhausted_without_calling_engine():
    a = Fake(); r = a.recommend(k=3, seeds=["999"])
    assert (r.items, r.exhausted, r.dropped_seeds, a.calls) == ([], True, ["999"], [])
    assert Fake().recommend(k=3, seeds=[]).exhausted is True


def test_dislike_wins_over_seed_and_is_not_reported_as_dropped():
    a = Fake(); r = a.recommend(k=3, seeds=["1", "2"], disliked=["2"])
    assert a.calls[0]["seeds"] == [1] and a.calls[0]["disliked"] == [2] and r.dropped_seeds == []


def test_unknown_exclusions_are_silently_ignored():
    a = Fake(); a.recommend(k=3, seeds=["1"], disliked=["777"], excluded=["3", "x"], seen=["4", "888"])
    assert (a.calls[0]["disliked"], a.calls[0]["excluded"], a.calls[0]["seen"]) == ([], [3], [4])


def test_exhausted_when_fewer_than_k():
    assert Fake(n_return=3).recommend(k=3, seeds=["1"]).exhausted is False
    assert Fake(n_return=2).recommend(k=3, seeds=["1"]).exhausted is True


def test_media_is_refused_by_platforms_without_media():
    with pytest.raises(ValueError, match="media"): Fake().recommend(k=3, seeds=["1"], media="movie")


def test_frame_items_maps_columns_and_skips_missing_factors():
    f = pd.DataFrame({"item_id": [10], "seed_similarity": [0.5], "final_score": [0.6], "dominant_seed": [1],
                      "quality": [1.0], "tag_fit": [float("nan")]})
    (it,) = frame_items(f, key_col="item_id", to_key=str, dominant_to_key=str,
                        factor_cols={"quality": "quality", "tag_fit": "tag_fit", "has_mc": "has_mc"},
                        episode_of=lambda k: 30)
    assert it == AdapterItem(key="10", dominant_seed="1", final=0.6, sim=0.5, factors={"quality": 1.0}, episode_count=30)


def test_frame_items_on_empty_frame_without_score_columns():
    assert frame_items(pd.DataFrame(columns=["item_id", "name", "rank"]), key_col="item_id", to_key=str,
                       dominant_to_key=str, factor_cols={}) == []
