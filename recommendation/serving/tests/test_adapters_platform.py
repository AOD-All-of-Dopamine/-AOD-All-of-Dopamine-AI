import pandas as pd, pytest
from aod_serving.engine.adapters import ADAPTERS
from aod_serving.engine.adapters.steam import SteamAdapter
from aod_serving.engine.adapters.tmdb import TmdbAdapter
from aod_serving.engine.adapters.webtoon import WebtoonAdapter
from aod_serving.engine.adapters.webnovel import WebnovelAdapter
from aod_serving.engine.overrides import EffectiveConfig


def cfg(production, postprocess=None):
    return EffectiveConfig(production=production, postprocess=postprocess or {}, hash="h" * 12, approved=False, verdict=None)


class Spy:
    def __init__(self, frame): self.frame, self.args, self.kw = frame, None, None
    def __call__(self, *args, **kw): self.args, self.kw = args, kw; return self.frame


def test_registry():
    assert set(ADAPTERS) == {"steam", "tmdb", "webtoon", "webnovel"} and ADAPTERS["tmdb"].supports_media


def test_steam_call_and_items():
    frame = pd.DataFrame({"steam_appid": [240], "seed_similarity": [0.58], "final_score": [1.19], "dominant_seed": [730],
                          "recommendations_percentile": [0.99], "quality": [1.0], "tag_fit": [0.75], "has_mc": [1.0]})
    spy = Spy(frame)
    a = SteamAdapter("/a", cfg({"strategy": "top2_mean", "rec_boost": 0.03})); a._bind(spy, "COMPS", known={730, 570, 10, 20, 30}, first=730)
    r = a.recommend(k=1, seeds=["730", "570"], disliked=["10"], excluded=["20"], seen=["30"])
    assert spy.args == ([730, 570],)
    assert spy.kw == {"seen_appids": {20, 30}, "page_size": 1, "strategy": "top2_mean", "rec_boost": 0.03,
                      "components": "COMPS", "disliked_appids": [10]}
    it = r.items[0]
    assert (it.key, it.dominant_seed, it.final, it.sim) == ("240", "730", 1.19, 0.58)
    assert it.factors == {"rec_pct": 0.99, "quality": 1.0, "tag_fit": 0.75, "has_mc": 1.0} and a.first_key() == "730"


def test_steam_passes_none_when_no_dislikes():
    spy = Spy(pd.DataFrame({"steam_appid": [], "final_score": [], "seed_similarity": []}))
    a = SteamAdapter("/a", cfg({"strategy": "top2_mean", "rec_boost": 0.03})); a._bind(spy, "C", known={730}, first=730)
    a.recommend(k=5, seeds=["730"]); assert spy.kw["disliked_appids"] is None and spy.kw["seen_appids"] == set()


def test_tmdb_keys_are_item_ids_and_everything_merges_into_seen_rows():
    frame = pd.DataFrame({"row": [3], "item_id": ["tv_9"], "seed_similarity": [0.5], "final_score": [0.6], "dominant_seed": [0]})
    spy = Spy(frame)
    a = TmdbAdapter("/a", cfg({"strategy": "mean"}, {"franchise_max": 1}))
    a._bind(spy, "COMPS", item_ids=["movie_1", "movie_2", "tv_3", "tv_9"])
    r = a.recommend(k=1, seeds=["movie_1", "movie_404", "7"], disliked=["movie_2"], excluded=["tv_3"], seen=["tv_9"], media="tv")
    assert spy.args == ([0],) and r.dropped_seeds == ["movie_404", "7"]
    assert spy.kw == {"seen_rows": {1, 2, 3}, "page_size": 1, "components": "COMPS", "strategy": "mean",
                      "postprocess_kwargs": {"franchise_max": 1}, "media": "tv"}
    assert (r.items[0].key, r.items[0].dominant_seed) == ("tv_9", "movie_1") and a.first_key() == "movie_1"


def test_webtoon_call():
    frame = pd.DataFrame({"item_id": [5], "seed_similarity": [0.5], "final_score": [0.6], "dominant_seed": [1]})
    spy = Spy(frame)
    prod = {"strategy": "top2_mean", "pop_boost": 0.0, "hub_lambda": 0.0, "star_boost": 0.0, "tag_w": 0.2, "creator_w": 0.0,
            "tag_drop_genre": False, "dislike_w": 0.0, "dislike_floor": 0.61}
    a = WebtoonAdapter("/a", cfg(prod, {"series_max": 1, "artist_max": 2, "drop_adult": True}))
    a._bind(spy, known={1, 2, 3, 4}, first=1)
    a.recommend(k=1, seeds=["1"], disliked=["2"], excluded=["3"], seen=["4"])
    assert spy.args == ([1],)
    assert spy.kw == {"k": 1, "seen": [3, 4], "disliked_ids": [2], **prod, "series_max": 1, "artist_max": 2, "drop_adult": True}


def test_webnovel_call_and_episode_count():
    frame = pd.DataFrame({"item_id": [5, 6], "seed_similarity": [0.5, 0.4], "final_score": [0.6, 0.5], "dominant_seed": [1, 1],
                          "interest_percentile": [0.9, 0.1]})
    spy = Spy(frame)
    a = WebnovelAdapter("/a", cfg({"strategy": "top2_mean", "pop_boost": 0.0, "min_interest_count": None, "drop_excluded_series": True}))
    a._bind(spy, "COMPS", known={1, 2, 3, 4, 5, 6}, first=1, episodes={5: 203, 6: None})
    r = a.recommend(k=2, seeds=["1"], disliked=["2"], excluded=["3"], seen=["4"])
    assert spy.args == ([1],)
    assert spy.kw == {"seen_ids": {2, 3, 4}, "page_size": 2, "strategy": "top2_mean", "pop_boost": 0.0, "components": "COMPS",
                      "min_interest_count": None, "drop_excluded_series": True}
    assert [(i.key, i.episode_count, i.factors) for i in r.items] == [("5", 203, {"interest_pct": 0.9}), ("6", None, {"interest_pct": 0.1})]
