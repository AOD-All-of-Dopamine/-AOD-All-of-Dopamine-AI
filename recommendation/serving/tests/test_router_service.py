import asyncio
import pytest
from aod_serving.common.models import EngineItem, EngineResponse, RouterRequest, Score, VersionInfo
from aod_serving.router.service import EnginesUnavailable, recommend


def resp(platform, n, *, dropped=(), exhausted=False, episodes=30, used_seeds=0):
    return EngineResponse(platform=platform, exhausted=exhausted, dropped_seeds=list(dropped), factor_schema=f"{platform}.v1",
                          version=VersionInfo(sha="e1", config="c1", corpus="cv"), took_ms=5, used_seeds=used_seeds,
                          items=[EngineItem(key=f"{platform[0]}{i}", rank=i, dominant_seed="seed", score=Score(final=1.0, sim=0.5),
                                            episode_count=episodes if platform == "webnovel" else None) for i in range(n)])


class Engines:
    def __init__(self, behaviour): self.behaviour, self.calls = behaviour, {}

    async def __call__(self, platform, req):
        self.calls[platform] = req
        b = self.behaviour[platform]
        if isinstance(b, Exception): raise b
        return b


def run(req, engines):
    return asyncio.run(recommend(RouterRequest.model_validate(req), engines, router_sha="r1"))


def test_game_tab_calls_steam_with_k_plus_buffer_and_keeps_engine_order():
    e = Engines({"steam": resp("steam", 30)})
    out = run({"tab": "game", "k": 20, "buffer": 10, "seeds": {"steam": ["730"]}, "seen": {"steam": ["1"]},
               "disliked": {"steam": ["2"]}, "excluded": {"steam": ["3"]}}, e)
    r = e.calls["steam"]
    assert (r.k, r.seeds, r.seen, r.disliked, r.excluded, r.media) == (30, ["730"], ["1"], ["2"], ["3"], None)
    assert [i.key for i in out.items] == [f"s{i}" for i in range(30)] and [i.rank for i in out.items] == list(range(30))
    assert out.items[0].platform == "steam" and out.items[0].factor_schema == "steam.v1" and out.items[0].propensity == 1.0
    assert out.exhausted == {"steam": False} and out.partial == []
    assert out.versions.router == "r1" and out.versions.engines["steam"].corpus == "cv"


@pytest.mark.parametrize("tab, media", [("movie", "movie"), ("tv", "tv")])
def test_movie_and_tv_tabs_use_tmdb_with_media(tab, media):
    e = Engines({"tmdb": resp("tmdb", 5, exhausted=True)})
    out = run({"tab": tab, "seeds": {"tmdb": ["movie_1"], "steam": ["730"]}}, e)
    assert set(e.calls) == {"tmdb"} and e.calls["tmdb"].media == media and out.exhausted == {"tmdb": True}


def test_engine_k_is_capped_at_100():
    e = Engines({"webtoon": resp("webtoon", 1)})
    run({"tab": "webtoon", "k": 50, "buffer": 50, "seeds": {"webtoon": ["1"]}}, e)
    assert e.calls["webtoon"].k == 100


def test_all_tab_calls_three_platforms_with_k50_and_mixes():
    e = Engines({"steam": resp("steam", 50, used_seeds=2), "tmdb": resp("tmdb", 50, used_seeds=3),
                "webnovel": resp("webnovel", 50, used_seeds=2)})
    out = run({"tab": "all", "k": 20, "buffer": 10, "seeds": {"steam": ["a", "b"], "tmdb": ["c", "d", "e"], "webnovel": ["f", "g"],
                                                              "webtoon": ["z"]}}, e)
    assert set(e.calls) == {"steam", "tmdb", "webnovel"} and all(r.k == 50 for r in e.calls.values())
    assert len(out.items) == 30 and [i.platform for i in out.items[:3]] == ["tmdb", "steam", "webnovel"]
    assert [i.rank for i in out.items] == list(range(30))


def test_platform_without_seeds_is_not_called_and_marked_exhausted():
    e = Engines({"steam": resp("steam", 50, used_seeds=2)})
    out = run({"tab": "all", "seeds": {"steam": ["a", "b"], "tmdb": []}}, e)
    assert set(e.calls) == {"steam"} and out.exhausted == {"steam": False, "tmdb": True, "webnovel": True}
    assert {i.platform for i in out.items} == {"steam"}


def test_no_seeds_at_all_is_an_empty_200():
    out = run({"tab": "game", "seeds": {}}, Engines({}))
    assert out.items == [] and out.exhausted == {"steam": True} and out.partial == []


def test_failed_platform_is_partial_and_its_quota_goes_to_the_others():
    e = Engines({"steam": TimeoutError("1.5s"), "tmdb": resp("tmdb", 50, used_seeds=2), "webnovel": resp("webnovel", 50, used_seeds=2)})
    out = run({"tab": "all", "k": 20, "buffer": 0, "seeds": {"steam": ["a", "b"], "tmdb": ["c", "d"], "webnovel": ["e", "f"]}}, e)
    assert out.partial == ["steam"] and "steam" not in out.exhausted and len(out.items) == 20
    assert {i.platform for i in out.items} == {"tmdb", "webnovel"}


def test_all_called_engines_failing_raises():
    e = Engines({"steam": TimeoutError()})
    with pytest.raises(EnginesUnavailable) as ex:
        run({"tab": "game", "seeds": {"steam": ["730"]}}, e)
    assert ex.value.partial == ["steam"]


def test_router_trusts_engines_reported_used_seeds_for_m6_quota():
    """라우터는 `usedSeeds` 를 그대로 믿는다 — 원본 요청 문자열을 다시 세지 않는다(I2).

    steam 요청 시드는 4개("a" 중복 1 · 코퍼스 밖 "x" 1 · 싫어요 "d" 1 — 원문 문자열만 보면
    유효 1개로 보인다)지만, 엔진이 보고하는 `usedSeeds` 가 진실이다. 여기서는 그 값을
    명시로 1 로 주고(steam) tmdb 는 3 으로 준다 — M6 은 시드 1개인 플랫폼의 쿼터를 절반으로
    깎으므로(`crossdomain/mix.py:M6`) k=9 에서 steam 쿼터가 3(=9*0.5/1.5)이 되어야 한다.
    """
    e = Engines({"steam": resp("steam", 50, dropped=["x"], used_seeds=1), "tmdb": resp("tmdb", 50, used_seeds=3)})
    out = run({"tab": "all", "k": 9, "buffer": 0, "seeds": {"steam": ["a", "a", "x", "d"], "tmdb": ["1", "2", "3"]},
               "disliked": {"steam": ["d"]}}, e)
    assert sum(i.platform == "steam" for i in out.items) == 3 and out.dropped_seeds == {"steam": ["x"]}


def test_all_tab_quota_review_example_seven_and_oh_oh_seven_dedup_to_one_seed():
    """리뷰가 지적한 사례: `seeds={"steam": ["7","007"], "tmdb": ["1"], "webnovel": ["2"]}`.

    "7"·"007" 은 어댑터가 파싱하면 같은 코퍼스 키라 엔진은 `usedSeeds=1` 을 보고한다(원문
    문자열 기준 raw dedup 이었다면 이것도 1이라 이 사례만으로는 버그가 안 드러나지만, 라우터가
    실제로 이 값을 쓰는지는 이 테스트가 확인한다). 세 플랫폼 다 시드 1개 → M6 가중치 0.5 로
    동일 → k=9 에서 쿼터 3/3/3(`crossdomain/mix.py:M6` — steam·tmdb·webnovel 균등 반올림).
    """
    e = Engines({"steam": resp("steam", 50, used_seeds=1), "tmdb": resp("tmdb", 50, used_seeds=1),
                "webnovel": resp("webnovel", 50, used_seeds=1)})
    out = run({"tab": "all", "k": 9, "buffer": 0,
               "seeds": {"steam": ["7", "007"], "tmdb": ["1"], "webnovel": ["2"]}}, e)
    counts = {p: sum(i.platform == p for i in out.items) for p in ("steam", "tmdb", "webnovel")}
    assert counts == {"steam": 3, "tmdb": 3, "webnovel": 3}


def test_platform_whose_seeds_were_all_dropped_is_left_out_of_the_mix():
    e = Engines({"steam": resp("steam", 0, dropped=["x"], exhausted=True, used_seeds=0), "tmdb": resp("tmdb", 50, used_seeds=2)})
    out = run({"tab": "all", "k": 10, "buffer": 0, "seeds": {"steam": ["x"], "tmdb": ["1", "2"]}}, e)
    assert len(out.items) == 10 and {i.platform for i in out.items} == {"tmdb"} and out.exhausted["steam"] is True


def test_engine_request_construction_failure_marks_only_that_platform_partial(monkeypatch):
    """방어적 이중화 — `EngineRequest(...)` 생성 자체가 실패해도(잔여 버그 등) 그 플랫폼만 partial 로
    빠지고 나머지 플랫폼은 정상 서빙돼야 한다(try 블록 안에서 만들어야 함)."""
    import aod_serving.router.service as service_module
    real_engine_request = service_module.EngineRequest

    def flaky(**kw):
        if kw["seeds"] == ["boom"]:
            raise ValueError("simulated construction failure")
        return real_engine_request(**kw)

    monkeypatch.setattr(service_module, "EngineRequest", flaky)
    e = Engines({"tmdb": resp("tmdb", 20, used_seeds=1), "webnovel": resp("webnovel", 20, used_seeds=1)})
    out = run({"tab": "all", "k": 10, "buffer": 0,
               "seeds": {"steam": ["boom"], "tmdb": ["c"], "webnovel": ["d"]}}, e)
    assert out.partial == ["steam"]
    assert "steam" not in e.calls          # 생성이 실패해서 엔진 호출까지 가지도 않았다
    assert {i.platform for i in out.items} == {"tmdb", "webnovel"}
