import json

import httpx, pytest
from fastapi.testclient import TestClient
from aod_serving.router.app import create_app
from aod_serving.router.client import EngineClient
from aod_serving.router.mixing import load_m6


def engine(platform, n=30, status=200):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(status, json={"ready": status == 200, "platform": platform})
        if status != 200:
            return httpx.Response(status, json={"error": "not_ready"})
        items = [{"key": f"{platform[0]}{i}", "rank": i, "dominantSeed": "x", "score": {"final": 1.0, "sim": 0.5, "factors": {}},
                  "episodeCount": 30 if platform == "webnovel" else None} for i in range(n)]
        # 실제 엔진처럼 요청 시드 수(원문 중복 제거)를 usedSeeds 로 되돌려준다 — 라우터가 이 값을
        # 믿고 전체 탭 M6 쿼터를 계산하므로(I2), 가짜 엔진도 0 이 아닌 그럴듯한 값을 줘야 한다.
        body_in = json.loads(request.content)
        used = len(dict.fromkeys(body_in.get("seeds", [])))
        return httpx.Response(200, json={"platform": platform, "items": items, "exhausted": False, "droppedSeeds": [],
                                         "factorSchema": f"{platform}.v1", "version": {"sha": "e", "config": "c", "corpus": "cv"},
                                         "tookMs": 3, "usedSeeds": used})
    return handler


def make_app(status_by_platform, transport=None):
    if transport is None:
        handlers = {f"rec-{p}": engine(p, status=s) for p, s in status_by_platform.items()}
        transport = httpx.MockTransport(lambda request: handlers[request.url.host](request))
    urls = {p: f"http://rec-{p}:8000" for p in status_by_platform}
    return create_app(EngineClient(urls, transport=transport), router_sha="r1")


def app_with(status_by_platform):
    return TestClient(make_app(status_by_platform))


ALL_OK = {"steam": 200, "tmdb": 200, "webtoon": 200, "webnovel": 200}


def test_recommend_all_tab():
    with app_with(ALL_OK) as c:
        r = c.post("/v1/recommend", json={"tab": "all", "k": 20, "buffer": 10,
                                          "seeds": {"steam": ["1", "2"], "tmdb": ["movie_1", "movie_2"], "webnovel": ["3", "4"]}})
    body = r.json()
    assert r.status_code == 200 and len(body["items"]) == 30 and body["partial"] == []
    first = body["items"][0]
    assert set(first) == {"platform", "key", "rank", "dominantSeed", "candidateSource", "isExploration", "propensity", "score", "factorSchema"}
    assert body["versions"]["router"] == "r1" and set(body["versions"]["engines"]) == {"steam", "tmdb", "webnovel"}


def test_one_engine_down_is_partial_200():
    with app_with({**ALL_OK, "steam": 503}) as c:
        body = c.post("/v1/recommend", json={"tab": "all", "seeds": {"steam": ["1"], "tmdb": ["movie_1"], "webnovel": ["3"]}}).json()
    assert body["partial"] == ["steam"] and {i["platform"] for i in body["items"]} == {"tmdb", "webnovel"}


def test_all_needed_engines_down_is_503():
    with app_with({**ALL_OK, "steam": 503}) as c:
        r = c.post("/v1/recommend", json={"tab": "game", "seeds": {"steam": ["1"]}})
    assert r.status_code == 503 and r.json() == {"error": "engines_unavailable", "partial": ["steam"]}


def test_bad_request_is_422():
    with app_with(ALL_OK) as c:
        assert c.post("/v1/recommend", json={"tab": "anime", "seeds": {}}).status_code == 422
        assert c.post("/v1/recommend", json={"tab": "game", "seeds": {"steam": [730]}}).status_code == 422


def test_health_is_always_200_and_lists_engines():
    with app_with({**ALL_OK, "webtoon": 503}) as c:
        r = c.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["ready"] is True and body["router_sha"] == "r1"
    assert body["engines"]["steam"]["ready"] is True and body["engines"]["webtoon"]["ready"] is False
    assert body["mix_loaded"] is True          # M6 은 기동 시 이미 적재돼 있다(I3)


def test_app_startup_fails_fast_when_mix_path_is_missing(monkeypatch, tmp_path):
    """M6(`crossdomain/mix.py`)이 없으면 첫 전체 탭 요청까지 기다리지 않고 기동 자체가 실패해야
    한다(I3) — 그래야 컨테이너 헬스체크가 절대 healthy 가 안 되고 배포가 트래픽을 안 받는다."""
    monkeypatch.setenv("MIX_PATH", str(tmp_path / "missing_mix.py"))
    load_m6.cache_clear()
    try:
        app = make_app(ALL_OK)
        with pytest.raises(Exception):
            with TestClient(app):
                pass
    finally:
        load_m6.cache_clear()


def test_too_many_seeds_is_422_and_no_engine_is_called():
    calls = []

    def counting(request):
        calls.append(request)
        return engine("steam")(request)

    app = make_app({"steam": 200}, transport=httpx.MockTransport(counting))
    with TestClient(app) as c:
        r = c.post("/v1/recommend", json={"tab": "game", "seeds": {"steam": [str(i) for i in range(51)]}})
    assert r.status_code == 422
    assert calls == []                     # 422 는 엔진을 부르기 전에 난다


def test_too_many_exclusions_is_422():
    with app_with(ALL_OK) as c:
        r = c.post("/v1/recommend", json={"tab": "game", "seeds": {"steam": ["1"]},
                                          "disliked": {"steam": [str(i) for i in range(5001)]}})
    assert r.status_code == 422


def test_unexpected_exception_returns_shaped_500_not_bare_text(monkeypatch):
    """`recommend()` 가 예상 못 한 예외를 던져도(엔진 쪽 §8-a 의 전역 핸들러와 같은 모양으로) 바
    text/plain 500 이 아니라 형태를 갖춘 JSON 500 이 나야 한다. `TestClient` 로 실제 바디를 보려면
    엔진 쪽 테스트와 마찬가지로 `raise_server_exceptions=False` 가 필요하다(Starlette 가 등록된
    핸들러로 응답한 뒤에도 예외를 다시 던지기 때문)."""
    import aod_serving.router.app as app_module

    async def boom(req, call, *, router_sha):
        raise RuntimeError("boom")

    monkeypatch.setattr(app_module, "recommend", boom)
    app = make_app(ALL_OK)
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/v1/recommend", json={"tab": "game", "seeds": {"steam": ["1"]}})
    assert r.status_code == 500 and r.json() == {"error": "internal"}
    assert r.headers["content-type"].startswith("application/json")
