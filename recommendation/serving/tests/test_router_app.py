import httpx
from fastapi.testclient import TestClient
from aod_serving.router.app import create_app
from aod_serving.router.client import EngineClient


def engine(platform, n=30, status=200):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(status, json={"ready": status == 200, "platform": platform})
        if status != 200:
            return httpx.Response(status, json={"error": "not_ready"})
        items = [{"key": f"{platform[0]}{i}", "rank": i, "dominantSeed": "x", "score": {"final": 1.0, "sim": 0.5, "factors": {}},
                  "episodeCount": 30 if platform == "webnovel" else None} for i in range(n)]
        return httpx.Response(200, json={"platform": platform, "items": items, "exhausted": False, "droppedSeeds": [],
                                         "factorSchema": f"{platform}.v1", "version": {"sha": "e", "config": "c", "corpus": "cv"}, "tookMs": 3})
    return handler


def app_with(status_by_platform):
    handlers = {f"rec-{p}": engine(p, status=s) for p, s in status_by_platform.items()}
    transport = httpx.MockTransport(lambda request: handlers[request.url.host](request))
    urls = {p: f"http://rec-{p}:8000" for p in status_by_platform}
    return TestClient(create_app(EngineClient(urls, transport=transport), router_sha="r1"))


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
