import asyncio, json
import httpx, pytest
from aod_serving.common.models import EngineRequest
from aod_serving.router.client import EngineCallError, EngineClient

OK = {"platform": "steam", "items": [], "exhausted": True, "droppedSeeds": [], "factorSchema": "steam.v1",
      "version": {"sha": "e", "config": "c", "corpus": "tags_full"}, "tookMs": 1}


def client(handler, **kw):
    return EngineClient({"steam": "http://rec-steam:8000"}, transport=httpx.MockTransport(handler), **kw)


def call(c, platform="steam"):
    async def go():
        try:
            return await c.recommend(platform, EngineRequest(k=5, seeds=["730"]))
        finally:
            await c.aclose()
    return asyncio.run(go())


def test_posts_camel_case_json_to_engine_and_parses_response():
    seen = {}
    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json=OK)
    r = call(client(handler))
    assert seen["url"] == "http://rec-steam:8000/engine/recommend"
    assert seen["body"] == {"k": 5, "seeds": ["730"], "disliked": [], "excluded": [], "seen": [], "media": None}
    assert r.platform == "steam" and r.version.corpus == "tags_full"


@pytest.mark.parametrize("response", [httpx.Response(503, json={"error": "busy"}), httpx.Response(500, text="boom"),
                                      httpx.Response(200, json={"nope": 1})])
def test_non_200_or_malformed_body_is_an_engine_error(response):
    with pytest.raises(EngineCallError): call(client(lambda request: response))


def test_connect_failure_is_an_engine_error():
    def handler(request): raise httpx.ConnectError("refused")
    with pytest.raises(EngineCallError, match="ConnectError"): call(client(handler))


def test_total_deadline_is_enforced():
    async def handler(request):
        await asyncio.sleep(0.5); return httpx.Response(200, json=OK)
    with pytest.raises(EngineCallError, match="timeout"): call(client(handler, timeout_s=0.1))


def test_unknown_platform_is_an_engine_error():
    with pytest.raises(EngineCallError, match="tmdb"): call(client(lambda r: httpx.Response(200, json=OK)), platform="tmdb")
