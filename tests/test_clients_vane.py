import json

import pytest
import respx
from httpx import Response

from aod_ai.clients.vane import VaneClient, _SOURCE_CHAR_BUDGET


@pytest.fixture()
def vane_env(monkeypatch):
    # v1.12.2: search 요청에 모델 지정 필수 — 모델/키는 env(§5)에서
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _providers_payload():
    return {
        "providers": [
            {
                "id": "uuid-1",
                "type": "openai",
                "name": "aod-openai",
                "chatModels": [{"name": "GPT 4o mini", "key": "gpt-4o-mini"}],
                "embeddingModels": [
                    {"name": "TE3 small", "key": "text-embedding-3-small"}
                ],
            }
        ]
    }


@respx.mock
def test_search_sends_models_dedupes_caps_and_posts_stream_false(monkeypatch, vane_env):
    monkeypatch.setenv("VANE_MAX_SOURCES", "2")  # 상한 N개 = config §5
    respx.get("http://vane.test/api/providers").mock(
        return_value=Response(200, json=_providers_payload())
    )
    route = respx.post("http://vane.test/api/search").mock(
        return_value=Response(
            200,
            json={
                "message": "합성답변",
                "sources": [
                    {"content": "리뷰1", "url": "http://a"},
                    {"content": "리뷰1-dup", "url": "http://a"},
                    {"content": "리뷰2", "url": "http://b"},
                    {"content": "리뷰3", "url": "http://c"},
                ],
            },
        )
    )
    client = VaneClient(base_url="http://vane.test")
    out = client.search(
        query='"오징어 게임" TV 리뷰 후기 재미',
        sources=["web", "discussions"],
        system_instructions="독자 반응·재미요소 위주",
    )
    assert [s.url for s in out] == ["http://a", "http://b"]
    body = json.loads(route.calls.last.request.content)
    assert body["stream"] is False
    assert body["sources"] == ["web", "discussions"]
    assert body["systemInstructions"] == "독자 반응·재미요소 위주"
    # v1.12.2 필수 필드: providerId(UUID) + key
    assert body["chatModel"] == {"providerId": "uuid-1", "key": "gpt-4o-mini"}
    assert body["embeddingModel"] == {
        "providerId": "uuid-1",
        "key": "text-embedding-3-small",
    }


@respx.mock
def test_search_caches_provider_resolution_across_calls(monkeypatch, vane_env):
    monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)
    providers_route = respx.get("http://vane.test/api/providers").mock(
        return_value=Response(200, json=_providers_payload())
    )
    respx.post("http://vane.test/api/search").mock(
        return_value=Response(200, json={"sources": []})
    )
    client = VaneClient(base_url="http://vane.test")
    client.search(query="q1", sources=["web"])
    client.search(query="q2", sources=["web"])
    assert providers_route.call_count == 1  # 캐시됨


@respx.mock
def test_search_provisions_openai_even_if_chatless_provider_exists(monkeypatch, vane_env):
    # 실통합 발견 버그: Vane 기본 'Transformers' 프로바이더는 chatModels가 비어 있음.
    # 이런 프로바이더로 fallback하면 500 — 반드시 openai를 새로 등록해야 한다.
    monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)
    respx.get("http://vane.test/api/providers").mock(
        return_value=Response(
            200,
            json={
                "providers": [
                    {
                        "id": "uuid-transformers",
                        "name": "Transformers",
                        "chatModels": [],
                        "embeddingModels": [{"name": "MiniLM", "key": "Xenova/all-MiniLM-L6-v2"}],
                    }
                ]
            },
        )
    )
    create_route = respx.post("http://vane.test/api/providers").mock(
        return_value=Response(200, json={"provider": {"id": "uuid-new"}})
    )
    search_route = respx.post("http://vane.test/api/search").mock(
        return_value=Response(200, json={"sources": []})
    )
    client = VaneClient(base_url="http://vane.test")
    client.search(query="q", sources=["web"])

    assert create_route.call_count == 1
    body = json.loads(search_route.calls.last.request.content)
    assert body["chatModel"]["providerId"] == "uuid-new"


@respx.mock
def test_search_auto_provisions_openai_provider_when_absent(monkeypatch, vane_env):
    monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)
    respx.get("http://vane.test/api/providers").mock(
        return_value=Response(200, json={"providers": []})
    )
    create_route = respx.post("http://vane.test/api/providers").mock(
        return_value=Response(200, json={"provider": {"id": "uuid-new"}})
    )
    search_route = respx.post("http://vane.test/api/search").mock(
        return_value=Response(200, json={"sources": []})
    )
    client = VaneClient(base_url="http://vane.test")
    client.search(query="q", sources=["web"])

    create_body = json.loads(create_route.calls.last.request.content)
    assert create_body["type"] == "openai"
    assert create_body["config"] == {
        "apiKey": "sk-test",
        "baseURL": "https://api.openai.com/v1",
    }
    body = json.loads(search_route.calls.last.request.content)
    assert body["chatModel"]["providerId"] == "uuid-new"


@respx.mock
def test_search_parses_metadata_url_shape(monkeypatch, vane_env):
    monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)  # 기본 8
    respx.get("http://vane.test/api/providers").mock(
        return_value=Response(200, json=_providers_payload())
    )
    respx.post("http://vane.test/api/search").mock(
        return_value=Response(
            200,
            json={
                "sources": [
                    {"pageContent": "본문", "metadata": {"url": "http://x"}},
                ]
            },
        )
    )
    client = VaneClient(base_url="http://vane.test")
    out = client.search(query="q", sources=["web"])
    assert out[0].content == "본문"
    assert out[0].url == "http://x"


@respx.mock
def test_search_applies_length_budget_trims_oversized_content(monkeypatch, vane_env):
    monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)
    respx.get("http://vane.test/api/providers").mock(
        return_value=Response(200, json=_providers_payload())
    )
    long_content = "가" * (_SOURCE_CHAR_BUDGET + 500)
    respx.post("http://vane.test/api/search").mock(
        return_value=Response(
            200,
            json={"sources": [{"content": long_content, "url": "http://a"}]},
        )
    )
    client = VaneClient(base_url="http://vane.test")
    out = client.search(query="q", sources=["web"])
    assert len(out[0].content) == _SOURCE_CHAR_BUDGET
    assert out[0].content == long_content[:_SOURCE_CHAR_BUDGET]
