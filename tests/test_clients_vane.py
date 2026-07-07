import json

import respx
from httpx import Response

from aod_ai.clients.vane import VaneClient, _SOURCE_CHAR_BUDGET


@respx.mock
def test_search_dedupes_by_url_caps_and_posts_stream_false(monkeypatch):
    monkeypatch.setenv("VANE_MAX_SOURCES", "2")  # 상한 N개 = config §5
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


@respx.mock
def test_search_parses_perplexica_pagecontent_shape(monkeypatch):
    monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)  # 기본 8
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
def test_search_applies_length_budget_trims_oversized_content(monkeypatch):
    monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)
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
