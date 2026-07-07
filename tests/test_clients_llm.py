import json

import respx
from httpx import Response

from aod_ai.clients.llm import LlmClient
from aod_ai.models import Extraction, ReviewSource, SelectedTarget

_GOOD = {
    "fun_tags": [
        {
            "tag": "회귀",
            "tag_score": 0.9,
            "tag_confidence": 0.8,
            "evidence": "회귀물",
            "is_new": False,
        }
    ],
    "normalized_summary": "요약",
    "profile_text": "프로파일",
    "extraction_quality": 0.7,
}


def _target():
    return SelectedTarget(
        content_id=1,
        domain="WEBNOVEL",
        master_title="회귀한 천재",
        original_title=None,
        synopsis="s",
        genres=["판타지"],
        content_hash="h",
    )


def _chat_response(content: str) -> Response:
    return Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
        },
    )


@respx.mock
def test_extract_returns_validated_extraction_and_sends_json_and_schema():
    route = respx.post("http://llm.test/chat/completions").mock(
        return_value=_chat_response(json.dumps(_GOOD))
    )
    client = LlmClient(base_url="http://llm.test", api_key="k", model="m")
    result = client.extract(
        metadata=_target(),
        sources=[ReviewSource(content="재밌음", url="http://u")],
        active_tags=["회귀", "먼치킨"],
    )
    assert isinstance(result, Extraction)
    assert result.fun_tags[0].tag == "회귀"
    body = json.loads(route.calls.last.request.content)
    assert body["response_format"] == {"type": "json_object"}
    prompt = body["messages"][-1]["content"]
    assert "json" in prompt
    assert "fun_tags" in prompt


@respx.mock
def test_extract_retries_on_invalid_json_then_succeeds(monkeypatch):
    monkeypatch.setenv("EXTRACT_MAX_RETRIES", "3")  # 재시도는 config §5에서
    route = respx.post("http://llm.test/chat/completions").mock(
        side_effect=[
            _chat_response('{"not":"an extraction"}'),
            _chat_response(json.dumps(_GOOD)),
        ]
    )
    client = LlmClient(base_url="http://llm.test", api_key="k", model="m")
    result = client.extract(metadata=_target(), sources=[], active_tags=["회귀"])
    assert result.extraction_quality == 0.7
    assert route.call_count == 2
