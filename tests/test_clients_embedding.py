import pytest
import respx
from httpx import Response

from aod_ai.clients.embedding import EmbeddingClient


def _embeddings_response(values: list[float]) -> Response:
    return Response(
        200,
        json={
            "object": "list",
            "model": "m",
            "data": [{"object": "embedding", "index": 0, "embedding": values}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


@respx.mock
def test_embed_text_returns_dim_length_vector_and_sends_dimensions():
    route = respx.post("http://emb.test/embeddings").mock(
        return_value=_embeddings_response([0.0] * 1024)
    )
    client = EmbeddingClient(base_url="http://emb.test", api_key="k", model="m")
    vec = client.embed_text("프로파일 텍스트")
    assert len(vec) == 1024
    import json

    body = json.loads(route.calls.last.request.content)
    assert body["dimensions"] == 1024


@respx.mock
def test_embed_text_asserts_on_wrong_length():
    respx.post("http://emb.test/embeddings").mock(
        return_value=_embeddings_response([0.0] * 512)
    )
    client = EmbeddingClient(base_url="http://emb.test", api_key="k", model="m")
    with pytest.raises(AssertionError):
        client.embed_text("x")
