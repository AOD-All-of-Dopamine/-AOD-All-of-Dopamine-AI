from __future__ import annotations

from openai import OpenAI


class EmbeddingClient:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1024):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._model, input=text, dimensions=self.dim
        )
        vector = list(resp.data[0].embedding)
        assert len(vector) == self.dim, f"expected {self.dim} dims, got {len(vector)}"
        return vector
