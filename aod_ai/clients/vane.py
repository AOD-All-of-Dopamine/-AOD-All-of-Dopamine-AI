from __future__ import annotations

import logging
import os

import httpx

from aod_ai.models import ReviewSource

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SOURCES = 8          # VANE_MAX_SOURCES 미설정 시 기본 (config §5)
_SOURCE_CHAR_BUDGET = 2000        # 길이예산: 소스당 content 상한 (evidence 뭉갬 방지)
_TIMEOUT = 120.0                  # Vane 내부 검색+LLM 합성이 느릴 수 있음
_PROVIDER_NAME = "aod-openai"


class VaneClient:
    """Vane(구 Perplexica) v1.12.x /api/search 클라이언트.

    v1.12.x는 요청에 chatModel/embeddingModel({providerId, key})가 필수다.
    providerId는 GET /api/providers에서 해석하고, 프로바이더가 없으면
    env(OPENAI_API_KEY/OPENAI_BASE_URL)로 openai 프로바이더를 자동 등록한다.
    """

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=_TIMEOUT)
        self._provider_id: str | None = None

    def _resolve_provider_id(self, llm_key: str) -> str:
        if self._provider_id is not None:
            return self._provider_id

        resp = self._client.get(f"{self._base_url}/api/providers")
        resp.raise_for_status()
        providers = resp.json().get("providers", []) or []

        # LLM 키를 실제로 제공하는 프로바이더만 사용 (기본 'Transformers'처럼
        # chatModels가 빈 프로바이더로 fallback하면 /api/search가 500을 낸다)
        chosen = next(
            (
                p for p in providers
                if any(m.get("key") == llm_key for m in p.get("chatModels", []))
            ),
            None,
        )
        if chosen is None:
            body = {
                "type": "openai",
                "name": _PROVIDER_NAME,
                "config": {
                    "apiKey": os.getenv("OPENAI_API_KEY", ""),
                    "baseURL": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                },
            }
            resp = self._client.post(f"{self._base_url}/api/providers", json=body)
            resp.raise_for_status()
            chosen = resp.json()["provider"]
            logger.info("openai 프로바이더 자동 등록: %s", chosen.get("id"))

        self._provider_id = chosen["id"]
        return self._provider_id

    def search(
        self,
        *,
        query: str,
        sources: list[str],
        system_instructions: str | None = None,
    ) -> list[ReviewSource]:
        max_sources = int(os.getenv("VANE_MAX_SOURCES", str(_DEFAULT_MAX_SOURCES)))
        llm_key = os.getenv("LLM_MODEL", "")
        emb_key = os.getenv("EMBEDDING_MODEL", "")
        provider_id = self._resolve_provider_id(llm_key)

        body: dict = {
            "query": query,
            "sources": sources,
            "stream": False,
            "chatModel": {"providerId": provider_id, "key": llm_key},
            "embeddingModel": {"providerId": provider_id, "key": emb_key},
        }
        if system_instructions is not None:
            body["systemInstructions"] = system_instructions

        resp = self._client.post(f"{self._base_url}/api/search", json=body)
        resp.raise_for_status()
        raw = resp.json().get("sources", []) or []

        seen: set[str] = set()
        out: list[ReviewSource] = []
        for item in raw:
            url = item.get("url") or (item.get("metadata") or {}).get("url")
            content = item.get("content") or item.get("pageContent")
            if not url or not content or url in seen:  # dedupe(url)
                continue
            seen.add(url)
            content = content[:_SOURCE_CHAR_BUDGET]  # 길이예산 적용
            out.append(ReviewSource(content=content, url=url))
            if len(out) >= max_sources:  # 상한 N개(config)
                break
        return out
