from __future__ import annotations

import os

import httpx

from aod_ai.models import ReviewSource

_DEFAULT_MAX_SOURCES = 8          # VANE_MAX_SOURCES 미설정 시 기본 (config §5)
_SOURCE_CHAR_BUDGET = 2000        # 길이예산: 소스당 content 상한 (evidence 뭉갬 방지)
_TIMEOUT = 60.0


class VaneClient:
    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=_TIMEOUT)

    def search(
        self,
        *,
        query: str,
        sources: list[str],
        system_instructions: str | None = None,
    ) -> list[ReviewSource]:
        max_sources = int(os.getenv("VANE_MAX_SOURCES", str(_DEFAULT_MAX_SOURCES)))

        body: dict = {"query": query, "sources": sources, "stream": False}
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
