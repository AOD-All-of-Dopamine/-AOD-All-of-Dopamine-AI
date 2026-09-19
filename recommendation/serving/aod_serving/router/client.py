"""엔진 HTTP 클라이언트 — 요청 전체에 제한 시간을 건다(§8-5: 라우터 → 엔진 1.5초)."""
from __future__ import annotations
import asyncio, os

import httpx
from pydantic import ValidationError

from aod_serving.common.models import EngineRequest, EngineResponse

PLATFORMS = ("steam", "tmdb", "webtoon", "webnovel")


class EngineCallError(Exception):
    """엔진 호출 실패 — 시간 초과 · 연결 실패 · 200 이외 · 본문 형식 오류. 라우터는 그 플랫폼을 partial 로 뺀다."""


def urls_from_env() -> dict[str, str]:
    return {p: os.environ.get(f"ENGINE_{p.upper()}_URL", f"http://rec-{p}:8000").rstrip("/") for p in PLATFORMS}


class EngineClient:
    def __init__(self, urls: dict[str, str], *, timeout_s: float = 1.5, connect_s: float = 0.3,
                 transport: httpx.AsyncBaseTransport | None = None):
        self._urls, self._timeout = urls, timeout_s
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=connect_s), transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def recommend(self, platform: str, req: EngineRequest) -> EngineResponse:
        base = self._urls.get(platform)
        if not base:
            raise EngineCallError(f"{platform}: 엔진 주소가 없다")
        try:
            r = await asyncio.wait_for(self._http.post(f"{base}/engine/recommend", json=req.model_dump(by_alias=True)),
                                       self._timeout)
        except (asyncio.TimeoutError, httpx.TimeoutException) as e:
            raise EngineCallError(f"{platform}: timeout {self._timeout}s") from e
        except httpx.HTTPError as e:
            raise EngineCallError(f"{platform}: {type(e).__name__}: {e}") from e
        if r.status_code != 200:
            raise EngineCallError(f"{platform}: HTTP {r.status_code} {r.text[:200]}")
        try:
            return EngineResponse.model_validate(r.json())
        except (ValueError, ValidationError) as e:
            raise EngineCallError(f"{platform}: 응답 형식 오류: {e}") from e

    async def health(self, platform: str, timeout_s: float = 0.5) -> dict:
        try:
            r = await asyncio.wait_for(self._http.get(f"{self._urls[platform]}/health"), timeout_s)
            body = r.json()
            return body if isinstance(body, dict) else {"ready": False, "error": "bad body"}
        except Exception as e:                        # noqa: BLE001 — 상태 조회는 실패해도 라우터 health 를 깨지 않는다
            return {"ready": False, "error": f"{type(e).__name__}"}
