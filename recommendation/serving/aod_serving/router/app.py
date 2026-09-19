"""라우터 HTTP — POST /v1/recommend · GET /health (REC_TAB_DESIGN §4-2)."""
from __future__ import annotations
import asyncio, logging, os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aod_serving.common.models import RouterRequest, RouterResponse
from aod_serving.router.client import PLATFORMS, EngineClient, urls_from_env
from aod_serving.router.mixing import load_m6
from aod_serving.router.service import EnginesUnavailable, recommend

log = logging.getLogger("aod.router")


def create_app(client: EngineClient, *, router_sha: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # M6(crossdomain/mix.py)을 기동 시 한 번 적재한다 — 요청까지 미루면 mix.py 가 없을 때
        # /health 는 준비됐다고 하는데 전체 탭 요청마다 500 이 난다. 실패는 기동을 그대로 중단한다
        # (fail fast) — 컨테이너 헬스체크가 절대 healthy 가 되지 않아 잘못 배포된 이미지가 트래픽을
        # 받지 않는다.
        load_m6()
        yield
        await client.aclose()

    app = FastAPI(title="aod-rec-router", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # 엔진 쪽 app.py 와 같은 방어 — 예상 못 한 예외가 bare text/plain 500 으로 새 나가지 않게,
        # 트레이스는 로그로만 남기고(요청 본문은 남기지 않는다) 형태를 갖춘 JSON 으로 돌려준다.
        log.exception("처리되지 않은 예외 — %s %s", request.method, request.url.path)
        return JSONResponse({"error": "internal"}, status_code=500)

    @app.post("/v1/recommend", response_model=RouterResponse)
    async def v1_recommend(req: RouterRequest):
        try:
            return await recommend(req, client.recommend, router_sha=router_sha)
        except EnginesUnavailable as e:
            return JSONResponse({"error": "engines_unavailable", "partial": e.partial}, status_code=503)

    @app.get("/health")
    async def health():
        known = [p for p in PLATFORMS if p in client.platforms()]
        states = await asyncio.gather(*(client.health(p) for p in known))
        try:
            load_m6(); mix_loaded = True      # 캐시돼 있다(lru_cache) — 이미 적재됐으면 공짜다
        except Exception:                     # noqa: BLE001 — /health 자체는 계속 200 으로 답한다
            mix_loaded = False
        return {"ready": True, "router_sha": router_sha, "mix_loaded": mix_loaded, "engines": dict(zip(known, states))}

    return app


def app_from_env() -> FastAPI:
    timeout_s = int(os.environ.get("ENGINE_TIMEOUT_MS", "1500")) / 1000
    return create_app(EngineClient(urls_from_env(), timeout_s=timeout_s), router_sha=os.environ.get("GIT_SHA", "dev"))
