"""라우터 HTTP — POST /v1/recommend · GET /health (REC_TAB_DESIGN §4-2)."""
from __future__ import annotations
import asyncio, os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from aod_serving.common.models import RouterRequest, RouterResponse
from aod_serving.router.client import PLATFORMS, EngineClient, urls_from_env
from aod_serving.router.service import EnginesUnavailable, recommend


def create_app(client: EngineClient, *, router_sha: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="aod-rec-router", lifespan=lifespan)

    @app.post("/v1/recommend", response_model=RouterResponse)
    async def v1_recommend(req: RouterRequest):
        try:
            return await recommend(req, client.recommend, router_sha=router_sha)
        except EnginesUnavailable as e:
            return JSONResponse({"error": "engines_unavailable", "partial": e.partial}, status_code=503)

    @app.get("/health")
    async def health():
        known = [p for p in PLATFORMS if p in client._urls]
        states = await asyncio.gather(*(client.health(p) for p in known))
        return {"ready": True, "router_sha": router_sha, "engines": dict(zip(known, states))}

    return app


def app_from_env() -> FastAPI:
    timeout_s = int(os.environ.get("ENGINE_TIMEOUT_MS", "1500")) / 1000
    return create_app(EngineClient(urls_from_env(), timeout_s=timeout_s), router_sha=os.environ.get("GIT_SHA", "dev"))
