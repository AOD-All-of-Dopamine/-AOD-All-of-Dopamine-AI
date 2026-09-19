"""엔진 HTTP 서비스 — 플랫폼 하나, 프로세스 하나 (REC_TAB_DESIGN §8-1·§8-6)."""
from __future__ import annotations
import logging, os, threading, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from aod_serving.common.models import EngineItem, EngineRequest, EngineResponse, Score, VersionInfo
from aod_serving.engine.contract import ArtifactError
from aod_serving.engine.overrides import ConfigError

log = logging.getLogger("aod.engine")


def default_loader(platform: str, corpus_version: str, mode: str) -> Callable[[], tuple[object, str]]:
    """설정 → 계약 검증 → 어댑터 적재. (어댑터, 설정 해시)를 돌려준다."""
    def load():
        from aod_serving.engine.adapters import ADAPTERS
        from aod_serving.engine.bootstrap import default_artifacts, enter_platform
        from aod_serving.engine.contract import load_schema, validate_artifacts
        from aod_serving.engine.overrides import platform_defaults, resolve_config
        d = default_artifacts(platform, corpus_version)
        enter_platform(platform, artifacts=d)
        prod, post = platform_defaults(platform)
        cfg = resolve_config(platform, d, defaults=prod, post_defaults=post, corpus_version=corpus_version, mode=mode)
        info = validate_artifacts(d, load_schema(platform), corpus_version=corpus_version, production=cfg.production)
        log.info("아티팩트 검증 통과 %s", info)
        adapter = ADAPTERS[platform](d, cfg)
        adapter.load()
        return adapter, cfg.hash
    return load


class EngineState:
    def __init__(self, *, platform: str, corpus_version: str, engine_sha: str, loader: Callable[[], tuple[object, str]]):
        self.platform, self.corpus_version, self.engine_sha, self._loader = platform, corpus_version, engine_sha, loader
        self.adapter = None; self.config_hash = ""; self.ready = False; self.reason = "loading"

    def load(self) -> None:
        t0 = time.perf_counter()
        try:
            adapter, self.config_hash = self._loader()
            adapter.recommend(k=10, seeds=[adapter.first_key()])      # 예열 — 임베딩 파일을 미리 읽혀 둔다(§8-6)
            self.adapter, self.ready, self.reason = adapter, True, ""
            log.info("%s 준비 완료 %.1fs", self.platform, time.perf_counter() - t0)
        except ArtifactError as e:
            self.reason = f"artifact_invalid: {e}"
        except ConfigError as e:
            self.reason = f"config_invalid: {e}"
        except Exception as e:                                         # noqa: BLE001 — 사유를 /health 로 보여 주려고 잡는다
            log.exception("적재 실패"); self.reason = f"load_failed: {type(e).__name__}: {e}"
        if not self.ready:
            log.error("%s 준비 실패 — %s", self.platform, self.reason)

    def version(self) -> VersionInfo:
        return VersionInfo(sha=self.engine_sha, config=self.config_hash, corpus=self.corpus_version)


class Busy(Exception):
    """대기열이 가득 찼다 — 즉시 503 busy."""


class _Gate:
    """한 번에 한 요청만 계산한다(FIFO). 파이썬 구간은 GIL 에 묶여 있어 동시에 돌려도 총 처리량은 같고
    개별 지연 · 최대 메모리만 나빠진다 — 그래서 계산은 직렬화하고, 대기만 `max_queue` 로 제한해 넘치면 즉시 거절한다.

    계획서 원안은 락 두 개(`_run`·`_count`)로 짰는데, `_run.locked()` 를 참고해서 거절 여부를 정하다 보니
    "락이 막 풀렸지만 대기하던 스레드가 아직 깨어나 돌려받기 전" 인 아주 짧은 틈에 새 스레드가 끼어들면
    `_waiting` 이 `max_queue` 를 순간적으로 넘을 수 있었다(거절해야 할 스레드가 못 거절되는 경쟁 상태).
    `threading.Condition` 하나로 `running`·`waiting` 상태 전이를 전부 같은 락 밑에서 처리하면 그 틈이 없다 —
    `wait()`/`notify()` 는 대기자 큐를 FIFO 로 깨우고(CPython 구현), 놓치는 깨우기(lost wakeup)도 없다.
    """

    def __init__(self, max_queue: int):
        self._cond = threading.Condition()
        self._running = False
        self._waiting = 0
        self.max_queue = max_queue

    def __enter__(self):
        with self._cond:
            if self._running:
                if self._waiting >= self.max_queue:
                    raise Busy()
                self._waiting += 1
                try:
                    while self._running:
                        self._cond.wait()
                finally:
                    self._waiting -= 1
            self._running = True
        return self

    def __exit__(self, *exc):
        with self._cond:
            self._running = False
            self._cond.notify()


def _kb(path: str, fields: dict[str, str]) -> dict[str, float | None]:
    out = {name: None for name in fields.values()}
    try:
        for line in Path(path).read_text().splitlines():
            k, _, v = line.partition(":")
            if k in fields:
                out[fields[k]] = round(int(v.split()[0]) / 1024, 1)
    except OSError:
        pass
    return out


def memory_report() -> dict:
    rep = _kb("/proc/self/status", {"VmRSS": "rss_mb", "RssFile": "rss_file_mb", "RssAnon": "rss_anon_mb"})
    cg = {}
    for name, f in (("current_mb", "memory.current"), ("peak_mb", "memory.peak")):
        try: cg[name] = round(int(Path("/sys/fs/cgroup", f).read_text()) / 2**20, 1)
        except (OSError, ValueError): cg[name] = None
    try:
        stat = dict(l.split() for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
        cg["anon_mb"], cg["file_mb"] = round(int(stat["anon"]) / 2**20, 1), round(int(stat["file"]) / 2**20, 1)
    except (OSError, KeyError, ValueError):
        cg["anon_mb"] = cg["file_mb"] = None
    return {**rep, "cgroup": cg}


def create_app(state: EngineState, *, load_in_background: bool = True, max_queue: int = 8) -> FastAPI:
    gate = _Gate(max_queue)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_in_background:
            threading.Thread(target=state.load, name="engine-load", daemon=True).start()
        yield

    app = FastAPI(title=f"aod-rec-engine:{state.platform}", lifespan=lifespan)

    @app.get("/health")
    def health():
        if not state.ready:
            return JSONResponse({"ready": False, "reason": state.reason}, status_code=503)
        return {"ready": True, "platform": state.platform, "corpus_version": state.corpus_version,
                "engine_sha": state.engine_sha, "config_hash": state.config_hash, **memory_report()}

    @app.post("/engine/recommend", response_model=EngineResponse)
    def recommend(req: EngineRequest):
        if not state.ready:
            return JSONResponse({"error": "not_ready", "reason": state.reason}, status_code=503)
        if req.media is not None and not state.adapter.supports_media:
            return JSONResponse({"detail": f"{state.platform} 은 media 를 받지 않는다"}, status_code=422)
        t0 = time.perf_counter()
        try:
            with gate:
                res = state.adapter.recommend(k=req.k, seeds=req.seeds, disliked=req.disliked, excluded=req.excluded,
                                              seen=req.seen, media=req.media)
        except Busy:
            return JSONResponse({"error": "busy"}, status_code=503)
        items = [EngineItem(key=i.key, rank=n, dominant_seed=i.dominant_seed, episode_count=i.episode_count,
                            score=Score(final=i.final, sim=i.sim, factors=i.factors)) for n, i in enumerate(res.items)]
        return EngineResponse(platform=state.platform, items=items, exhausted=res.exhausted, dropped_seeds=res.dropped_seeds,
                              factor_schema=state.adapter.factor_schema, version=state.version(),
                              took_ms=int((time.perf_counter() - t0) * 1000))

    return app


def app_from_env() -> FastAPI:
    from aod_serving.engine.bootstrap import BASELINE_CORPORA, PLATFORMS
    platform = os.environ.get("PLATFORM")
    if platform not in PLATFORMS:
        raise SystemExit(f"PLATFORM={platform!r} — {PLATFORMS} 중 하나")
    corpus = os.environ.get("CORPUS_VERSION") or BASELINE_CORPORA[platform]
    state = EngineState(platform=platform, corpus_version=corpus, engine_sha=os.environ.get("GIT_SHA", "dev"),
                        loader=default_loader(platform, corpus, os.environ.get("SERVING_MODE", "dev")))
    return create_app(state, max_queue=int(os.environ.get("MAX_QUEUE", "8")))
