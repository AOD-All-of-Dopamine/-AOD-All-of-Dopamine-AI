"""엔진 HTTP 서비스 — 플랫폼 하나, 프로세스 하나 (REC_TAB_DESIGN §8-1·§8-6).

엔진 계산(적재 · 예열 · 모든 요청)은 **전용 계산 스레드 하나**에서만 돈다 — `EngineState` 가
`ThreadPoolExecutor(max_workers=1)` 로 들고 있고, 엔드포인트는 거기에 일을 맡기고 기다린다.
소유권이 아니라 **안전** 때문이다: pandas 3 의 문자열 컬럼은 PyArrow 가 뒤를 받치고, pyarrow 25 의
기본 Arrow 메모리 풀(mimalloc)은 Arrow 를 쓰던 스레드가 **종료**한 뒤 그 TLS 블록을 재활용한
새 스레드가 Arrow 할당을 하면 `mi_thread_init()` 에서 SIGSEGV 를 낸다. 예전 구조는 적재를
`engine-load` 스레드에서 하고(그 스레드는 적재가 끝나면 죽는다) 요청은 anyio `to_thread` 워커에서
처리해 그 전제를 매번 만들었다 — 웹툰은 첫 요청에서 100% 죽었다. 계산 스레드를 하나로 고정하면
Arrow 를 만지는 스레드가 프로세스 수명 내내 한 개뿐이라 전제 자체가 없어진다.
할당자 쪽도 `aod_serving.native` 가 `ARROW_DEFAULT_MEMORY_POOL=system` 으로 한 겹 더 막는다.

`/health` 는 계산 스레드를 쓰지 않는다 — 긴 요청이 도는 중에도 바로 답해야 하기 때문이다.
"""
from __future__ import annotations
import logging, os, threading, time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from aod_serving.common.models import EngineItem, EngineRequest, EngineResponse, Score, VersionInfo
from aod_serving.engine.contract import ArtifactError
from aod_serving.engine.overrides import ConfigError
from aod_serving.native import arrow_pool_backend, pool_warning

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


class Busy(Exception):
    """대기열이 가득 찼다 — 즉시 503 busy."""


class EngineState:
    """엔진 한 대의 상태 + **전용 계산 스레드 하나**.

    계산은 이 스레드에서만 돈다(적재 · 예열 · 모든 요청). 한 번에 한 요청만 계산하는 예전
    `_Gate` 의 성질은 `max_workers=1` 이 그대로 준다 — 파이썬 구간은 어차피 GIL 에 묶여 있어
    동시에 돌려도 총 처리량은 같고 개별 지연 · 최대 메모리만 나빠진다. 대기만 `max_queue` 로
    제한해 넘치면 즉시 거절한다(`Busy` → 503 busy). 실행 1 + 대기 `max_queue` 까지 받는다.

    `_pending` 은 **받아들인 요청 수**(실행 중 + 대기 중)다. 큐 길이를 executor 내부에서
    들여다보지 않고 입장에서만 세기 때문에, 예전 `_Gate` 주석이 걱정하던 "락이 막 풀렸는데
    대기자가 아직 못 깨어난 틈" 같은 경쟁 상태가 없다 — 증감이 전부 `_lock` 하나 밑에 있다.
    """

    def __init__(self, *, platform: str, corpus_version: str, engine_sha: str,
                 loader: Callable[[], tuple[object, str]], max_queue: int = 8,
                 queue_deadline_s: float = 1.5, now: Callable[[], float] = time.monotonic):
        self.platform, self.corpus_version, self.engine_sha, self._loader = platform, corpus_version, engine_sha, loader
        self.adapter = None; self.config_hash = ""; self.ready = False; self.reason = "loading"
        self.max_queue, self.queue_deadline_s, self._now = max_queue, queue_deadline_s, now
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-compute")
        self._lock = threading.Lock()
        self._pending = 0

    # ── 전용 계산 스레드 ──────────────────────────────────────────────
    def start_load(self) -> Future:
        """적재를 계산 스레드에 맡기고 **즉시** 돌아온다(기동 시). 그 동안 /health 는 503 loading."""
        return self._pool.submit(self._load)

    def load(self) -> None:
        """적재를 계산 스레드에서 돌리고 끝날 때까지 기다린다(테스트 · 동기 기동용)."""
        self._pool.submit(self._load).result()

    def call(self, fn: Callable, /, *args, timing: dict | None = None, **kwargs):
        """`fn` 을 계산 스레드에서 돌리고 결과를 돌려준다. 대기열이 넘치면 `Busy`.

        입장할 때 `deadline = now() + queue_deadline_s` 를 찍어 두고, 계산 스레드가 실제로 집을 때
        그 시각을 다시 재서 넘겼으면 `fn` 을 부르지 않고 바로 `Busy` — 라우터가 이미 포기한(1.5s)
        요청을 계산해 뒤에 줄선 다른 요청까지 물고 늘어지지 않는다. `timing` 을 주면 대기 시간(ms)을
        채워 준다(요청 로그용) — 이 인자는 `fn` 에 넘기지 않는다.
        """
        with self._lock:
            if self._pending > self.max_queue:       # 실행 1 + 대기 max_queue 까지
                raise Busy()
            self._pending += 1
        admitted_at = self._now()
        deadline = admitted_at + self.queue_deadline_s

        def run():
            started_at = self._now()
            if timing is not None:
                timing["queue_wait_ms"] = max(0.0, (started_at - admitted_at) * 1000)
            if started_at > deadline:                # 대기열에서 이미 마감을 넘겼다 — 계산하지 않는다
                raise Busy()
            return fn(*args, **kwargs)

        try:
            try:
                future = self._pool.submit(run)
            except RuntimeError:                     # 풀이 닫혔다(종료 중) — 새 일은 받지 않는다
                raise Busy() from None
            return future.result()
        finally:
            with self._lock:
                self._pending -= 1

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)   # SIGTERM 이 대기열을 다 비울 때까지 기다리지 않는다

    # ── 적재 ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        t0 = time.perf_counter()
        try:
            adapter, self.config_hash = self._loader()
            adapter.recommend(k=10, seeds=[adapter.first_key()])      # 예열 — 임베딩 파일을 미리 읽혀 둔다(§8-6)
            self.adapter, self.ready, self.reason = adapter, True, ""
            if (warning := pool_warning()):
                log.warning("%s", warning)
            log.info("%s 준비 완료 %.1fs (arrow_pool=%s)", self.platform, time.perf_counter() - t0, arrow_pool_backend())
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


def create_app(state: EngineState, *, load_in_background: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_in_background:
            state.start_load()          # 적재도 요청과 **같은** 계산 스레드에서 돈다
        yield
        state.shutdown()

    app = FastAPI(title=f"aod-rec-engine:{state.platform}", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # 어댑터·랭커 안의 예상 못 한 예외가 그대로 500 text/plain 으로 새 나가지 않게 — 트레이스는
        # 로그로만 남기고(요청 본문은 남기지 않는다), 응답은 형태를 갖춘 JSON 으로 돌려준다.
        log.exception("처리되지 않은 예외 — %s %s", request.method, request.url.path)
        return JSONResponse({"error": "internal"}, status_code=500)

    @app.get("/health")
    def health():
        # 계산 스레드를 쓰지 않는다 — 긴 추천이 도는 중에도 뒤에 줄서지 않고 바로 답한다.
        if not state.ready:
            return JSONResponse({"ready": False, "reason": state.reason}, status_code=503)
        return {"ready": True, "platform": state.platform, "corpus_version": state.corpus_version,
                "engine_sha": state.engine_sha, "config_hash": state.config_hash,
                "arrow_pool": arrow_pool_backend(), **memory_report()}

    @app.post("/engine/recommend", response_model=EngineResponse)
    def recommend(req: EngineRequest):
        if not state.ready:
            return JSONResponse({"error": "not_ready", "reason": state.reason}, status_code=503)
        if req.media is not None and not state.adapter.supports_media:
            return JSONResponse({"detail": f"{state.platform} 은 media 를 받지 않는다"}, status_code=422)
        t0 = time.perf_counter()
        timing: dict = {}
        try:
            res = state.call(state.adapter.recommend, k=req.k, seeds=req.seeds, disliked=req.disliked,
                             excluded=req.excluded, seen=req.seen, media=req.media, timing=timing)
        except Busy:
            return JSONResponse({"error": "busy"}, status_code=503)
        items = [EngineItem(key=i.key, rank=n, dominant_seed=i.dominant_seed, episode_count=i.episode_count,
                            score=Score(final=i.final, sim=i.sim, factors=i.factors)) for n, i in enumerate(res.items)]
        took_ms = int((time.perf_counter() - t0) * 1000)
        # 요청 로그 — 개수만 남긴다(시드·키·본문은 남기지 않는다). 1s 넘으면 눈에 띄게 WARNING.
        (log.warning if took_ms >= 1000 else log.info)(
            "recommend platform=%s k=%d seeds=%d disliked=%d excluded=%d seen=%d items=%d dropped=%d "
            "exhausted=%s tookMs=%d queueWaitMs=%d", state.platform, req.k, len(req.seeds), len(req.disliked),
            len(req.excluded), len(req.seen), len(items), len(res.dropped_seeds), res.exhausted, took_ms,
            int(timing.get("queue_wait_ms", 0)))
        return EngineResponse(platform=state.platform, items=items, exhausted=res.exhausted, dropped_seeds=res.dropped_seeds,
                              factor_schema=state.adapter.factor_schema, version=state.version(), took_ms=took_ms)

    return app


def app_from_env() -> FastAPI:
    from aod_serving.engine.bootstrap import BASELINE_CORPORA, PLATFORMS
    platform = os.environ.get("PLATFORM")
    if platform not in PLATFORMS:
        raise SystemExit(f"PLATFORM={platform!r} — {PLATFORMS} 중 하나")
    corpus = os.environ.get("CORPUS_VERSION") or BASELINE_CORPORA[platform]
    state = EngineState(platform=platform, corpus_version=corpus, engine_sha=os.environ.get("GIT_SHA", "dev"),
                        loader=default_loader(platform, corpus, os.environ.get("SERVING_MODE", "dev")),
                        max_queue=int(os.environ.get("MAX_QUEUE", "8")),
                        queue_deadline_s=int(os.environ.get("QUEUE_DEADLINE_MS", "1500")) / 1000)
    return create_app(state)
