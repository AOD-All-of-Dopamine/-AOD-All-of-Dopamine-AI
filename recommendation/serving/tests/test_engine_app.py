"""엔진 HTTP — /health · /engine/recommend · 예열 · 한 번에 한 요청 (T11)."""
from __future__ import annotations
import threading, time
import pytest
from fastapi.testclient import TestClient
from aod_serving.engine.adapters.base import AdapterItem, AdapterResult
from aod_serving.engine.app import Busy, EngineState, app_from_env, create_app
from aod_serving.engine.contract import ArtifactError


class FakeAdapter:
    platform = "steam"; factor_schema = "steam.v1"; supports_media = False

    def __init__(self, delay=0.0): self.delay, self.calls, self.threads = delay, [], []
    def first_key(self): return "1"

    def recommend(self, **kw):
        self.calls.append(kw); self.threads.append(threading.current_thread().name); time.sleep(self.delay)
        if kw.get("media") and not self.supports_media: raise ValueError("steam 은 media 를 받지 않는다")
        items = [AdapterItem(key="240", dominant_seed="730", final=1.19, sim=0.58, factors={"quality": 1.0})]
        dropped = ["없는키"] if "없는키" in kw["seeds"] else []
        used = len(set(kw["seeds"]) - set(dropped))
        return AdapterResult(items, dropped, exhausted=kw["k"] > 1, used_seeds=used)


class RaisingAdapter(FakeAdapter):
    """예열(1번째 호출)은 성공하고, 그 뒤 요청은 어댑터 안에서 예상 못 한 예외를 던진다."""

    def recommend(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return AdapterResult([], [], True)          # 예열
        raise RuntimeError("어댑터 내부 오류")


class FakeClock:
    """`EngineState(now=...)` 에 꽂는 가짜 시계 — 실제로 자지 않고 큐 마감을 검증한다."""

    def __init__(self, t: float = 0.0): self.t = t
    def __call__(self) -> float: return self.t
    def advance(self, dt: float) -> None: self.t += dt


def client(adapter=None, loader=None, **kw):
    adapter = adapter or FakeAdapter()
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=loader or (lambda: (adapter, "cfghash00000")))
    app = create_app(state, load_in_background=False, **kw)
    state.load()
    return TestClient(app), state, adapter


def test_health_ready_after_load_and_warmup():
    c, state, adapter = client()
    body = c.get("/health").json()
    assert body["ready"] is True and body["platform"] == "steam" and body["corpus_version"] == "tags_full"
    assert body["engine_sha"] == "abc123" and body["config_hash"] == "cfghash00000" and "rss_mb" in body
    assert adapter.calls[0]["seeds"] == ["1"]            # 예열 = first_key 로 한 번


def test_health_reports_artifact_failure_and_recommend_is_503():
    def boom(): raise ArtifactError("필수 파일 없음: dataset.parquet")
    c, _, _ = client(loader=boom)
    r = c.get("/health")
    assert r.status_code == 503 and r.json() == {"ready": False, "reason": "artifact_invalid: 필수 파일 없음: dataset.parquet"}
    assert c.post("/engine/recommend", json={"k": 5, "seeds": ["1"]}).status_code == 503


def test_recommend_contract():
    c, _, _ = client()
    r = c.post("/engine/recommend", json={"k": 1, "seeds": ["730", "없는키"], "seen": ["5"]})
    body = r.json()
    assert r.status_code == 200 and body["platform"] == "steam" and body["exhausted"] is False
    assert body["droppedSeeds"] == ["없는키"] and body["factorSchema"] == "steam.v1"
    assert body["version"] == {"sha": "abc123", "config": "cfghash00000", "corpus": "tags_full"}
    assert body["items"] == [{"key": "240", "rank": 0, "dominantSeed": "730",
                              "score": {"final": 1.19, "sim": 0.58, "factors": {"quality": 1.0}}, "episodeCount": None}]
    assert isinstance(body["tookMs"], int)
    assert body["usedSeeds"] == 1          # seeds=["730", "없는키"] 중 "없는키" 는 코퍼스 밖 — 1개만 랭커로 간다


@pytest.mark.parametrize("payload", [{"k": 0, "seeds": []}, {"k": 5, "seeds": [730]}, {"k": 5, "seeds": ["1"], "bogus": 1},
                                     {"k": 5, "seeds": ["1"], "media": "tv"}])
def test_invalid_requests_are_422(payload):
    c, _, _ = client()
    assert c.post("/engine/recommend", json=payload).status_code == 422


def test_requests_are_serialized_and_overflow_is_busy():
    # 직렬화 테스트 — 시간에 민감하다. Starlette 의 TestClient 는 스레드 간 공유가 안전하지 않을 수 있어
    # (내부 httpx.Client 연결 풀을 동시 접근) 스레드마다 같은 app 을 감싼 TestClient 를 새로 만든다.
    adapter = FakeAdapter(delay=0.3)
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=lambda: (adapter, "cfghash00000"), max_queue=1)
    app = create_app(state, load_in_background=False)
    state.load()
    codes = []

    def call():
        codes.append(TestClient(app).post("/engine/recommend", json={"k": 1, "seeds": ["1"]}).status_code)

    ts = [threading.Thread(target=call) for _ in range(4)]
    for t in ts: t.start(); time.sleep(0.05)
    for t in ts: t.join()
    assert sorted(codes) == [200, 200, 503, 503]      # 1 실행 + 1 대기, 나머지는 busy


# ── 전용 계산 스레드 (pyarrow/mimalloc SIGSEGV 회귀 방지) ─────────────────────────────
# 세그폴트의 전제는 "Arrow 를 쓰던 스레드가 죽고, 그 TLS 블록을 재활용한 새 스레드가 Arrow 를 쓴다"였다.
# 그래서 지키는 성질은 하나다 — **엔진 계산은 전부 같은 스레드 하나에서 돈다.**
# 진짜 크래시 재현·검증은 scripts/repro_segv.py + scripts/stress_engine.sh (도커 필요).

def test_all_engine_work_runs_on_one_dedicated_compute_thread():
    adapter, loaded_on = FakeAdapter(), []

    def loader():
        loaded_on.append(threading.current_thread().name)
        return adapter, "cfghash00000"

    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123", loader=loader)
    c = TestClient(create_app(state, load_in_background=False))
    state.load()
    for _ in range(3):
        assert c.post("/engine/recommend", json={"k": 1, "seeds": ["730"]}).status_code == 200

    names = set(loaded_on) | set(adapter.threads)     # 적재 + 예열 + 요청 3개
    assert len(names) == 1, f"엔진 계산이 여러 스레드에 흩어졌다: {names}"
    assert names.pop().startswith("engine-compute")
    assert threading.current_thread().name not in names   # 요청을 받은 스레드가 계산하지는 않았다


def test_call_rejects_exactly_above_capacity():
    """`_Gate` 를 대체한 입장 계산 — 시간에 기대지 않고 정확히 '실행 1 + 대기 max_queue' 인지 본다."""
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=lambda: (FakeAdapter(), "cfghash00000"), max_queue=2)
    block, admitted = threading.Event(), []

    def hold():
        admitted.append(1); block.wait(5)

    ts = [threading.Thread(target=lambda: state.call(hold)) for _ in range(3)]   # 실행 1 + 대기 2
    for t in ts: t.start()
    while len(admitted) < 1: time.sleep(0.01)
    while state._pending < 3: time.sleep(0.01)

    with pytest.raises(Busy):
        state.call(hold)                              # 4번째 = 정원 초과
    block.set()
    for t in ts: t.join()
    assert state._pending == 0


def test_health_answers_while_a_request_is_computing():
    """/health 는 계산 스레드를 쓰지 않는다 — 긴 추천 뒤에 줄서면 컨테이너 헬스체크가 죽는다."""
    adapter = FakeAdapter(delay=1.0)
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=lambda: (adapter, "cfghash00000"))
    app = create_app(state, load_in_background=False)
    state.load()

    started = threading.Event()

    def slow_call():
        started.set()
        TestClient(app).post("/engine/recommend", json={"k": 1, "seeds": ["730"]})

    t = threading.Thread(target=slow_call); t.start()
    started.wait(); time.sleep(0.2)                   # 계산이 확실히 돌고 있는 동안
    t0 = time.perf_counter()
    r = TestClient(app).get("/health")
    waited = time.perf_counter() - t0
    t.join()
    assert r.status_code == 200 and waited < 0.5, f"/health 가 계산 뒤에 줄섰다 ({waited:.2f}s)"


# ── 대기열 마감(§ 라우터가 1.5s 만에 포기한다) ─────────────────────────────────────

def test_queue_deadline_skips_stale_work_without_computing_it():
    """큐에서 마감을 넘긴 작업은 계산 스레드가 집어도 실행하지 않는다 — 503 busy, 어댑터는 안 불린다."""
    clock = FakeClock()
    adapter = FakeAdapter()
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=lambda: (adapter, "cfghash00000"), max_queue=2,
                        queue_deadline_s=1.0, now=clock)
    state.load()                                        # 예열 — 마감과 무관하게 항상 계산된다(가짜 시계라도)

    block, started, real_calls = threading.Event(), threading.Event(), []

    def blocking(**kw):
        started.set(); real_calls.append(kw); block.wait(5)
        return AdapterResult([], [], True)

    adapter.recommend = blocking                         # 이후 호출은 계산 스레드를 붙잡아 둔다

    t1 = threading.Thread(target=lambda: state.call(adapter.recommend, k=1, seeds=["1"]))
    t1.start(); assert started.wait(5)                    # 첫 작업이 계산 스레드를 잡고 블락 중

    results = {}

    def second():
        try:
            state.call(adapter.recommend, k=1, seeds=["1"]); results["second"] = "ok"
        except Busy:
            results["second"] = "busy"

    t2 = threading.Thread(target=second); t2.start()
    while state._pending < 2: time.sleep(0.01)            # 두 번째 작업이 대기열에 들어갈 때까지

    clock.advance(2.0)                                     # 마감(1.0s)을 이미 넘겼다 — 아직 대기 중일 때
    block.set()                                            # 첫 작업을 풀어 계산 스레드가 두 번째를 집게 한다
    t1.join(5); t2.join(5)

    assert results["second"] == "busy"
    assert len(real_calls) == 1, "마감을 넘긴 두 번째 작업이 실제로 계산됐다"


def test_queue_deadline_still_computes_job_within_deadline():
    """마감을 넘기지 않은 작업은 평소대로 계산된다."""
    clock = FakeClock()
    adapter = FakeAdapter()
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=lambda: (adapter, "cfghash00000"), queue_deadline_s=1.0, now=clock)
    state.load()
    calls_before = len(adapter.calls)
    r = state.call(adapter.recommend, k=1, seeds=["730"])
    assert isinstance(r, AdapterResult) and len(adapter.calls) == calls_before + 1


def test_recommend_returns_busy_when_deadline_expired_in_queue():
    """엔드포인트 관점: 마감을 넘긴 요청도 오버플로와 같은 503 busy 모양으로 나간다."""
    clock = FakeClock()
    adapter = FakeAdapter(delay=0.0)
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=lambda: (adapter, "cfghash00000"), max_queue=2, queue_deadline_s=1.0, now=clock)
    app = create_app(state, load_in_background=False)
    state.load()

    block, started = threading.Event(), threading.Event()
    orig = adapter.recommend

    def blocking(**kw):
        started.set(); block.wait(5); return orig(**kw)

    adapter.recommend = blocking
    t1 = threading.Thread(target=lambda: TestClient(app).post("/engine/recommend", json={"k": 1, "seeds": ["730"]}))
    t1.start(); assert started.wait(5)

    codes = {}

    def second():
        codes["r"] = TestClient(app).post("/engine/recommend", json={"k": 1, "seeds": ["730"]}).status_code

    t2 = threading.Thread(target=second); t2.start()
    while state._pending < 2: time.sleep(0.01)

    clock.advance(2.0)
    block.set()
    t1.join(5); t2.join(5)
    assert codes["r"] == 503


# ── 처리 안 된 예외 → 형태를 갖춘 500 ────────────────────────────────────────

def test_unexpected_adapter_exception_returns_shaped_500_not_bare_text():
    """`ServerErrorMiddleware` 는 등록된 500 핸들러로 응답을 보낸 뒤에도 항상 예외를 다시 던진다
    (Starlette 의 의도된 동작 — 서버 로그·테스트 클라이언트가 원하면 잡게). 그래서 TestClient 로
    실제 응답 바디를 보려면 `raise_server_exceptions=False` 가 필요하다 — 실제 클라이언트(uvicorn
    뒤)는 이미 보내진 응답을 그대로 받는다."""
    adapter = RaisingAdapter()
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=lambda: (adapter, "cfghash00000"))
    app = create_app(state, load_in_background=False)
    state.load()
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/engine/recommend", json={"k": 1, "seeds": ["1"]})
    assert r.status_code == 500 and r.json() == {"error": "internal"}
    assert r.headers["content-type"].startswith("application/json")


# ── 요청 로그 한 줄 (플랫폼·개수만 — 시드/본문 없음) ──────────────────────────────

def test_recommend_logs_one_line_with_counts_not_bodies(caplog):
    import logging
    c, _, _ = client()
    with caplog.at_level(logging.INFO, logger="aod.engine"):
        r = c.post("/engine/recommend", json={"k": 1, "seeds": ["730", "없는키"], "seen": ["5"]})
    assert r.status_code == 200
    lines = [rec for rec in caplog.records if rec.message.startswith("recommend platform=")]
    assert len(lines) == 1 and lines[0].levelno == logging.INFO
    msg = lines[0].message
    assert "platform=steam" in msg and "seeds=2" in msg and "seen=1" in msg and "tookMs=" in msg and "queueWaitMs=" in msg
    assert "730" not in msg and "없는키" not in msg     # 시드 값 자체는 로그에 없다


# ── 서빙 가능 목록 (spec3 §10) ────────────────────────────────────────────────

class CatalogAdapter(FakeAdapter):
    """`set_catalog` 이 언제·어느 스레드에서 불렸는지 기록하는 어댑터."""

    def __init__(self, delay=0.0):
        super().__init__(delay); self.catalog_calls = []

    def set_catalog(self, keys):
        self.catalog_calls.append((list(keys), len(self.calls), threading.current_thread().name))
        return len(keys), len(keys)


def catalog_client(keys_text="1\n2\n", *, source_kind="file", fetcher=None, refresh_s=600.0, adapter=None):
    from aod_serving.engine.catalog import CatalogLoader, CatalogSource
    adapter = adapter or CatalogAdapter()
    src = CatalogSource(source_kind, "keys.txt", refresh_s=refresh_s)
    loader = CatalogLoader(src, fetcher=fetcher or (lambda s: keys_text))
    state = EngineState(platform="steam", corpus_version="tags_full", engine_sha="abc123",
                        loader=lambda: (adapter, "cfghash00000"), catalog=loader)
    app = create_app(state, load_in_background=False)
    state.load()
    return TestClient(app), state, adapter, loader


def test_health_catalog_is_disabled_when_no_env_is_set():
    from aod_serving.engine.catalog import DISABLED
    c, _, _ = client()
    assert c.get("/health").json()["catalog"] == DISABLED


def test_catalog_is_applied_after_warmup_and_before_ready_on_the_compute_thread():
    """예열은 목록 없이(전체 코퍼스) 돌고, 목록은 `ready` 전에 적용된다 — healthy 가 되는 순간
    이미 걸려 있다. 1회차는 **계산 스레드**에서 돌아 Arrow 스레드 계약(§8-a)을 건드리지 않는다."""
    c, state, adapter, _ = catalog_client()
    (keys, calls_before, thread), = adapter.catalog_calls
    assert keys == ["1", "2"]
    assert calls_before == 1                       # 예열 1회가 이미 끝난 뒤 = 예열은 목록 전
    assert thread.startswith("engine-compute")
    body = c.get("/health").json()
    assert body["ready"] is True
    assert body["catalog"] == {"enabled": True, "size": 2, "matched": 2, "source": "file", "blocking_all": False,
                               "last_error": None, "loaded_at": body["catalog"]["loaded_at"]}
    assert body["catalog"]["loaded_at"] is not None


def test_catalog_failure_at_startup_does_not_block_ready_and_serves_unfiltered():
    def boom(src): raise OSError("백엔드가 없다")
    c, _, adapter, _ = catalog_client(fetcher=boom)
    body = c.get("/health").json()
    assert body["ready"] is True and adapter.catalog_calls == []
    assert body["catalog"]["loaded_at"] is None and "백엔드가 없다" in body["catalog"]["last_error"]
    assert c.post("/engine/recommend", json={"k": 1, "seeds": ["730"]}).status_code == 200


def test_periodic_refresh_runs_on_the_timer_thread_not_the_compute_thread():
    c, state, adapter, _ = catalog_client(refresh_s=0.02)
    for _ in range(200):
        if len(adapter.catalog_calls) >= 2: break
        time.sleep(0.01)
    state.shutdown()
    assert len(adapter.catalog_calls) >= 2
    assert adapter.catalog_calls[0][2].startswith("engine-compute")          # 기동 1회차
    assert all(t == "catalog-refresh" for _, _, t in adapter.catalog_calls[1:])   # 이후 갱신
    n = len(adapter.catalog_calls); time.sleep(0.08)
    assert len(adapter.catalog_calls) == n                                   # shutdown 이 타이머를 멈춘다


def test_recommend_logs_warning_when_slow(caplog):
    """tookMs 는 요청 전체를 재고, 1s 넘으면 WARNING. `time.perf_counter` 는 anyio/starlette 내부도
    같이 쓰므로 전역으로 갈아 끼우면 무관한 호출까지 어긋난다 — 대신 어댑터를 실제로 느리게 만든다."""
    import logging
    c, _, _ = client(adapter=FakeAdapter(delay=1.05))     # 예열(1회) + 이 요청(1회) 모두 1.05s 씩 걸린다
    with caplog.at_level(logging.INFO, logger="aod.engine"):
        r = c.post("/engine/recommend", json={"k": 1, "seeds": ["730"]})
    assert r.status_code == 200
    lines = [rec for rec in caplog.records if rec.message.startswith("recommend platform=")]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING


# ── SERVING_MODE 는 fail closed (I1) ──────────────────────────────────────────
# `app_from_env()` 는 `EngineState` 를 만들 뿐 그 자리에서 적재를 돌리지 않는다(적재는
# lifespan startup 이 계산 스레드에 맡긴다) — 그래서 `default_loader` 를 가짜로 갈아 끼우면
# 진짜 아티팩트·플랫폼 코드 없이 mode 파싱만 hermetic 하게 검증할 수 있다.

def _capture_mode(monkeypatch):
    import aod_serving.engine.app as app_module
    captured = {}

    def fake_loader(platform, corpus_version, mode):
        captured["platform"], captured["corpus_version"], captured["mode"] = platform, corpus_version, mode
        return lambda: (None, "hash")

    monkeypatch.setattr(app_module, "default_loader", fake_loader)
    return captured


def test_serving_mode_defaults_to_prod_when_unset(monkeypatch):
    monkeypatch.setenv("PLATFORM", "steam")
    monkeypatch.delenv("SERVING_MODE", raising=False)
    captured = _capture_mode(monkeypatch)
    app_from_env()
    assert captured["mode"] == "prod"


@pytest.mark.parametrize("raw", ["PROD ", " Prod", "prod", "PROD"])
def test_serving_mode_is_trimmed_and_lowercased_to_prod(monkeypatch, raw):
    monkeypatch.setenv("PLATFORM", "steam")
    monkeypatch.setenv("SERVING_MODE", raw)
    captured = _capture_mode(monkeypatch)
    app_from_env()
    assert captured["mode"] == "prod"


def test_serving_mode_dev_is_accepted(monkeypatch):
    monkeypatch.setenv("PLATFORM", "steam")
    monkeypatch.setenv("SERVING_MODE", "dev")
    captured = _capture_mode(monkeypatch)
    app_from_env()
    assert captured["mode"] == "dev"


def test_unknown_serving_mode_exits(monkeypatch):
    monkeypatch.setenv("PLATFORM", "steam")
    monkeypatch.setenv("SERVING_MODE", "staging")
    _capture_mode(monkeypatch)
    with pytest.raises(SystemExit, match="staging"):
        app_from_env()
