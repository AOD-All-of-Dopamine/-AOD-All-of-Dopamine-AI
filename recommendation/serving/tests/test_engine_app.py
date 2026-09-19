"""엔진 HTTP — /health · /engine/recommend · 예열 · 한 번에 한 요청 (T11)."""
from __future__ import annotations
import threading, time
import pytest
from fastapi.testclient import TestClient
from aod_serving.engine.adapters.base import AdapterItem, AdapterResult
from aod_serving.engine.app import Busy, EngineState, create_app
from aod_serving.engine.contract import ArtifactError


class FakeAdapter:
    platform = "steam"; factor_schema = "steam.v1"; supports_media = False

    def __init__(self, delay=0.0): self.delay, self.calls, self.threads = delay, [], []
    def first_key(self): return "1"

    def recommend(self, **kw):
        self.calls.append(kw); self.threads.append(threading.current_thread().name); time.sleep(self.delay)
        if kw.get("media") and not self.supports_media: raise ValueError("steam 은 media 를 받지 않는다")
        items = [AdapterItem(key="240", dominant_seed="730", final=1.19, sim=0.58, factors={"quality": 1.0})]
        return AdapterResult(items, ["없는키"] if "없는키" in kw["seeds"] else [], exhausted=kw["k"] > 1)


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
                        loader=lambda: (adapter, "cfghash00000"))
    app = create_app(state, load_in_background=False, max_queue=1)
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
