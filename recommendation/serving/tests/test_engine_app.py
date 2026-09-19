"""엔진 HTTP — /health · /engine/recommend · 예열 · 한 번에 한 요청 (T11)."""
from __future__ import annotations
import threading, time
import pytest
from fastapi.testclient import TestClient
from aod_serving.engine.adapters.base import AdapterItem, AdapterResult
from aod_serving.engine.app import EngineState, create_app
from aod_serving.engine.contract import ArtifactError


class FakeAdapter:
    platform = "steam"; factor_schema = "steam.v1"; supports_media = False

    def __init__(self, delay=0.0): self.delay, self.calls = delay, []
    def first_key(self): return "1"

    def recommend(self, **kw):
        self.calls.append(kw); time.sleep(self.delay)
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
