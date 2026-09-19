"""소켓 없는 세그폴트 재현 하네스 — 스레드 친화도(affinity) 가설을 빠르게 재는 도구.

    python -m aod_serving.engine  대신 이 스크립트를 컨테이너 안에서 돌린다:
      dev.sh run env PLATFORM=webtoon python scripts/repro_segv.py --mode fresh --iters 20

모드
  main      : 메인 스레드에서 적재·예열·요청 전부      (대조군 — 크래시 없어야 한다)
  fresh     : 적재는 스레드 A(적재 후 **종료**), 요청마다 새 스레드 (app.py 의 anyio to_thread 워커를 흉내)
  keepalive : fresh 와 같되 적재 스레드를 살려 둔다    (mimalloc 힙 abandon 가설의 대조군)
  pool      : 적재는 스레드 A, 요청은 ThreadPoolExecutor(max_workers=N)
  dedicated : 적재·예열·요청 전부 한 전용 스레드에서   (제안한 수정안)
  anyio     : 실제 anyio to_thread.run_sync 워커 풀 위에서 (uvicorn 과 같은 경로)

크래시하면 프로세스가 SIGSEGV 로 죽는다 — 호출한 쪽이 종료 코드(-11/139)로 판정한다.
끝까지 살아남으면 마지막 줄에 SURVIVED 를 찍는다.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def _build_adapter(platform: str, warmup: bool = True):
    from aod_serving.engine.adapters import ADAPTERS
    from aod_serving.engine.bootstrap import BASELINE_CORPORA, default_artifacts, enter_platform
    from aod_serving.engine.overrides import platform_defaults, resolve_config

    corpus = os.environ.get("CORPUS_VERSION") or BASELINE_CORPORA[platform]
    d = default_artifacts(platform, corpus)
    enter_platform(platform, artifacts=d)
    prod, post = platform_defaults(platform)
    cfg = resolve_config(platform, d, defaults=prod, post_defaults=post, corpus_version=corpus, mode="dev")
    adapter = ADAPTERS[platform](d, cfg)
    adapter.load()
    if warmup:
        adapter.recommend(k=10, seeds=[adapter.first_key()])  # 예열
    return adapter


def _seed_pool(adapter, n: int) -> list[str]:
    """어댑터가 아는 키에서 고르게 n 개. 정수 키 플랫폼은 `_known`, TMDB 는 `_item_ids`."""
    known = getattr(adapter, "_known", None)
    keys = sorted(str(k) for k in known) if known else [str(k) for k in getattr(adapter, "_item_ids", ())]
    if not keys:
        return [adapter.first_key()]
    step = max(1, len(keys) // max(1, n))
    return keys[::step][:n] or [adapter.first_key()]


def _one_call(adapter, seeds: list[str], i: int) -> int:
    res = adapter.recommend(k=10, seeds=[seeds[i % len(seeds)]])
    return len(res.items)


def run(mode: str, platform: str, iters: int, workers: int, warmup: bool = True) -> None:
    box: dict = {}

    def load_into_box():
        box["adapter"] = _build_adapter(platform, warmup=warmup)

    park = threading.Event()
    if mode == "main":
        load_into_box()
    elif mode == "dedicated":
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine-compute")
        pool.submit(load_into_box).result()
    elif mode == "keepalive":
        done = threading.Event()

        def load_and_park():
            load_into_box()
            done.set()
            park.wait()                      # 적재 스레드를 **종료시키지 않는다**

        threading.Thread(target=load_and_park, name="engine-load", daemon=True).start()
        done.wait()
    else:
        t = threading.Thread(target=load_into_box, name="engine-load", daemon=True)
        t.start()
        t.join()                             # 적재 스레드가 여기서 **죽는다**

    adapter = box["adapter"]
    seeds = _seed_pool(adapter, 12)
    print(f"[repro] mode={mode} platform={platform} iters={iters} seeds={len(seeds)} ready", flush=True)

    t0 = time.perf_counter()
    if mode == "main":
        for i in range(iters):
            n = _one_call(adapter, seeds, i)
            print(f"[repro] {i} items={n}", flush=True)
    elif mode in ("fresh", "keepalive"):
        for i in range(iters):
            out: dict = {}

            def target(i=i):
                out["n"] = _one_call(adapter, seeds, i)

            th = threading.Thread(target=target, name=f"fresh-{i}")
            th.start()
            th.join()
            print(f"[repro] {i} items={out.get('n')}", flush=True)
    elif mode == "pool":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i in range(iters):
                n = ex.submit(_one_call, adapter, seeds, i).result()
                print(f"[repro] {i} items={n}", flush=True)
    elif mode == "dedicated":
        for i in range(iters):
            n = pool.submit(_one_call, adapter, seeds, i).result()
            print(f"[repro] {i} items={n}", flush=True)
        pool.shutdown(wait=True)
    elif mode == "anyio":
        import anyio
        from anyio import to_thread

        async def main():
            for i in range(iters):
                n = await to_thread.run_sync(_one_call, adapter, seeds, i)
                print(f"[repro] {i} items={n}", flush=True)

        anyio.run(main)
    else:
        raise SystemExit(f"모르는 모드 {mode!r}")

    park.set()
    print(f"[repro] SURVIVED {iters} calls in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="fresh",
                   choices=["main", "fresh", "keepalive", "pool", "dedicated", "anyio"])
    p.add_argument("--platform", default=os.environ.get("PLATFORM", "webtoon"))
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-warmup", action="store_true", help="적재 스레드에서 예열을 건너뛴다")
    a = p.parse_args()
    if os.environ.get("PYTHONFAULTHANDLER"):
        import faulthandler
        faulthandler.enable()
    run(a.mode, a.platform, a.iters, a.workers, warmup=not a.no_warmup)
    sys.exit(0)
