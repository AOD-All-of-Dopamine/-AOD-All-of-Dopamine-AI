"""부하·메모리 관문 — compose 망 안에서 돈다: scripts/dev.sh net python -m aod_serving.tools.loadgate --out /tmp/gate.json

시드·seen 은 코퍼스 키에서 고정 난수로 뽑는다(요청 비용은 어떤 키냐와 거의 무관하고 개수에 비례한다).

엔진은 전용 계산 스레드 + 대기열(MAX_QUEUE=8, QUEUE_DEADLINE_MS=1500)을 쓴다 — 동시 20 에서는
대기가 마감을 넘긴 요청이 계산 없이 즉시 503 {"error":"busy"} 로 빠지는 것이 설계된 동작이다(빨리
포기하기). 그래서 상태 코드를 세 갈래로 나눈다: ok(200) · busy(503 busy) · other_err(그 외).
지연 백분위(p50/p95/max)는 ok 요청만으로 계산한다.

라우터 전체 탭은 엔진이 바쁘거나 느리면 200 인 채로 partial 목록을 채워 돌려준다(§8-5) — 그래서
라우터 쪽은 ok/busy/other_err 대신 ok(200) 수와 그 중 partial 이 비어있지 않은 응답 수, 그리고
503(engines_unavailable, 전부 실패) 수를 따로 센다.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

REC = Path("/rec")
CORPUS = {"steam": ("steam/artifacts/tags_full", "steam_appid"), "tmdb": ("tmdb/artifacts/tmdb_v1", "item_id"),
          "webtoon": ("webtoon/artifacts/wt_v1", "item_id"), "webnovel": ("webnovel/artifacts/wn_v6", "item_id")}
SEEDS, SEEN, CONC = (1, 10, 50), (0, 200, 500), (1, 5, 20)


def keys_of(platform: str) -> list[str]:
    import pandas as pd
    d, col = CORPUS[platform]
    return [str(k) for k in pd.read_parquet(REC / d / "corpus_index.parquet")[col].tolist()]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def cell_size(per_cell: int, conc: int) -> int:
    """셀당 요청 수 — `per_cell` 이상, 동시 수의 배수로 올림. 동시 20 은 최소 40(요구사항)."""
    minimum = max(per_cell, 40 if conc == 20 else 0)
    return -(-minimum // conc) * conc


def fire(http, url, bodies, conc, *, router=False):
    """`bodies` 를 동시성 `conc` 로 쏘고 상태를 센다.

    각 요청 결과는 (지연ms|None, 상태코드, busy 여부, partial 비었나 여부) 로 모은다 — 리스트
    append 는 GIL 아래 원자적이라 스레드 간 락 없이도 안전하다.
    """
    results: list[tuple[float | None, int, bool, bool]] = []

    def one(body):
        t0 = time.perf_counter()
        try:
            r = http.post(url, json=body)
        except httpx.HTTPError:
            results.append((None, -1, False, False))
            return
        ms = (time.perf_counter() - t0) * 1000
        busy = False
        partial = False
        if r.status_code == 503:
            try:
                busy = r.json().get("error") == "busy"
            except Exception:
                pass
        elif r.status_code == 200 and router:
            try:
                partial = bool(r.json().get("partial"))
            except Exception:
                pass
        results.append((ms, r.status_code, busy, partial))

    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(one, bodies))

    ok = [ms for ms, code, _, _ in results if code == 200]
    busy = sum(1 for _, _, b, _ in results if b)
    other_err = sum(1 for _, code, b, _ in results if code != 200 and not b)
    out = {"n": len(bodies), "ok": len(ok), "busy": busy, "other_err": other_err,
           "p50": pct(ok, .5), "p95": pct(ok, .95), "max": max(ok, default=None)}
    if router:
        out["partial"] = sum(1 for _, _, _, p in results if p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--platforms", default="steam,tmdb,webtoon,webnovel")
    ap.add_argument("--per-cell", type=int, default=12, help="셀당 요청 수(동시 수의 배수로 올림)")
    ap.add_argument("--skip-router", action="store_true")
    ap.add_argument("--skip-engines", action="store_true", help="엔진별 표는 건너뛰고 라우터 표만(엔진은 이미 따로 쟀을 때)")
    a = ap.parse_args(argv)
    rng = random.Random(20260919); http = httpx.Client(timeout=60.0, limits=httpx.Limits(max_connections=64))
    report = {"cells": [], "memory": {}, "router": []}
    pools = {p: keys_of(p) for p in a.platforms.split(",")}

    def body(p, n_seeds, n_seen):
        ks = rng.sample(pools[p], n_seeds + n_seen)
        return {"k": 50, "seeds": ks[:n_seeds], "seen": ks[n_seeds:]}

    if not a.skip_engines:
        for p in pools:
            url = f"http://rec-{p}:8000"
            # 예열 — 몇 건 먼저 쏴서 콜드스타트가 측정에 안 섞이게 한다.
            for _ in range(5):
                http.post(f"{url}/engine/recommend", json=body(p, 1, 0))
            report["memory"][p] = {"before": http.get(f"{url}/health").json().get("cgroup")}
            for n_seeds in SEEDS:
                for n_seen in SEEN:
                    for conc in CONC:
                        n = cell_size(a.per_cell, conc)
                        cell = fire(http, f"{url}/engine/recommend", [body(p, n_seeds, n_seen) for _ in range(n)], conc)
                        report["cells"].append({"platform": p, "seeds": n_seeds, "seen": n_seen, "conc": conc, **cell})
                        print(p, n_seeds, n_seen, conc, cell, flush=True, file=sys.stderr)
            report["memory"][p]["after"] = http.get(f"{url}/health").json().get("cgroup")

    if not a.skip_router and {"steam", "tmdb", "webnovel"} <= set(pools):
        for n_seeds in SEEDS:
            for conc in CONC:
                n = cell_size(a.per_cell, conc)
                bodies = [{"tab": "all", "k": 20, "buffer": 10, "seeds": {p: body(p, n_seeds, 0)["seeds"] for p in ("steam", "tmdb", "webnovel")}}
                          for _ in range(n)]
                cell = fire(http, "http://rec-router:8080/v1/recommend", bodies, conc, router=True)
                report["router"].append({"seeds": n_seeds, "conc": conc, **cell})
                print("router", n_seeds, conc, cell, flush=True, file=sys.stderr)

    text = json.dumps(report, ensure_ascii=False, indent=1)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)     # 표준출력에도 최종 보고서를 남긴다 — `net` 은 레포를 읽기 전용으로 마운트하므로
                     # 컨테이너 안 --out 파일은 컨테이너와 함께 사라진다; 호스트는 표준출력을 받는다.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
