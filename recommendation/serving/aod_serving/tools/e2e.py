"""compose 로 띄운 서비스의 끝-끝 점검 — L3 동일성 · 계약 · 장애.

    scripts/dev.sh net python -m aod_serving.tools.e2e                          # 정상 상태
    docker compose -p aod-rec stop rec-steam
    scripts/dev.sh net python -m aod_serving.tools.e2e --expect-partial steam    # 엔진 1개 중단
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import httpx

from aod_serving.router.mixing import load_m6

REC = Path("/rec")
ROUTER = "http://rec-router:8080"
ENGINE = {p: f"http://rec-{p}:8000" for p in ("steam", "tmdb", "webtoon", "webnovel")}


def seed_bundles(n: int) -> list[dict[str, list[str]]]:
    """평가 프로필에서 플랫폼별 시드를 뽑아 i번째끼리 묶는다(전체 탭용). TMDB 는 행 번호 → item_id."""
    import pandas as pd
    st = pd.read_parquet(REC / "steam/artifacts/p1/profiles.parquet").sort_values("profile_order")
    tm = pd.read_parquet(REC / "tmdb/artifacts/p1/profiles.parquet")
    tm_ids = pd.read_parquet(REC / "tmdb/artifacts/tmdb_v1/corpus_index.parquet")["item_id"].tolist()
    wn = pd.read_parquet(REC / "webnovel/artifacts/wn_v6/profiles_v6.parquet")
    wt = json.load(open(REC / "webtoon/eval/profiles.json", encoding="utf-8"))
    out = []
    for i in range(n):
        out.append({"steam": [str(int(a)) for a in st.iloc[i].liked_appids],
                    "tmdb": [tm_ids[int(r)] for r in tm.iloc[i].seed_rows],
                    "webnovel": [str(int(s)) for s in wn.iloc[i].seed_ids],
                    "webtoon": [str(int(s)) for s in wt[i]["seeds"]]})
    out[1]["steam"] = out[1]["steam"][:1]          # 시드 1개 플랫폼 → M6 쿼터 절반 경로
    return out


class Check:
    def __init__(self): self.n = 0; self.fail: list[str] = []
    def ok(self, cond: bool, msg: str):
        self.n += 1
        if not cond: self.fail.append(msg); print("  ✗", msg, file=sys.stderr)


def engine_call(http, p, body):
    r = http.post(f"{ENGINE[p]}/engine/recommend", json=body); r.raise_for_status(); return r.json()


def expected_all(http, seeds, seen, want):
    lists, n_seeds, episodes = {}, {}, {}
    for p, name in (("steam", "steam"), ("tmdb", "tmdb"), ("webnovel", "wn")):
        e = engine_call(http, p, {"k": 50, "seeds": seeds[p], "seen": seen.get(p, [])})
        lists[name] = [i["key"] for i in e["items"]]
        n = len(dict.fromkeys(seeds[p])) - len(e["droppedSeeds"])
        if n > 0: n_seeds[name] = n
        if p == "webnovel": episodes = {i["key"]: (i["episodeCount"] or 0) for i in e["items"]}
    mixed = load_m6()({k: v for k, v in lists.items() if k in n_seeds}, n_seeds, None, k=want, episodes=episodes)
    return [("webnovel" if p == "wn" else p, key) for p, key, _ in mixed]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect-partial", default="", help="중단해 둔 플랫폼(쉼표 구분). 주면 장애 점검만 한다")
    ap.add_argument("--bundles", type=int, default=10)
    a = ap.parse_args(argv)
    c = Check(); http = httpx.Client(timeout=30.0)
    bundles = seed_bundles(a.bundles)

    if a.expect_partial:
        down = a.expect_partial.split(",")
        r = http.post(f"{ROUTER}/v1/recommend", json={"tab": "all", "seeds": {p: bundles[0][p] for p in ("steam", "tmdb", "webnovel")}})
        body = r.json()
        c.ok(r.status_code == 200 and sorted(body["partial"]) == sorted(down), f"전체 탭 partial={body.get('partial')} (기대 {down})")
        c.ok(len(body["items"]) > 0 and not {i["platform"] for i in body["items"]} & set(down), "남은 플랫폼으로 목록이 채워진다")
        if "steam" in down:
            r = http.post(f"{ROUTER}/v1/recommend", json={"tab": "game", "seeds": {"steam": bundles[0]["steam"]}})
            c.ok(r.status_code == 503 and r.json()["error"] == "engines_unavailable", f"game 탭 503 (실제 {r.status_code})")
        h = http.get(f"{ROUTER}/health").json()
        c.ok(all(h["engines"][p]["ready"] is False for p in down), "라우터 /health 가 중단된 엔진을 보여 준다")
    else:
        for b in bundles:                                             # L3 — 전체 탭
            seeds = {p: b[p] for p in ("steam", "tmdb", "webnovel")}; seen: dict[str, list[str]] = {}
            for page in (1, 2):
                body = http.post(f"{ROUTER}/v1/recommend", json={"tab": "all", "k": 20, "buffer": 10, "seeds": seeds, "seen": seen}).json()
                got = [(i["platform"], i["key"]) for i in body["items"]]
                c.ok(got == expected_all(http, seeds, seen, 30), f"L3 전체 탭 {seeds['steam'][:1]}… p{page}")
                c.ok(body["partial"] == [], "partial 없음")
                for p, k in got: seen.setdefault(p, []).append(k)
        b = bundles[0]
        for tab, p, media in (("game", "steam", None), ("movie", "tmdb", "movie"), ("tv", "tmdb", "tv"),
                              ("webtoon", "webtoon", None), ("webnovel", "webnovel", None)):
            body = http.post(f"{ROUTER}/v1/recommend", json={"tab": tab, "k": 20, "buffer": 10, "seeds": {p: b[p]}}).json()
            e = engine_call(http, p, {"k": 30, "seeds": b[p], "media": media})
            c.ok([i["key"] for i in body["items"]] == [i["key"] for i in e["items"]] and len(body["items"]) == 30, f"단일 탭 {tab}")
            c.ok(all(i["score"]["final"] == j["score"]["final"] for i, j in zip(body["items"], e["items"])), f"{tab} 점수 전달")
        body = http.post(f"{ROUTER}/v1/recommend", json={"tab": "game", "seeds": {"steam": b["steam"] + ["없는키", "999999999"]}}).json()
        c.ok(body["droppedSeeds"] == {"steam": ["없는키", "999999999"]}, f"코퍼스 밖 시드 → droppedSeeds ({body['droppedSeeds']})")
        body = http.post(f"{ROUTER}/v1/recommend", json={"tab": "webtoon", "seeds": {}}).json()
        c.ok(body["items"] == [] and body["exhausted"] == {"webtoon": True}, "빈 시드 → 빈 200")
        c.ok(http.post(f"{ROUTER}/v1/recommend", json={"tab": "game", "seeds": {"steam": [730]}}).status_code == 422, "숫자 키 → 422")
        v = http.post(f"{ROUTER}/v1/recommend", json={"tab": "game", "seeds": {"steam": b["steam"]}}).json()["versions"]
        c.ok(v["engines"]["steam"]["corpus"] == "tags_full" and len(v["engines"]["steam"]["config"]) == 12, f"versions {v}")

    print(json.dumps({"checks": c.n, "failed": len(c.fail)}, ensure_ascii=False))
    return 1 if c.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
