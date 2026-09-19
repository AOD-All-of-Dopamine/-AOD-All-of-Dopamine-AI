"""동일성 점검 — 서비스(어댑터) 결과 == 평가 경로 결과.

    python -m aod_serving.tools.identity --platform tmdb [--level l1|l2|all] [--allow-ties] [--limit N]
    python -m aod_serving.tools.identity --platform tmdb --catalog-check      # 서빙 가능 목록(spec3 §10)

종료 코드 0 = 전부 일치. 마지막 줄에 JSON 요약을 찍는다. 플랫폼마다 별도 프로세스로 돌린다(최상위 `src` 충돌).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from aod_serving.engine.bootstrap import default_artifacts, enter_platform, rec_root
from aod_serving.tools.compare import pages_equal

#: 기준 목록과의 점수 허용 오차. 순서 비교에는 쓰지 않는다 — id 목록은 항상 완전 일치여야 한다.
SCORE_TOL = 1e-6


def _adapter(platform: str):
    from aod_serving.engine.adapters import ADAPTERS
    from aod_serving.engine.contract import load_schema, validate_artifacts
    from aod_serving.engine.overrides import platform_defaults, resolve_config
    d = default_artifacts(platform); enter_platform(platform, artifacts=d)
    prod, post = platform_defaults(platform)
    cfg = resolve_config(platform, d, defaults=prod, post_defaults=post, corpus_version=d.name)
    validate_artifacts(d, load_schema(platform), corpus_version=d.name, production=cfg.production)
    a = ADAPTERS[platform](d, cfg); a.load()
    return a


def _pages(adapter, seeds, n_pages, k, disliked=()):
    """서빙과 같은 방식으로 페이지를 잇는다 — 돌려받은 키를 seen 에 누적."""
    seen, out = [], []
    for _ in range(n_pages):
        r = adapter.recommend(k=k, seeds=seeds, disliked=list(disliked), seen=seen)
        out.append(r.items); seen += [i.key for i in r.items]
    return out


def _profiles(platform: str, adapter):
    """(pid, 시드 키 목록[str]) — 플랫폼별 평가 프로필."""
    import pandas as pd
    R = rec_root()
    if platform == "steam":
        P = pd.read_parquet(R / "steam/artifacts/p1/profiles.parquet").sort_values("profile_order")
        return [(r.profile_id, [str(int(a)) for a in r.liked_appids]) for r in P.itertuples(index=False)]
    if platform == "tmdb":
        P = pd.read_parquet(R / "tmdb/artifacts/p1/profiles.parquet")
        return [(r.profile_id, [adapter._item_ids[int(x)] for x in r.seed_rows]) for r in P.itertuples(index=False)]
    if platform == "webtoon":
        return [(p["pid"], [str(int(s)) for s in p["seeds"]]) for p in json.load(open(R / "webtoon/eval/profiles.json", encoding="utf-8"))]
    P = pd.read_parquet(R / "webnovel/artifacts/wn_v6/profiles_v6.parquet")
    return [(r.profile_id, [str(int(s)) for s in r.seed_ids]) for r in P.itertuples(index=False)]


def level2(platform: str, adapter, allow_ties: bool, limit: int | None) -> tuple[int, list[str]]:
    R = rec_root(); problems: list[str] = []; n = 0
    if platform == "steam":
        doc = json.load(open(R / "steam/eval/s3_pages.json", encoding="utf-8"))
        for c in doc["cases"][:limit]:
            got = _pages(adapter, [str(s) for s in c["seeds"]], len(c["pages"]), c["page_size"], [str(d) for d in c["disliked"]])
            for i, (w, g) in enumerate(zip(c["pages"], got)):
                n += 1
                gi, gs = [int(x.key) for x in g], [x.final for x in g]
                # 목록(id·순서)은 완전 일치, 점수는 SCORE_TOL 안. BLAS 스레드 수가 다르면 행렬 곱의 누적 순서가 달라져
                # 점수 끝자리가 바뀐다(실측: 스레드 2 에서 271쪽 중 7쪽, 순서가 바뀐 쪽은 0). 같은 머신·같은 설정에서의
                # 비트 단위 비교는 `steam_baseline --check` 가 맡는다.
                close = len(w["scores"]) == len(gs) and all(abs(a - b) <= SCORE_TOL for a, b in zip(w["scores"], gs))
                same = (w["ids"] == gi and close) or (allow_ties and pages_equal(w["ids"], gi, w["scores"], True))
                if not same: problems.append(f"{c['id']} p{i + 1}: want {w['ids']} got {gi}")
        return n, problems

    base = json.load(open(R / {"tmdb": "crossdomain/x27_tmdb_pages.json", "webtoon": "webtoon/eval/t10_pages.json",
                               "webnovel": "webnovel/eval/w6_pages.json"}[platform], encoding="utf-8"))
    prefix = {"tmdb": "", "webtoon": "n0|", "webnovel": "n1|"}[platform]
    norm = str
    if platform == "webnovel":
        # 같은 작품의 판본(본편 ↔ 맛보기판)은 임베딩·점수가 완전히 같아 순서가 CPU 마다 다르다 → 작품 키로 비교한다(스펙 §2)
        ds = adapter._comps[3].dataset
        work = {str(int(i)): (str(nm).strip(), str(au).strip()) for i, nm, au in zip(ds.index, ds["name"], ds["author"])}
        norm = lambda k: work[str(k)]
    for pid, seeds in _profiles(platform, adapter)[:limit]:
        want = base[prefix + pid]
        got = _pages(adapter, seeds, len(want), 10)
        for i, (w, g) in enumerate(zip(want, got)):
            n += 1
            if [norm(x) for x in w] != [norm(x.key) for x in g]:
                problems.append(f"{pid} p{i + 1}: want {w} got {[x.key for x in g]}")
    return n, problems


def _direct(platform: str, adapter, seeds, disliked, excluded, seen, k, media: str | None = None):
    """평가 경로를 **어댑터를 거치지 않고** 직접 부른다 — §8-3 의 합치기 규칙을 여기서 독립적으로 다시 적는다."""
    if platform == "steam":
        from src.personalized_retrieve import next_page
        df = next_page([int(s) for s in seeds], seen_appids={int(x) for x in seen + excluded}, page_size=k,
                       components=adapter._comps, disliked_appids=[int(x) for x in disliked] or None)
        return [str(int(x)) for x in df["steam_appid"]], [float(x) for x in df["final_score"]]
    if platform == "tmdb":
        from src.personalized_retrieve import next_page
        row = adapter._row_of
        df = next_page([row[s] for s in seeds], seen_rows={row[x] for x in seen + excluded + disliked}, page_size=k,
                       components=adapter._comps, media=media)
        return [str(x) for x in df["item_id"]], [float(x) for x in df["final_score"]]
    if platform == "webtoon":
        df = adapter._fn.__self__.next_page([int(s) for s in seeds], k=k, seen=[int(x) for x in seen + excluded],
                                            disliked_ids=[int(x) for x in disliked] or None)
        return [str(int(x)) for x in df["item_id"]], [float(x) for x in df["final_score"]]
    from src.personalized_retrieve import next_page
    df = next_page([int(s) for s in seeds], seen_ids={int(x) for x in seen + excluded + disliked}, page_size=k,
                   components=adapter._comps)
    return [str(int(x)) for x in df["item_id"]], [float(x) for x in df["final_score"]]


#: L1 어댑터 결과의 첫 항목이 실어야 하는 factors 키 — 플랫폼별 `factor_schema` 와 어긋나면 문제로 센다.
EXPECTED_FACTOR_KEYS = {"steam": {"rec_pct", "quality", "tag_fit", "has_mc"}, "tmdb": set(),
                        "webtoon": set(), "webnovel": {"interest_pct"}}


def level1(platform: str, adapter, limit: int | None) -> tuple[int, list[str]]:
    problems: list[str] = []; n = 0
    expected_factors = EXPECTED_FACTOR_KEYS[platform]
    for pid, seeds in _profiles(platform, adapter)[: (limit or 5)]:
        first = [i.key for i in adapter.recommend(k=30, seeds=seeds).items]
        if len(first) < 15:
            continue
        disliked, excluded, seen = first[:2], first[2:5], first[5:15]
        got = adapter.recommend(k=30, seeds=seeds, disliked=disliked, excluded=excluded, seen=seen)
        want_ids, want_scores = _direct(platform, adapter, seeds, disliked, excluded, seen, 30)
        n += 1
        if [i.key for i in got.items] != want_ids or [i.final for i in got.items] != want_scores:
            problems.append(f"{pid}: want {want_ids[:10]}… got {[i.key for i in got.items][:10]}…")
        if got.items and set(got.items[0].factors) != expected_factors:
            problems.append(f"{pid}: factors {sorted(got.items[0].factors)} ≠ {sorted(expected_factors)}")
    if platform == "tmdb":
        n2, p2 = _level1_tmdb_media(adapter, limit)
        n += n2; problems += p2
    return n, problems


def _level1_tmdb_media(adapter, limit: int | None) -> tuple[int, list[str]]:
    """TMDB 전용 — L1 의 기본 경로는 media(영화/드라마 탭)를 한 번도 안 거친다. 어댑터 vs
    직접 `next_page(..., media=…)` 를 movie·tv 둘 다 비교한다."""
    problems: list[str] = []; n = 0
    for pid, seeds in _profiles("tmdb", adapter)[: (limit or 5)]:
        first = [i.key for i in adapter.recommend(k=30, seeds=seeds).items]
        if len(first) < 15:
            continue
        disliked, excluded, seen = first[:2], first[2:5], first[5:15]
        for media in ("movie", "tv"):
            got = adapter.recommend(k=30, seeds=seeds, disliked=disliked, excluded=excluded, seen=seen, media=media)
            want_ids, want_scores = _direct("tmdb", adapter, seeds, disliked, excluded, seen, 30, media=media)
            n += 1
            if [i.key for i in got.items] != want_ids or [i.final for i in got.items] != want_scores:
                problems.append(f"{pid} media={media}: want {want_ids[:10]}… got {[i.key for i in got.items][:10]}…")
    return n, problems


#: 실제 코퍼스로 돌리는 목록 점검의 크기 — 프로필 5개 × 3쪽 × k=30 (spec3 §10).
CATALOG_PAGES, CATALOG_K, CATALOG_STRIDE = 3, 30, 7


def catalog_check(platform: str, adapter, limit: int | None) -> tuple[int, list[str]]:
    """서빙 가능 목록을 **실제 코퍼스**에 걸고 네 가지를 본다(spec3 §10).

      (a) 돌려준 키가 전부 목록 안이다
      (b) 쪽이 비지 않고, 쪽 사이에 중복이 없다
      (c) 목록 = 코퍼스 전체 이면 목록을 끈 것과 **완전히** 같다(id·점수)
      (d) 목록 밖 시드도 그대로 동작한다 — 시드는 목록과 무관하다

    목록은 정렬한 코퍼스 키의 7번째마다로 **결정적으로** 만든다(기계마다 같은 목록).
    """
    problems: list[str] = []; n = 0
    keys = sorted(adapter.corpus_keys())
    every_nth = keys[::CATALOG_STRIDE]
    allowed = set(every_nth)
    profiles = _profiles(platform, adapter)[: (limit or 5)]
    pages = lambda seeds: _pages(adapter, seeds, CATALOG_PAGES, CATALOG_K)
    flat = lambda ps: [[(i.key, i.final) for i in p] for p in ps]

    # (c) 목록이 코퍼스 전체면 결과가 기능 OFF 와 비트 단위로 같아야 한다
    adapter.set_catalog(None)
    off = {pid: flat(pages(seeds)) for pid, seeds in profiles}
    size, matched = adapter.set_catalog(keys)
    if matched != len(keys):
        problems.append(f"(c) 코퍼스 전체를 넣었는데 matched={matched} ≠ 코퍼스 {len(keys)}")
    for pid, seeds in profiles:
        n += 1
        if flat(pages(seeds)) != off[pid]:
            problems.append(f"(c) {pid}: 목록=코퍼스 전체 인데 목록 없음과 결과가 다르다")

    # (a)(b) 목록 = 7번째마다
    adapter.set_catalog(every_nth)
    for pid, seeds in profiles:
        n += 1
        got = pages(seeds); keys_seen: list[str] = []
        for i, p in enumerate(got, 1):
            if not p:
                problems.append(f"(b) {pid} p{i}: 빈 쪽")
            if bad := [x.key for x in p if x.key not in allowed]:
                problems.append(f"(a) {pid} p{i}: 목록 밖 {bad[:5]}")
            keys_seen += [x.key for x in p]
        if len(keys_seen) != len(set(keys_seen)):
            problems.append(f"(b) {pid}: 쪽 사이 중복")

    # (d) 목록 밖 시드
    outside = next(k for k in keys if k not in allowed)
    n += 1
    r = adapter.recommend(k=CATALOG_K, seeds=[outside])
    if not r.items or r.dropped_seeds:
        problems.append(f"(d) 목록 밖 시드 {outside}: items={len(r.items)} dropped={r.dropped_seeds}")
    if bad := [x.key for x in r.items if x.key not in allowed]:
        problems.append(f"(d) 목록 밖 시드 결과에 목록 밖 항목 {bad[:5]}")

    adapter.set_catalog(None)
    return n, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", required=True, choices=["steam", "tmdb", "webtoon", "webnovel"])
    ap.add_argument("--level", default="all", choices=["l1", "l2", "all"])
    ap.add_argument("--allow-ties", action="store_true"); ap.add_argument("--limit", type=int)
    ap.add_argument("--catalog-check", action="store_true",
                    help="서빙 가능 목록만 점검한다(L1·L2 는 건너뛴다) — spec3 §10")
    a = ap.parse_args(argv)
    adapter = _adapter(a.platform)
    summary = {"platform": a.platform}; failed = False
    if a.catalog_check:
        n, problems = catalog_check(a.platform, adapter, a.limit)
        summary["catalog"] = {"compared": n, "mismatch": len(problems)}
        failed = bool(problems) or n == 0
        for p in problems[:5]:
            print(f"[catalog] {p}", file=sys.stderr)
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if failed else 0
    for name, run in (("l1", lambda: level1(a.platform, adapter, a.limit)),
                      ("l2", lambda: level2(a.platform, adapter, a.allow_ties, a.limit))):
        if a.level in (name, "all"):
            n, problems = run()
            summary[name] = {"compared": n, "mismatch": len(problems)}
            failed |= bool(problems) or n == 0
            for p in problems[:5]:
                print(f"[{name}] {p}", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
