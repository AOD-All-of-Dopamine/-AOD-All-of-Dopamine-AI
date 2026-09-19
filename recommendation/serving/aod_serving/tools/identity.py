"""동일성 점검 — 서비스(어댑터) 결과 == 평가 경로 결과.

    python -m aod_serving.tools.identity --platform tmdb [--level l1|l2|all] [--allow-ties] [--limit N]

종료 코드 0 = 전부 일치. 마지막 줄에 JSON 요약을 찍는다. 플랫폼마다 별도 프로세스로 돌린다(최상위 `src` 충돌).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from aod_serving.engine.bootstrap import default_artifacts, enter_platform, rec_root
from aod_serving.tools.compare import pages_equal


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
                same = (w["ids"] == gi and w["scores"] == gs) or (allow_ties and pages_equal(w["ids"], gi, w["scores"], True))
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


def _direct(platform: str, adapter, seeds, disliked, excluded, seen, k):
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
                       components=adapter._comps)
        return [str(x) for x in df["item_id"]], [float(x) for x in df["final_score"]]
    if platform == "webtoon":
        df = adapter._fn.__self__.next_page([int(s) for s in seeds], k=k, seen=[int(x) for x in seen + excluded],
                                            disliked_ids=[int(x) for x in disliked] or None)
        return [str(int(x)) for x in df["item_id"]], [float(x) for x in df["final_score"]]
    from src.personalized_retrieve import next_page
    df = next_page([int(s) for s in seeds], seen_ids={int(x) for x in seen + excluded + disliked}, page_size=k,
                   components=adapter._comps)
    return [str(int(x)) for x in df["item_id"]], [float(x) for x in df["final_score"]]


def level1(platform: str, adapter, limit: int | None) -> tuple[int, list[str]]:
    problems: list[str] = []; n = 0
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
    return n, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", required=True, choices=["steam", "tmdb", "webtoon", "webnovel"])
    ap.add_argument("--level", default="all", choices=["l1", "l2", "all"])
    ap.add_argument("--allow-ties", action="store_true"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    adapter = _adapter(a.platform)
    summary = {"platform": a.platform}; failed = False
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
