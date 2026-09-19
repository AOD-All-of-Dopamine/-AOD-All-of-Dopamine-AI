"""Steam 제품 경로(`next_page`) 기준 목록을 만들고(--out), 지금 코드와 비교한다(--check), 지연을 잰다(--bench).

기준은 id 와 final_score 를 함께 담는다. 같은 머신에서의 비교는 **완전 일치**가 기준이고,
다른 머신에서는 점수가 같은 자리끼리 순서만 바뀐 것(`--allow-ties`)을 허용한다 — 동점의 순서는
pandas 기본 정렬(불안정)이 정하고 CPU 에 따라 달라진다.
"""
from __future__ import annotations
import argparse, json, platform as _pf, statistics, subprocess, sys, time
from datetime import datetime, timezone

from aod_serving.engine.bootstrap import enter_platform, rec_root
from aod_serving.tools.compare import pages_equal


def _cases(profiles) -> list[dict]:
    """(id, seeds, disliked, page_size, n_pages). 시드 1·10·50 은 프로필 시드를 순서대로 모아 만든다."""
    rows = list(profiles.sort_values("profile_order").itertuples(index=False))
    cases = [dict(id=f"profile:{r.profile_id}", seeds=[int(a) for a in r.liked_appids], disliked=[],
                  page_size=10, n_pages=5) for r in rows]
    pool: list[int] = []
    for r in rows:
        for a in r.liked_appids:
            if int(a) not in pool:
                pool.append(int(a))
    cases += [dict(id="synthetic:seeds1", seeds=pool[:1], disliked=[], page_size=10, n_pages=2),
              dict(id="synthetic:seeds10", seeds=pool[:10], disliked=[], page_size=10, n_pages=2),
              dict(id="synthetic:seeds50", seeds=pool[:50], disliked=[], page_size=10, n_pages=2)]
    for r in rows[:3]:
        cases.append(dict(id=f"k50:{r.profile_id}", seeds=[int(a) for a in r.liked_appids], disliked=[],
                          page_size=50, n_pages=1))
    return cases


def _run(case, next_page, comps) -> list[dict]:
    seen: set[int] = set(); pages = []
    for _ in range(case["n_pages"]):
        df = next_page(case["seeds"], seen_appids=set(seen), page_size=case["page_size"], components=comps,
                       disliked_appids=case["disliked"] or None)
        ids = [int(x) for x in df["steam_appid"]]
        pages.append({"ids": ids, "scores": [float(x) for x in df["final_score"]]})
        seen |= set(ids)
    return pages


def _diff(want: dict, got: dict, allow_ties: bool) -> list[str]:
    problems = []
    for cid, wp in want.items():
        gp = got[cid]
        if len(wp) != len(gp):   # zip 이 조용히 잘라내지 않게 먼저 본다
            problems.append(f"{cid} 쪽 수 다름: 기준 {len(wp)} · 결과 {len(gp)}")
        for i, (w, g) in enumerate(zip(wp, gp)):
            if w["ids"] == g["ids"] and w["scores"] == g["scores"]:
                continue
            if allow_ties and pages_equal(w["ids"], g["ids"], w["scores"], allow_ties=True):
                print(f"  동점 순서 차이(허용) {cid} p{i + 1}", file=sys.stderr); continue
            problems.append(f"{cid} p{i + 1}\n    want {w['ids']}\n    got  {g['ids']}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out"); g.add_argument("--check"); g.add_argument("--bench", action="store_true")
    ap.add_argument("--allow-ties", action="store_true")
    ap.add_argument("--cases", help="사례 id 에 이 문자열이 든 것만 돌린다(개발용 빠른 반복). "
                                    "--check 도 고른 사례만 비교한다 — 최종 확인은 필터 없이.")
    a = ap.parse_args(argv)
    if a.cases and (a.out or a.bench):
        print("--cases 는 --check 에서만 쓴다 — 기준 목록(--out)은 전체로만 만들고 "
              "--bench 는 고정 사례를 잰다", file=sys.stderr)
        return 2

    enter_platform("steam")
    import numpy, pandas as pd
    from src.personalized_retrieve import build_components, next_page
    comps = build_components()
    profiles = pd.read_parquet(rec_root() / "steam" / "artifacts" / "p1" / "profiles.parquet")
    cases = _cases(profiles)

    if a.bench:
        next_page(cases[0]["seeds"], seen_appids=set(), page_size=10, components=comps)      # 예열
        for cid in ("profile:" + profiles.sort_values("profile_order").profile_id.iloc[0], "synthetic:seeds10", "synthetic:seeds50"):
            c = next(x for x in cases if x["id"] == cid); ts = []
            for _ in range(5):
                t0 = time.perf_counter(); next_page(c["seeds"], seen_appids=set(), page_size=10, components=comps)
                ts.append((time.perf_counter() - t0) * 1000)
            print(f"{cid:28s} seeds={len(c['seeds']):2d}  median {statistics.median(ts):7.0f} ms  max {max(ts):7.0f} ms")
        return 0

    # 싫어요 사례: 첫 프로필의 1쪽 상위 2개를 싫어요로 넣고 다시 뽑는다
    first = cases[0]; dislike_id = "dislike:" + first["id"].split(":", 1)[1]
    if not a.cases or a.cases in dislike_id:
        p1 = _run({**first, "n_pages": 1}, next_page, comps)[0]["ids"]
        cases.append(dict(id=dislike_id, seeds=first["seeds"], disliked=p1[:2], page_size=10, n_pages=2))
    if a.cases:
        cases = [c for c in cases if a.cases in c["id"]]
        if not cases:
            print(f"--cases {a.cases!r} 와 맞는 사례가 없다", file=sys.stderr); return 2

    got = {}
    for n, c in enumerate(cases, 1):
        got[c["id"]] = _run(c, next_page, comps)
        print(f"[{n}/{len(cases)}] {c['id']}", file=sys.stderr)

    if a.out:
        try:
            sha = subprocess.run(["git", "-C", str(rec_root()), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        except OSError:
            sha = ""   # dev 이미지에는 git 이 없다 — 호스트 값으로 나중에 손으로 채운다
        doc = {"meta": {"what": "Steam next_page 제품 경로 기준 목록 (서빙 동일성 테스트용)",
                        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "machine": "팀장 PC · Docker python:3.12-slim (평가 서버 아님)", "cpu": _pf.processor() or _pf.machine(),
                        "numpy": numpy.__version__, "pandas": pd.__version__, "corpus": "tags_full", "code_sha": sha or "unknown"},
               "cases": [{**{k: c[k] for k in ("id", "seeds", "disliked", "page_size")}, "pages": got[c["id"]]} for c in cases]}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"저장 {a.out} — 사례 {len(cases)}개"); return 0

    want_doc = json.load(open(a.check, encoding="utf-8"))
    want = {c["id"]: c["pages"] for c in want_doc["cases"]}
    if a.cases:   # 고른 사례만 비교한다 — 나머지를 "사례 없음"으로 세지 않는다
        want = {k: v for k, v in want.items() if a.cases in k}
    missing = set(want) - set(got)
    problems = [f"사례 없음: {m}" for m in missing] + _diff({k: v for k, v in want.items() if k in got}, got, a.allow_ties)
    print(f"사례 {len(want)} · 어긋남 {len(problems)}")
    for p in problems[:10]:
        print(" ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
