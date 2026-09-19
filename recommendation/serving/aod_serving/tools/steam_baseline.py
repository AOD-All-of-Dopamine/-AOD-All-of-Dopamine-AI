"""Steam 제품 경로(`next_page`) 기준 목록을 만들고(--out), 지금 코드와 비교한다(--check), 지연을 잰다(--bench).

기준은 id 와 final_score 를 함께 담는다. 같은 머신에서의 비교는 **완전 일치**가 기준이고,
다른 머신에서는 점수가 같은 자리끼리 순서만 바뀐 것(`--allow-ties`)을 허용한다 — 동점의 순서는
pandas 기본 정렬(불안정)이 정하고 CPU 에 따라 달라진다.

CLI·git sha·쪽 비교는 `baseline_common` 에 있다 — webnovel 쪽과 규칙이 갈라지지 않게 한 곳에 둔다.
"""
from __future__ import annotations
import statistics, sys, time

from aod_serving.engine.bootstrap import enter_platform, rec_root
from aod_serving.tools import baseline_common as bc


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


def main(argv=None) -> int:
    a = bc.parse_args(__doc__, argv)
    if a is None:
        return 2

    enter_platform("steam")
    import pandas as pd
    from src.personalized_retrieve import build_components, next_page
    comps = build_components()
    profiles = pd.read_parquet(rec_root() / "steam" / "artifacts" / "p1" / "profiles.parquet")
    cases = _cases(profiles)

    if a.bench:
        next_page(cases[0]["seeds"], seen_appids=set(), page_size=10, components=comps)      # 예열
        first = "profile:" + profiles.sort_values("profile_order").profile_id.iloc[0]
        # k=50 은 **전체 탭이 실제로 쓰는 쪽 크기**다. 후보 풀이 rank_n=7,500 으로 커져
        # k=10(1,500)과 병목이 다르므로 같이 잰다. k=10 줄의 형식은 그대로 둔다.
        for cid, k in ((first, 10), ("synthetic:seeds10", 10), ("synthetic:seeds50", 10),
                       (first, 50), ("synthetic:seeds50", 50)):
            c = next(x for x in cases if x["id"] == cid); ts = []
            for _ in range(5):
                t0 = time.perf_counter(); next_page(c["seeds"], seen_appids=set(), page_size=k, components=comps)
                ts.append((time.perf_counter() - t0) * 1000)
            tag = "" if k == 10 else f"  k={k}"
            print(f"{cid:28s} seeds={len(c['seeds']):2d}{tag}  median {statistics.median(ts):7.0f} ms  max {max(ts):7.0f} ms")
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
        doc = {"meta": bc.meta("Steam next_page 제품 경로 기준 목록 (서빙 동일성 테스트용)",
                               "tags_full", bc.code_sha(rec_root())),
               "cases": [{**{k: c[k] for k in ("id", "seeds", "disliked", "page_size")}, "pages": got[c["id"]]} for c in cases]}
        bc.save(a.out, doc)
        print(f"저장 {a.out} — 사례 {len(cases)}개"); return 0

    return bc.check(a.check, got, a.cases, a.allow_ties)


if __name__ == "__main__":
    raise SystemExit(main())
