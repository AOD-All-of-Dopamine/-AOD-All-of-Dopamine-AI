"""웹소설 제품 경로(`next_page`) 기준 목록을 만들고(--out), 지금 코드와 비교한다(--check), 지연을 잰다(--bench).

기준은 id 와 final_score 를 함께 담는다. 같은 머신에서의 비교는 **완전 일치**가 기준이고,
다른 머신에서는 점수가 같은 자리끼리 순서만 바뀐 것(`--allow-ties`)을 허용한다 — 동점의 순서는
pandas 기본 정렬(불안정)이 정하고 CPU 에 따라 달라진다. 웹소설은 **같은 작품의 판본**(본편 ↔
맛보기판)이 임베딩까지 같아 동점이 특히 흔하다. 그래서 서빙 동일성 도구(`identity`)는 평가
목록(w6_pages)과 작품 키로 비교하지만, 이 파일은 지연 개선이 결과를 바꾸지 않았는지 보는 것이
목적이라 **같은 머신·같은 코퍼스에서 id 까지 완전히 같아야 한다.**

사례는 제품이 실제로 부르는 모양을 덮는다 — 전체 탭이 쓰는 k=50, 웹소설 탭의 k=30, 시드 1/10/50,
그리고 `seen_ids` 가 크게 쌓인 뒤(500개)의 요청.

CLI·git sha·쪽 비교는 `baseline_common` 에 있다 — steam 쪽과 규칙이 갈라지지 않게 한 곳에 둔다.
"""
from __future__ import annotations
import statistics, sys, time

from aod_serving.engine.bootstrap import enter_platform, rec_root
from aod_serving.tools import baseline_common as bc

#: 기준을 만드는 코퍼스. `enter_platform("webnovel")` 이 잡아주는 확정 코퍼스와 같다.
CORPUS = "wn_v6"
#: `seen500` 사례가 미리 채우는 "이미 본 것"의 개수.
SEEN_PREFILL = 500


def _cases(profiles, index) -> list[dict]:
    """(id, seeds, seen, page_size, n_pages).

    · `profile:*`   52프로필 · k=10 · 3쪽 — 평가 목록(w6_pages `n1|*`)과 같은 모양이다.
    · `k30:*`/`k50:*` 앞 12프로필 · 2쪽 — 제품이 실제로 부르는 쪽 크기(웹소설 탭 30 · 전체 탭 50).
    · `synthetic:seeds{1,10,50}` 프로필 시드를 순서대로 모은 것 · k=50 · 2쪽 — 시드 수에 따른 경로.
    · `seen500:*`   앞 3프로필 · k=50 · 1쪽 — 제외 집합이 클 때의 경로. 미리 채우는 500개는
      `corpus_index.parquet` 의 앞에서 시드가 아닌 것을 순서대로 고른다(결정적).
    """
    rows = list(profiles.sort_values("profile_order").itertuples(index=False))
    seeds_of = {r.profile_id: [int(s) for s in r.seed_ids] for r in rows}

    cases = [dict(id=f"profile:{r.profile_id}", seeds=seeds_of[r.profile_id], seen=[],
                  page_size=10, n_pages=3) for r in rows]
    for k in (30, 50):
        cases += [dict(id=f"k{k}:{r.profile_id}", seeds=seeds_of[r.profile_id], seen=[],
                       page_size=k, n_pages=2) for r in rows[:12]]

    pool: list[int] = []
    for r in rows:
        for s in seeds_of[r.profile_id]:
            if s not in pool:
                pool.append(s)
    cases += [dict(id=f"synthetic:seeds{n}", seeds=pool[:n], seen=[], page_size=50, n_pages=2)
              for n in (1, 10, 50)]

    corpus = [int(x) for x in index["item_id"]]
    for r in rows[:3]:
        sd = set(seeds_of[r.profile_id])
        pre = [i for i in corpus if i not in sd][:SEEN_PREFILL]
        cases.append(dict(id=f"seen500:{r.profile_id}", seeds=seeds_of[r.profile_id], seen=pre,
                          page_size=50, n_pages=1))
    return cases


def _run(case, next_page, comps) -> list[dict]:
    """서빙과 같은 방식으로 쪽을 잇는다 — 돌려받은 id 를 `seen_ids` 에 누적."""
    seen: set[int] = set(case["seen"]); pages = []
    for _ in range(case["n_pages"]):
        df = next_page(case["seeds"], seen_ids=set(seen), page_size=case["page_size"], components=comps)
        ids = [int(x) for x in df["item_id"]]
        pages.append({"ids": ids, "scores": [float(x) for x in df["final_score"]]})
        seen |= set(ids)
    return pages


def main(argv=None) -> int:
    a = bc.parse_args(__doc__, argv)
    if a is None:
        return 2

    art = enter_platform("webnovel")
    import pandas as pd
    from src.personalized_retrieve import build_components, next_page
    comps = build_components()
    profiles = pd.read_parquet(art / "profiles_v6.parquet")
    index = pd.read_parquet(art / "corpus_index.parquet")
    cases = _cases(profiles, index)

    if a.bench:
        first = f"profile:{profiles.sort_values('profile_order').profile_id.iloc[0]}"
        next_page(cases[0]["seeds"], seen_ids=set(), page_size=10, components=comps)      # 예열
        for cid, k in ((first, 10), (first, 30), (first, 50), ("synthetic:seeds50", 50)):
            c = next(x for x in cases if x["id"] == cid); ts = []
            for _ in range(5):
                t0 = time.perf_counter(); next_page(c["seeds"], seen_ids=set(), page_size=k, components=comps)
                ts.append((time.perf_counter() - t0) * 1000)
            print(f"{cid:28s} seeds={len(c['seeds']):2d} k={k:2d}  "
                  f"median {statistics.median(ts):7.0f} ms  max {max(ts):7.0f} ms")
        return 0

    if a.cases:
        cases = [c for c in cases if a.cases in c["id"]]
        if not cases:
            print(f"--cases {a.cases!r} 와 맞는 사례가 없다", file=sys.stderr); return 2

    got = {}
    for n, c in enumerate(cases, 1):
        got[c["id"]] = _run(c, next_page, comps)
        print(f"[{n}/{len(cases)}] {c['id']}", file=sys.stderr)

    if a.out:
        doc = {"meta": bc.meta("웹소설 next_page 제품 경로 기준 목록 (지연 개선 전 고정)",
                               CORPUS, bc.code_sha(rec_root())),
               "cases": [{**{k: c[k] for k in ("id", "seeds", "seen", "page_size")}, "pages": got[c["id"]]}
                         for c in cases]}
        bc.save(a.out, doc)
        print(f"저장 {a.out} — 사례 {len(cases)}개"); return 0

    return bc.check(a.check, got, a.cases, a.allow_ties)


if __name__ == "__main__":
    raise SystemExit(main())
