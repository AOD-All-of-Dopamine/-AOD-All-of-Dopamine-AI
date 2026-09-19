"""TMDB 제품 경로(`next_page`) 기준 목록을 만들고(--out), 지금 코드와 비교한다(--check), 지연을 잰다(--bench).

기준은 id 와 final_score 를 함께 담는다. 같은 머신에서의 비교는 **완전 일치**가 기준이고,
다른 머신에서는 점수가 같은 자리끼리 순서만 바뀐 것(`--allow-ties`)을 허용한다 — 동점의 순서는
pandas 기본 정렬(불안정)이 정하고 CPU 에 따라 달라진다.

사례는 제품이 실제로 부르는 모양을 덮는다 — 전체 탭의 k=50, TMDB 탭의 k=30, **영화/드라마
탭(`media`)**, 시드 1/10/50, 그리고 `seen_rows` 가 크게 쌓인 뒤(500개)의 요청.
`media` 를 덮는 것이 TMDB 판만의 추가다: 탭 분리는 후보 풀을 자른 뒤 랭킹하므로
혼합 경로와 코드 경로가 다르다(`recommend` 의 `servable & (_media == media)`).

`profile:*` 52사례는 이미 있는 id 전용 기준 `crossdomain/x27_tmdb_pages.json` 과 **같은 설정·같은
호출**이다. 그래서 만들 때도 비교할 때도 그 파일과 쪽 단위로 대조한다(`_x27_sanity`) — 이 기준이
제품 경로에서 벗어나면 바로 드러난다.

CLI·git sha·쪽 비교는 `baseline_common` 에 있다 — steam·webnovel 쪽과 규칙이 갈라지지 않게 한 곳에 둔다.
"""
from __future__ import annotations
import json, statistics, sys, time

from aod_serving.engine.bootstrap import enter_platform, rec_root
from aod_serving.tools import baseline_common as bc

#: 기준을 만드는 코퍼스. `enter_platform("tmdb")` 이 잡아주는 확정 코퍼스와 같다.
CORPUS = "tmdb_v1"
#: `seen500` 사례가 미리 채우는 "이미 본 것"의 개수.
SEEN_PREFILL = 500
#: 쪽 크기·매체 사례를 만드는 앞쪽 프로필 수. 52개 전부 돌리면 사례가 300개를 넘어 반복이 느려진다.
N_TAB = 12
#: `seen500` 사례를 만드는 앞쪽 프로필 수.
N_SEEN = 3


def _cases(profiles, n_rows: int) -> list[dict]:
    """(id, seeds, seen, page_size, media, n_pages).

    · `profile:*`    52프로필 · k=10 · 5쪽 — `x27_tmdb_pages.json` 과 같은 모양이다.
    · `k30:*`/`k50:*` 앞 12프로필 · 2쪽 — 제품이 실제로 부르는 쪽 크기(TMDB 탭 30 · 전체 탭 50).
    · `movie:*`/`tv:*` 앞 12프로필 · k=30 · 2쪽 — 영화/드라마 탭(`media`)의 별도 경로.
    · `synthetic:seeds{1,10,50}` 프로필 시드를 순서대로 모은 것 · k=50 · 2쪽 — 시드 수에 따른 경로.
      `synthetic:seeds50:movie` 는 시드 50개 + 탭 분리가 겹친 최악의 모양이다(실사용 상한).
    · `seen500:*`    앞 3프로필 · k=50 · 1쪽 — 제외 집합이 클 때의 경로. 미리 채우는 500개는
      코퍼스 행 번호 앞에서 시드가 아닌 것을 순서대로 고른다(결정적).
    """
    rows = list(profiles.sort_values("profile_order").itertuples(index=False))
    seeds_of = {r.profile_id: [int(s) for s in r.seed_rows] for r in rows}

    cases = [dict(id=f"profile:{r.profile_id}", seeds=seeds_of[r.profile_id], seen=[],
                  page_size=10, media=None, n_pages=5) for r in rows]
    for k in (30, 50):
        cases += [dict(id=f"k{k}:{r.profile_id}", seeds=seeds_of[r.profile_id], seen=[],
                       page_size=k, media=None, n_pages=2) for r in rows[:N_TAB]]
    for m in ("movie", "tv"):
        cases += [dict(id=f"{m}:{r.profile_id}", seeds=seeds_of[r.profile_id], seen=[],
                       page_size=30, media=m, n_pages=2) for r in rows[:N_TAB]]

    pool: list[int] = []
    for r in rows:
        for s in seeds_of[r.profile_id]:
            if s not in pool:
                pool.append(s)
    cases += [dict(id=f"synthetic:seeds{n}", seeds=pool[:n], seen=[], page_size=50, media=None,
                   n_pages=2) for n in (1, 10, 50)]
    cases.append(dict(id="synthetic:seeds50:movie", seeds=pool[:50], seen=[], page_size=50,
                      media="movie", n_pages=2))

    for r in rows[:N_SEEN]:
        sd = set(seeds_of[r.profile_id])
        pre = [i for i in range(n_rows) if i not in sd][:SEEN_PREFILL]
        cases.append(dict(id=f"seen500:{r.profile_id}", seeds=seeds_of[r.profile_id], seen=pre,
                          page_size=50, media=None, n_pages=1))
    return cases


def _run(case, next_page, comps) -> list[dict]:
    """서빙과 같은 방식으로 쪽을 잇는다 — 돌려받은 `row`(코퍼스 행)를 `seen_rows` 에 누적."""
    seen: set[int] = set(case["seen"]); pages = []
    for _ in range(case["n_pages"]):
        df = next_page(case["seeds"], seen_rows=set(seen), page_size=case["page_size"],
                       components=comps, media=case["media"])
        pages.append({"ids": [str(x) for x in df["item_id"]],
                      "scores": [float(x) for x in df["final_score"]]})
        seen |= {int(x) for x in df["row"]}
    return pages


def _x27_sanity(got: dict) -> None:
    """`profile:*` 의 id 가 `x27_tmdb_pages.json` 과 같은지 본다 — 기준이 제품 경로임을 확인하는 대조군.

    x27 은 같은 확정값·같은 호출로 만든 **id 전용** 기준이다. 점수는 안 들고 있어서 여기서는
    id 만 본다. 한 쪽이라도 다르면 이 기준 목록 자체가 제품 경로에서 벗어난 것이다.
    """
    prof = {cid: p for cid, p in got.items() if cid.startswith("profile:")}
    if not prof:
        return
    base = json.load(open(rec_root() / "crossdomain" / "x27_tmdb_pages.json", encoding="utf-8"))
    n = same = 0
    for cid, pages in prof.items():
        want = base[cid.split(":", 1)[1]]
        n += max(len(want), len(pages))
        same += sum(1 for w, g in zip(want, pages) if w == g["ids"])
    print(f"x27 대조: profile:* 쪽 {same}/{n} 일치")
    assert same == n, "profile:* 가 x27_tmdb_pages.json 과 다르다 — 기준이 제품 경로가 아니다"


def main(argv=None) -> int:
    a = bc.parse_args(__doc__, argv)
    if a is None:
        return 2

    enter_platform("tmdb")
    import pandas as pd
    from src.personalized_retrieve import build_components, next_page
    from src.config import PRODUCTION
    # x27 빌더와 **같은 방식**으로 조립한다 — strategy 는 `next_page` 가 기본값으로 쓴다.
    comps = build_components(**{k: v for k, v in PRODUCTION.items() if k != "strategy"})
    # 프로필은 코퍼스가 아니라 평가 자산이라 `artifacts/p1/` 에 있다(코퍼스 폴더 옆).
    profiles = pd.read_parquet(rec_root() / "tmdb" / "artifacts" / "p1" / "profiles.parquet")
    cases = _cases(profiles, len(comps[3].dataset))

    if a.bench:
        by_id = {c["id"]: c for c in cases}
        # 시드 3개짜리 프로필(28/52 의 모양) · 시드 50개 합성 사례 · 시드 50 + 영화 탭
        first = next(c for c in cases if c["id"].startswith("profile:") and len(c["seeds"]) == 3)
        s50 = by_id["synthetic:seeds50"]
        plan = [(first["id"], first["seeds"], 10, None), (first["id"], first["seeds"], 30, None),
                (first["id"], first["seeds"], 50, None), (s50["id"], s50["seeds"], 50, None),
                (s50["id"] + ":movie", s50["seeds"], 50, "movie")]
        next_page(first["seeds"], seen_rows=set(), page_size=10, components=comps)      # 예열
        for label, seeds, k, media in plan:
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                next_page(seeds, seen_rows=set(), page_size=k, components=comps, media=media)
                ts.append((time.perf_counter() - t0) * 1000)
            print(f"{label:30s} seeds={len(seeds):2d} k={k:2d} media={str(media):5s}  "
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
    _x27_sanity(got)

    if a.out:
        doc = {"meta": bc.meta("TMDB next_page 제품 경로 기준 목록 (지연 개선 전 고정)",
                               CORPUS, bc.code_sha(rec_root())),
               "cases": [{**{k: c[k] for k in ("id", "seeds", "seen", "page_size", "media")},
                          "pages": got[c["id"]]} for c in cases]}
        bc.save(a.out, doc)
        print(f"저장 {a.out} — 사례 {len(cases)}개 · 쪽 {sum(len(v) for v in got.values())}개"); return 0

    return bc.check(a.check, got, a.cases, a.allow_ties)


if __name__ == "__main__":
    raise SystemExit(main())
