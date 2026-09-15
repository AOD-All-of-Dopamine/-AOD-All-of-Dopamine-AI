"""X-26 집계 — 사전등록 eval/x26_preregister.md. 플랫폼별 페이지 1·3·5 3역할 다수결 적합률과 판정.

    python crossdomain/x26_score.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
REF = {"steam(X-20)": (0.942, 0.842), "webnovel(X-20)": (0.825, 0.742)}

def load(plat):
    K = json.load(open(OUT / f"x26_{plat}_key.json")); key = {t["id"]: t for t in K["todo"]}
    G = {x: {} for x in "ABC"}
    for x in "ABC":
        for f in sorted(glob.glob(str(OUT / f"x26/{plat}/{x}_*.json"))):
            for i, g in json.load(open(f)).items():
                if i in key: G[x][(key[i]["pid"], key[i]["item"])] = int(g)
    return K["rows"], G, len(K["todo"])

bad = False
for plat in ("tmdb", "webtoon"):
    rows, G, n = load(plat)
    miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
    print(f"\n## {plat} — 슬롯 {len(rows)} · 채점 A {len(G['A'])} B {len(G['B'])} C {len(G['C'])} / {n} · 미채점 {miss}")
    if miss: bad = True; continue
    M = lambda r: sum(G[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2
    F = {p: float(np.mean([M(r) for r in rows if r["page"] == p])) for p in (1, 3, 5)}
    role = {x: {p: float(np.mean([G[x][(r["pid"], r["item"])] >= 2 for r in rows if r["page"] == p])) for p in (1, 3, 5)} for x in "ABC"}
    drop = F[1] - F[5]
    print(f"  다수결 F1 {F[1]:.3f} · F3 {F[3]:.3f} · F5 {F[5]:.3f} · 하락 {drop:+.3f}")
    for x in "ABC": print(f"  역할 {x}: " + " · ".join(f"p{p} {v:.3f}" for p, v in role[x].items()))
    prof = {}
    for r in rows:
        if r["page"] in (1, 5): prof.setdefault(r["pid"], {1: [], 5: []})[r["page"]].append(M(r))
    d = sorted(((np.mean(v[1]) - np.mean(v[5]), pid) for pid, v in prof.items()), reverse=True)
    print(f"  프로필별 p1−p5: 최대 {d[0][0]:+.2f} ({d[0][1]}) · 중앙 {np.median([x for x, _ in d]):+.2f} · 하락 0.3 이상 {sum(x >= 0.3 for x, _ in d)}/{len(d)}")
    if drop <= 0.10 and F[5] >= 0.75: v = "(A) 검증 완료 — 변경 없음"
    elif drop > 0.15 or F[5] < 0.70: v = "(C) 붕괴 — 원인 진단 후 별건 사전등록"
    else: v = "(B) 경계 — 기록, 깊이 축 후보"
    print(f"  판정: {v}")
print("\n참조:", " · ".join(f"{k} {a:.3f}→{b:.3f} ({a-b:+.3f})" for k, (a, b) in REF.items()))
if bad: sys.exit("미채점이 남아 판정하지 않는다")
