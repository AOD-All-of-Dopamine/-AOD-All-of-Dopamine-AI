"""X-27 집계 — 사전등록 eval/x27_preregister.md. TMDB 52프로필 1·5페이지 3역할 다수결."""
import glob, json, sys
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent
K = json.load(open(OUT / "x27_tmdb_key.json")); rows = K["rows"]; key = {t["id"]: t for t in K["todo"]}
G = {x: {} for x in "ABC"}
for x in "ABC":
    for f in sorted(glob.glob(str(OUT / f"x27/tmdb/{x}_*.json"))):
        for i, g in json.load(open(f)).items():
            if i in key: G[x][(key[i]["pid"], key[i]["item"])] = int(g)
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
print(f"슬롯 {len(rows)} · 채점 A {len(G['A'])} B {len(G['B'])} C {len(G['C'])} / {len(K['todo'])} · 미채점 {miss}")
if miss: sys.exit("미채점이 남아 판정하지 않는다")
M = lambda r: sum(G[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2
F = {p: float(np.mean([M(r) for r in rows if r["page"] == p])) for p in (1, 5)}
drop = F[1] - F[5]
print(f"\n다수결 F1 {F[1]:.3f} · F5 {F[5]:.3f} · 하락 {drop:+.3f}")
for x in "ABC":
    print(f"  역할 {x}: " + " · ".join(f"p{p} {np.mean([G[x][(r['pid'], r['item'])] >= 2 for r in rows if r['page'] == p]):.3f}" for p in (1, 5)))
prof = {}
for r in rows: prof.setdefault(r["pid"], {1: [], 5: []})[r["page"]].append(M(r))
d = np.array([np.mean(v[1]) - np.mean(v[5]) for v in prof.values()])
print(f"  프로필별 하락: 중앙 {np.median(d):+.2f} · 0.3 이상 {(d >= 0.3).sum()}/{len(d)} · 음수(5페이지가 더 좋음) {(d < 0).sum()}")
sub = json.load(open(OUT / "x26_tmdb_pages.json")).keys()
S = [r for r in rows if r["pid"] in sub]
print(f"  X-26 13프로필 부분집합(새 채점): F1 {np.mean([M(r) for r in S if r['page']==1]):.3f} · F5 {np.mean([M(r) for r in S if r['page']==5]):.3f} (X-26: 0.900 · 0.754)")
if drop <= 0.10 and F[5] >= 0.75: v = "(A) 검증 완료 — TMDB 깊이 축 열지 않음, X-26 (B) 는 소표본 과장"
elif drop > 0.15 or F[5] < 0.70: v = "(C) 붕괴 — 원인 진단 후 별건 사전등록"
else: v = "(B) 경계 확인 — 깊이 축 후보 유지, 원인 진단은 전 프로필 채점으로"
print(f"\n판정: {v}")
