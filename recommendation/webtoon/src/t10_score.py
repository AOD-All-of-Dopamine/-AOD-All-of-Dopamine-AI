"""T-10 집계 — 사전등록 eval/t10_preregister.md. Δ5 = 5페이지 다수결(n1−n0) · Δ1 = 1페이지 표본 가드."""
import glob, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
K = json.load(open(ROOT / "eval/t10_key.json")); rows, todo = K["rows"], K["todo"]; key = {t["id"]: t for t in todo}
G = {x: {} for x in "ABC"}
for x in "ABC":
    for f in sorted(glob.glob(str(ROOT / f"eval/t10/{x}_*.json"))):
        for i, g in json.load(open(f)).items():
            if i in key: G[x][(key[i]["pid"], key[i]["item"])] = int(g)
for x in "ABC": print(f"역할 {x} 채점 {len(G[x]):,}/{len(todo):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
print(f"미채점 (슬롯×역할) {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")
M = lambda r: sum(G[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2
maj = lambda sel: float(np.mean([M(r) for r in sel])) if sel else float("nan")
fitx = lambda x, sel: float(np.mean([G[x][(r["pid"], r["item"])] >= 2 for r in sel])) if sel else float("nan")
S = lambda arm, page: [r for r in rows if r["arm"] == arm and r["page"] == page]
F = {(a, p): maj(S(a, p)) for a in ("n0", "n1") for p in (1, 5)}
d5 = F[("n1", 5)] - F[("n0", 5)]; d1 = F[("n1", 1)] - F[("n0", 1)]
print("\n### 주 지표 — 3역할 다수결 · 프로필 67")
for a in ("n0", "n1"):
    print(f"  {a}: 5페이지 {F[(a, 5)]:.4f} (n={len(S(a, 5))}) · 1페이지 표본 {F[(a, 1)]:.4f} (n={len(S(a, 1))})")
print(f"\nΔ5 {d5:+.4f} · Δ1 {d1:+.4f}")
print("\n### 부수 관측 (판정 아님)")
for a in ("n0", "n1"):
    print(f"  {a} 5페이지 역할 A {fitx('A', S(a, 5)):.3f} · B {fitx('B', S(a, 5)):.3f} · C {fitx('C', S(a, 5)):.3f}")
P = {p["pid"]: p for p in json.load(open(ROOT / "eval/profiles.json"))}
for ax in ("rule", "coh", "mix", "count"):
    sel = lambda a: [r for r in S(a, 5) if P[r["pid"]].get("axis") == ax]
    if sel("n0"): print(f"  축 {ax}: 5페이지 {maj(sel('n0')):.3f} → {maj(sel('n1')):.3f} ({maj(sel('n1')) - maj(sel('n0')):+.3f})")
per = {pid: maj([r for r in S("n1", 5) if r["pid"] == pid]) - maj([r for r in S("n0", 5) if r["pid"] == pid]) for pid in P}
v = np.array(list(per.values()))
print(f"  프로필 Δ5 부호 +{(v > 0).sum()} / 0 {(v == 0).sum()} / −{(v < 0).sum()} · 중앙 {np.median(v):+.2f}")
if d5 >= 0.05 and d1 >= -0.03: print("\n판정: (A) 채택 — PRODUCTION['tag_drop_genre'] = True")
elif d5 >= 0.02 and d1 >= -0.03: print("\n판정: (B) 보류 — False 유지, 재현 라운드 필요")
else: print("\n판정: (C) 기각 — False 유지")
