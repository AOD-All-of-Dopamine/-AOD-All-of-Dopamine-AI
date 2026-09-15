"""T-11 집계 — 사전등록 eval/t11_preregister.md. (나) 반영 · 무관 가드 · (가) 품질 Δ → 가장 작은 통과 w."""
import glob, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PG = json.load(open(ROOT / "eval/t11_pages.json"))
K = json.load(open(ROOT / "eval/t11_key.json")); rows, todo = K["rows"], K["todo"]; key = {t["id"]: t for t in todo}
G = {x: {} for x in "ABC"}
for x in "ABC":
    for f in sorted(glob.glob(str(ROOT / f"eval/t11/{x}_*.json"))):
        for i, g in json.load(open(f)).items():
            if i in key: G[x][(key[i]["pid"], key[i]["item"])] = int(g)
for x in "ABC": print(f"역할 {x} 채점 {len(G[x]):,}/{len(todo):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
print(f"미채점 (행×역할) {miss}")
if miss: sys.exit("미채점이 남아 판정하지 않는다")
M = lambda pid, i: sum(G[x][(pid, i)] >= 2 for x in "ABC") >= 2
n = len(PG)

N = {a: float(np.mean([len(set(r["t"][a]) & set(r["nn50"])) for r in PG.values()])) for a in ("d0", "w1", "w2", "w3")}
O = {a: float(np.mean([len(set(r["u"][a]) & set(r["u"]["d0"])) / 20 for r in PG.values()])) for a in ("w1", "w2", "w3")}
def delta(arm, fit=M):
    s = sum(fit(r["pid"], r["item"]) * (1 if r["kind"] == "in" else -1) for r in rows if r["arm"] == arm)
    return s / (n * 20)
D = {a: delta(a) for a in ("w1", "w2", "w3")}

print(f"\n### 판정 지표 — 프로필 {n}")
print(f"  d0: N {N['d0']:.2f}")
ok = {}
for a in ("w1", "w2", "w3"):
    c1, c2, c3 = N[a] <= 0.5 * N["d0"], O[a] >= 0.95, D[a] >= -0.05
    ok[a] = c1 and c2 and c3
    print(f"  {a}: (나) N {N[a]:.2f} ({N[a] / N['d0']:.2f}배) {'통과' if c1 else '미달'} · 무관 O {O[a]:.3f} {'통과' if c2 else '미달'} · (가) Δ {D[a]:+.4f} {'통과' if c3 else '미달'}")

print("\n### 부수 관측 (판정 아님)")
for a in ("w1", "w2", "w3"):
    role = " · ".join(f"{x} {delta(a, lambda pid, i, x=x: G[x][(pid, i)] >= 2):+.3f}" for x in "ABC")
    nin = sum(1 for r in rows if r["arm"] == a and r["kind"] == "in")
    fin = np.mean([M(r["pid"], r["item"]) for r in rows if r["arm"] == a and r["kind"] == "in"]) if nin else float("nan")
    fout = np.mean([M(r["pid"], r["item"]) for r in rows if r["arm"] == a and r["kind"] == "out"]) if nin else float("nan")
    print(f"  {a}: 역할별 Δ {role} · 들어온 {nin} (적합 {fin:.3f}) · 빠진 적합 {fout:.3f}")
P = {p["pid"]: p for p in json.load(open(ROOT / "eval/profiles.json"))}
for ax in ("rule", "coh", "mix", "count"):
    sel = [r for pid, r in PG.items() if P[pid].get("axis") == ax]
    if sel:
        print(f"  축 {ax} ({len(sel)}): N d0 {np.mean([len(set(r['t']['d0']) & set(r['nn50'])) for r in sel]):.2f} → w2 {np.mean([len(set(r['t']['w2']) & set(r['nn50'])) for r in sel]):.2f}")
for a in ("w1", "w2", "w3"):
    o = np.array([len(set(r["u"][a]) & set(r["u"]["d0"])) / 20 for r in PG.values()])
    print(f"  {a} 무관 가드 프로필별: 1.00 {(o == 1).sum()} · 0.90 미만 {(o < 0.9).sum()} · 최소 {o.min():.2f}")

win = next((a for a in ("w1", "w2", "w3") if ok[a]), None)
print(f"\n판정: (A) 채택 — PRODUCTION['dislike_w'] = {float(win[1:])}" if win else "\n판정: (C) 기각 — dislike_w 0.0 유지 (싫어요 작품·시리즈 제외만)")
