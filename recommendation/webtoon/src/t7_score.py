"""T-7 집계 — 사전등록 8454695c… 판정 규칙. 주 지표 역할 B 단독 · 대상군 64."""
import json, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
K = json.load(open(ROOT / "eval/t7_key.json")); rows, todo, target = K["rows"], K["todo"], set(K["target"])
key = {t["id"]: t for t in todo}
grade = {x: {} for x in "ABC"}
for role in "ABC":
    for f in sorted(glob.glob(str(ROOT / f"eval/t7/{role}_*.json"))):
        for i, g in json.load(open(f)).items():
            if i in key: grade[role][(key[i]["pid"], key[i]["item"])] = int(g)
for role in "ABC": print(f"역할 {role} 채점 {len(grade[role]):,}/{len(todo):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in grade[x])
print(f"미채점 (슬롯×역할) {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")
prof = {p["pid"]: p for p in json.load(open(ROOT / "eval/profiles.json"))}
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
eng = Engine(); L = json.load(open(ROOT / "eval/t7_lists.json"))
FAV = dict(zip([int(x) for x in eng.ds["item_id"]], pd.to_numeric(eng.ds["favorite_count"], errors="coerce").fillna(0)))
sf = {p: np.median([FAV[int(s)] for s in prof[p]["seeds"]]) for p in prof}
q25 = np.quantile(list(sf.values()), 0.25); low = {p for p, v in sf.items() if v <= q25}
print(f"저관심 {len(low)} (≤ {q25:,.0f})")
ARMS = ["c0", "c10", "c25"]
fit = lambda role, sel: float(np.mean([grade[role][(r["pid"], r["item"])] >= 2 for r in sel])) if sel else float("nan")
maj = lambda sel: float(np.mean([sum(grade[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2 for r in sel])) if sel else float("nan")
def ils(arm):
    v = []
    for p in prof:
        ids = L[f"{arm}|{p}"][:50]; M = eng.emb[[eng.row[i] for i in ids]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); v.append(S[iu].mean())
    return float(np.mean(v))
B, A, MIX, IL = {}, {}, {}, {}
print(f"\n### 주 지표 — 역할 B · 대상군 {len(target)} · k=50")
for a in ARMS:
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]
    B[a] = fit("B", s); IL[a] = ils(a)
    MIX[a] = fit("B", [r for r in rows if r["arm"] == a and prof[r["pid"]]["axis"] == "mix"])
    print(f"  {a:4} B {B[a]:.4f} · 1-10 {fit('B',[r for r in s if r['part']=='top10']):.4f} · 11-50 {fit('B',[r for r in s if r['part']=='tail']):.4f} · Δ_B {B[a]-B['c0']:+.4f} · mix(B) {MIX[a]:.4f} · ILS {IL[a]:.3f}")
print(f"\n### 안전 — 역할 A · 저관심 {len(low)}")
for a in ARMS:
    A[a] = fit("A", [r for r in rows if r["arm"] == a and r["pid"] in low]); print(f"  {a:4} A 저관심 {A[a]:.4f} Δ {A[a]-A['c0']:+.4f}")
print("\n### 부수 관측 (판정 아님)")
for a in ARMS:
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]; s67 = [r for r in rows if r["arm"] == a]
    print(f"  {a:4} 3역할다수 {maj(s):.4f} · A {fit('A',s):.4f} · C {fit('C',s):.4f} · B 67전체 {fit('B',s67):.4f}")
print("\n### 축별 역할 B")
for ax in ("rule", "coh", "mix", "count"):
    print(f"  {ax:<6}" + "".join(f"  {a} {fit('B',[r for r in rows if r['arm']==a and prof[r['pid']]['axis']==ax]):.3f}" for a in ARMS))
best = max(("c10", "c25"), key=lambda a: B[a]); d = B[best] - B["c0"]
mono = B["c10"] > B["c0"] and B["c25"] > B["c0"]; safe = A[best] - A["c0"] >= -0.03
e_ok = IL[best] < IL["c0"] + 0.03; f_ok = MIX[best] >= MIX["c0"] - 0.05
print(f"\n최고 {best} · Δ_B {d:+.4f} · 단조성 {'OK' if mono else 'NG'} · 저관심 안전 {'OK' if safe else 'NG'} · (E) {'OK' if e_ok else 'NG'} · (F) {'OK' if f_ok else 'NG'}")
if d >= 0.04 and mono and safe and e_ok and f_ok: print(f"판정: (A) 채택 — creator_w = {'0.10' if best=='c10' else '0.25'}")
elif d >= 0.04 and not (safe and e_ok and f_ok): print("판정: (C) 기각 — 안전장치 위반. Δ_B 와 무관하게 채택하지 않는다")
elif d >= 0.02 or (d >= 0.04 and not mono): print("판정: (B) 보류 — creator_w = 0 유지, 재현 필요. 문턱을 낮추지 않는다")
else: print("판정: (C) 기각 — creator_w = 0 유지")
