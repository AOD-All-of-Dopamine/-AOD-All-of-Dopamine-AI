"""T-8 집계 — 사전등록 eval/t8_preregister.md 판정 규칙. 주 지표 역할 B 단독 · 대상군 64 · 팔 2개."""
import json, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

def load(tag):
    K = json.load(open(ROOT / f"eval/{tag}_key.json")); key = {t["id"]: t for t in K["todo"]}
    G = {x: {} for x in "ABC"}
    for x in "ABC":
        for f in sorted(glob.glob(str(ROOT / f"eval/{tag}/{x}_*.json"))):
            for i, g in json.load(open(f)).items():
                if i in key: G[x][(key[i]["pid"], key[i]["item"])] = int(g)
    return K, G

K, G = load("t8"); rows, target = K["rows"], set(K["target"])
for x in "ABC": print(f"역할 {x} 채점 {len(G[x]):,}/{len(K['todo']):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
print(f"미채점 (슬롯×역할) {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")

prof = {p["pid"]: p for p in json.load(open(ROOT / "eval/profiles.json"))}
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
eng = Engine(); L = json.load(open(ROOT / "eval/t8_lists.json"))
FAV = dict(zip([int(x) for x in eng.ds["item_id"]], pd.to_numeric(eng.ds["favorite_count"], errors="coerce").fillna(0)))
sf = {p: np.median([FAV[int(s)] for s in prof[p]["seeds"]]) for p in prof}
q25 = np.quantile(list(sf.values()), 0.25); low = {p for p, v in sf.items() if v <= q25}
fit = lambda Gx, sel: float(np.mean([Gx[(r["pid"], r["item"])] >= 2 for r in sel])) if sel else float("nan")
maj = lambda G_, sel: float(np.mean([sum(G_[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2 for r in sel])) if sel else float("nan")
def ils(arm):
    v = []
    for p in prof:
        M = eng.emb[[eng.row[i] for i in L[f"{arm}|{p}"][:50]]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); v.append(S[iu].mean())
    return float(np.mean(v))

B, A, MIX, IL = {}, {}, {}, {}
print(f"\n### 주 지표 — 역할 B · 대상군 {len(target)} · k=50  (저관심 {len(low)}, ≤ {q25:,.0f})")
for a in ("r0", "r10"):
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]
    B[a] = fit(G["B"], s); IL[a] = ils(a)
    MIX[a] = fit(G["B"], [r for r in rows if r["arm"] == a and prof[r["pid"]]["axis"] == "mix"])
    A[a] = fit(G["A"], [r for r in rows if r["arm"] == a and r["pid"] in low])
    print(f"  {a:4} B {B[a]:.4f} · 1-10 {fit(G['B'],[r for r in s if r['part']=='top10']):.4f} · 11-50 {fit(G['B'],[r for r in s if r['part']=='tail']):.4f} · "
          f"mix(B) {MIX[a]:.4f} · 저관심 A {A[a]:.4f} · ILS {IL[a]:.3f}")
d = B["r10"] - B["r0"]; dA = A["r10"] - A["r0"]
e_ok = IL["r10"] < IL["r0"] + 0.03; f_ok = MIX["r10"] >= MIX["r0"] - 0.05; safe = dA >= -0.03
print(f"\nΔ_B {d:+.4f} · 저관심 Δ_A {dA:+.4f} ({'OK' if safe else 'NG'}) · (E) {'OK' if e_ok else 'NG'} · (F) {'OK' if f_ok else 'NG'}")

print("\n### 부수 관측 (판정 아님)")
for a in ("r0", "r10"):
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]; s67 = [r for r in rows if r["arm"] == a]
    print(f"  {a:4} 3역할다수 {maj(G, s):.4f} · A {fit(G['A'],s):.4f} · C {fit(G['C'],s):.4f} · B 67전체 {fit(G['B'],s67):.4f}")
print("  축별 역할 B:", " · ".join(f"{ax} {fit(G['B'],[r for r in rows if r['arm']=='r0' and prof[r['pid']]['axis']==ax]):.3f}→{fit(G['B'],[r for r in rows if r['arm']=='r10' and prof[r['pid']]['axis']==ax]):.3f}" for ax in ("rule","coh","mix","count")))
K7, G7 = load("t7")
pool = {}
for arm7, arm8 in (("c0", "r0"), ("c10", "r10")):
    s = [(G7["B"], r) for r in K7["rows"] if r["arm"] == arm7 and r["pid"] in target] + [(G["B"], r) for r in rows if r["arm"] == arm8 and r["pid"] in target]
    pool[arm8] = float(np.mean([Gx[(r["pid"], r["item"])] >= 2 for Gx, r in s]))
print(f"  T-7+T-8 합산 역할 B (프로필당 20슬롯): r0 {pool['r0']:.4f} · r10 {pool['r10']:.4f} · Δ {pool['r10']-pool['r0']:+.4f}  (T-7 단독 +0.0281)")
common = [k for k in G["B"] if k in G7["B"]]
if common:
    agree = np.mean([G["B"][k] == G7["B"][k] for k in common]); agree2 = np.mean([(G["B"][k] >= 2) == (G7["B"][k] >= 2) for k in common])
    print(f"  T-7·T-8 에서 같은 쌍 재채점 {len(common)}건 · 역할 B 등급 일치 {agree:.3f} · 적합 여부 일치 {agree2:.3f}")

if d >= 0.04 and safe and e_ok and f_ok: print("\n판정: (A) 채택 — creator_w = 0.10")
elif not (safe and e_ok and f_ok): print("\n판정: (C) 재현 실패(안전장치 위반) — creator_w = 0 유지, 축 종료")
elif d >= 0.02: print("\n판정: (B') 두 번째 보류 → 축 종료 — creator_w = 0 유지 (효과가 있더라도 채택할 만큼 크지 않다)")
else: print("\n판정: (C) 재현 실패 — creator_w = 0 유지, T-7 +0.028 은 재현되지 않음, 축 종료")
