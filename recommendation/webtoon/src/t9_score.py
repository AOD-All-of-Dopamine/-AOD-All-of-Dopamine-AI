"""T-9 집계 — 사전등록 eval/t9_preregister.md. 주 지표 3역할 다수결 합산 · 프로필 67 · 팔 2개."""
import json, glob, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
K = json.load(open(ROOT / "eval/t9_key.json")); rows = K["rows"]; key = {t["id"]: t for t in K["todo"]}
G = {x: {} for x in "ABC"}
for x in "ABC":
    for f in sorted(glob.glob(str(ROOT / f"eval/t9/{x}_*.json"))):
        for i, g in json.load(open(f)).items():
            if i in key: G[x][(key[i]["pid"], key[i]["item"])] = int(g)
for x in "ABC": print(f"역할 {x} 채점 {len(G[x]):,}/{len(K['todo']):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
print(f"미채점 (슬롯×역할) {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")
prof = {p["pid"]: p for p in json.load(open(ROOT / "eval/profiles.json"))}
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
eng = Engine(); L = json.load(open(ROOT / "eval/t9_lists.json"))
M = lambda r: sum(G[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2
maj = lambda sel: float(np.mean([M(r) for r in sel])) if sel else float("nan")
fitx = lambda x, sel: float(np.mean([G[x][(r["pid"], r["item"])] >= 2 for r in sel])) if sel else float("nan")
def ils(arm):
    v = []
    for p in prof:
        E = eng.emb[[eng.row[i] for i in L[f"{arm}|{p}"][:50]]]; S = E @ E.T; iu = np.triu_indices(len(E), 1); v.append(S[iu].mean())
    return float(np.mean(v))
def cov(arm):
    v = []
    for p in prof:
        st = [eng._tags[eng.row[int(s)]] for s in prof[p]["seeds"]]
        for i in L[f"{arm}|{p}"][:50]:
            v.append(max((len(eng._tags[eng.row[i]] & s) / len(s) if s else 0.0) for s in st))
    return float(np.mean(v))
V = {}
print("\n### 주 지표 — 3역할 다수결 · 프로필 67 · k=50")
for a in ("g0", "g2"):
    s = [r for r in rows if r["arm"] == a]
    mix = [r for r in s if prof[r["pid"]]["axis"] == "mix"]
    V[a] = dict(m=maj(s), mix=maj(mix), ils=ils(a))
    print(f"  {a} 합산 {V[a]['m']:.4f} · 1-10 {maj([r for r in s if r['part']=='top10']):.4f} · 11-50 {maj([r for r in s if r['part']=='tail']):.4f} · "
          f"mix {V[a]['mix']:.4f} · ILS {V[a]['ils']:.3f} · 태그피복 {cov(a):.3f}")
d = V["g2"]["m"] - V["g0"]["m"]
e_ok = V["g2"]["ils"] < V["g0"]["ils"] + 0.03; f_ok = V["g2"]["mix"] >= V["g0"]["mix"] - 0.05
print(f"\nΔ {d:+.4f} · (E) ILS {'OK' if e_ok else 'NG'} · (F) mix Δ {V['g2']['mix']-V['g0']['mix']:+.4f} {'OK' if f_ok else 'NG'}")
print("\n### 부수 관측 (판정 아님)")
for a in ("g0", "g2"):
    s = [r for r in rows if r["arm"] == a]
    print(f"  {a} 역할 B {fitx('B',s):.4f} · A {fitx('A',s):.4f} · C {fitx('C',s):.4f}")
print("  축별 다수결:", " · ".join(f"{ax} {maj([r for r in rows if r['arm']=='g0' and prof[r['pid']]['axis']==ax]):.3f}→{maj([r for r in rows if r['arm']=='g2' and prof[r['pid']]['axis']==ax]):.3f}" for ax in ("rule","coh","mix","count")))
T3 = {(r["pid"], r["item"]): r["maj"] for r in json.load(open(ROOT / "eval/t3_graded.json"))["rows"] if "maj" in r}
common = [k for k in T3 if (k[0], k[1]) in G["B"]]
if common:
    print(f"  T-3 과 같은 쌍 {len(common)}건 · 다수결 적합 일치 {np.mean([T3[k] == (sum(G[x][k] >= 2 for x in 'ABC') >= 2) for k in common]):.3f}")
if not (e_ok and f_ok): print("\n판정: (C) 재현 실패(안전장치 위반) — tag_w = 0 으로 되돌린다")
elif d >= 0.03: print("\n판정: (A) 재현 성공 — tag_w = 0.2 유지·확정")
elif d >= 0.02: print("\n판정: (B') 약한 재현 — tag_w = 0.2 유지, 크기 불확실로 기록")
else: print("\n판정: (C) 재현 실패 — tag_w = 0 으로 되돌린다")
