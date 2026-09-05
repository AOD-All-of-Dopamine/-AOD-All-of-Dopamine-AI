"""T-3 집계 — 사전등록 28f3cea3… 문턱 + 안전장치 (E) ILS · (F) mix."""
import json, glob, collections, sys, os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
K = json.load(open(ROOT / "eval/t6_key.json")); rows, todo = K["rows"], K["todo"]
key = {t["id"]: t for t in todo}; votes = collections.defaultdict(list)
for f in sorted(glob.glob(str(ROOT / "eval/t6/[ABC]_*.json"))):
    for i, g in json.load(open(f)).items():
        if i in key: votes[i].append(int(g) >= 2)
maj, unan = {}, 0
for i, v in votes.items():
    if len(v) >= 3: maj[i] = sum(v) >= 2; unan += (len(set(v)) == 1)
print(f"신규 판정 {len(maj)}/{len(todo)} · 만장일치 {unan/max(len(maj),1):.2f}")
pair = {(key[i]["pid"], key[i]["item"]): m for i, m in maj.items()}; miss = 0
for r in rows:
    if "maj" in r: continue
    k = (r["pid"], r["item"])
    if k in pair: r["maj"] = pair[k]
    else: miss += 1
print(f"미채점 슬롯 {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")
json.dump(dict(rows=rows), open(ROOT / "eval/t6_graded.json", "w"), ensure_ascii=False)
P = lambda rs: float(np.mean([r["maj"] for r in rs])) if rs else float("nan")
prof = {p["pid"]: p for p in json.load(open(ROOT / "eval/profiles.json"))}
L = json.load(open(ROOT / "eval/t6_lists.json"))
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
eng = Engine()
def ils(arm):
    v = []
    for p in prof:
        ids = L[f"{arm}|{p}"][:50]; M = eng.emb[[eng.row[i] for i in ids]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); v.append(S[iu].mean())
    return float(np.mean(v))
V = {}
print("\n### 팔별 (rep_v2 · pop 0 · tag_w 0.2 · top2_mean · hub 0 · 프로필 67)")
for a in ("b0", "b10", "b20"):
    s = [r for r in rows if r["arm"] == a]; h = [r for r in s if r["part"] == "top10"]; t = [r for r in s if r["part"] == "tail"]
    mix = [r for r in s if prof[r["pid"]]["axis"] == "mix"]
    V[a] = (P(s), P(mix), ils(a)); print(f"  {a:<3} 1-10 {P(h):.3f} · 11-50 {P(t):.3f} · 합산 {P(s):.4f} · mix {P(mix):.3f} · ILS {ils(a):.3f}")
best = max(("b10", "b20"), key=lambda a: V[a][0]); d = V[best][0] - V["b0"][0]
print(f"\n최선 {best}: Δ합산 {d:+.4f} · Δmix {V[best][1]-V['b0'][1]:+.4f} · ΔILS {V[best][2]-V['b0'][2]:+.4f}")
if d >= 0.03:
    if V[best][2] >= V["b0"][2] + 0.03: print("판정: (E) ILS 안전장치 → 채택하지 않음")
    elif V[best][1] < V["b0"][1] - 0.05: print("판정: (F) mix 안전장치 → 채택하지 않음")
    else: print(f"판정: (A) 채택 → star_boost = {'0.10' if best=='b10' else '0.20'}")
elif d < 0.02: print("판정: (C) 0 유지")
else: print("판정: 보류 → 0 유지")
print("\n### 축별")
for ax in ("rule", "coh", "mix", "count"):
    print(f"  {ax:<6}" + "".join(f"  {a} {P([r for r in rows if r['arm']==a and prof[r['pid']]['axis']==ax]):.3f}" for a in ("b0","b10","b20")))
