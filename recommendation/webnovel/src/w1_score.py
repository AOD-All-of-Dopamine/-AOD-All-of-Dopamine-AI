"""W-1 집계 — 사전등록 cdaeb8e8… 문턱 + 안전장치 (E) ILS · (F) 저관심."""
import json, glob, collections, sys, os
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
K = json.load(open(ROOT / "eval/w1_key.json")); rows, todo = K["rows"], K["todo"]
key = {t["id"]: t for t in todo}; votes = collections.defaultdict(list)
for f in sorted(glob.glob(str(ROOT / "eval/w1/[ABC]_*.json"))):
    for i, g in json.load(open(f)).items():
        if i in key: votes[i].append(int(g) >= 2)
maj, unan = {}, 0
for i, v in votes.items():
    if len(v) >= 3: maj[i] = sum(v) >= 2; unan += (len(set(v)) == 1)
print(f"신규 판정 {len(maj)}/{len(todo)} · 만장일치 {unan/max(len(maj),1):.2f}")
pair = {(key[i]["pid"], key[i]["item"]): m for i, m in maj.items()}
miss = 0
for r in rows:
    k = (r["pid"], r["item"])
    if k in pair: r["maj"] = pair[k]
    else: miss += 1
print(f"미채점 슬롯 {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")
json.dump(dict(rows=rows), open(ROOT / "eval/w1_graded.json", "w"), ensure_ascii=False)
P_ = lambda rs: float(np.mean([r["maj"] for r in rs])) if rs else float("nan")
os.environ["AOD_ARTIFACTS"] = str(ROOT / "artifacts/wn_v6")
from src.wn_eval import Engine
eng = Engine(); L = json.load(open(ROOT / "eval/w1_lists.json"))
prof = pd.read_parquet(ROOT / "artifacts/wn_v6/profiles_v6.parquet")
ds = pd.read_parquet(ROOT / "artifacts/wn_v6/dataset.parquet").set_index("item_id")
FAV = ds["interest_count"].fillna(0).to_dict()
seed_fav = {r.profile_id: np.median([FAV.get(int(s), 0) for s in r["seed_ids"]]) for _, r in prof.iterrows()}
q25 = np.quantile(list(seed_fav.values()), 0.25)
low = {p for p, v in seed_fav.items() if v <= q25}
print(f"저관심 프로필 {len(low)}개 (시드 관심수 중앙 ≤ {q25:,.0f})")
def ils(arm):
    v = []
    for _, r in prof.iterrows():
        ids = L[f"{arm}|{r.profile_id}"][:50]; M = eng.emb[[eng.id_to_row[i] for i in ids]]
        S = M @ M.T; iu = np.triu_indices(len(M), 1); v.append(S[iu].mean())
    return float(np.mean(v))
V = {}
print("\n### 팔별 (wn_v6 · 교정 프로필 52 · 3역할 다수결)")
for a in ("p0", "p03", "p10"):
    s = [r for r in rows if r["arm"] == a]; h = [r for r in s if r["part"] == "top10"]; t = [r for r in s if r["part"] == "tail"]
    lo = [r for r in s if r["pid"] in low]
    fav = np.median([FAV.get(i, 0) for _, r in prof.iterrows() for i in L[f"{a}|{r.profile_id}"][:50]])
    V[a] = (P_(s), P_(lo), ils(a))
    print(f"  {a:<4} 1-10 {P_(h):.3f} · 11-50 {P_(t):.3f} · 합산 {P_(s):.4f} · 저관심 {P_(lo):.3f} · ILS {ils(a):.3f} · 관심수 중앙 {fav:,.0f}")
best = max(("p03", "p10"), key=lambda a: V[a][0]); d = V[best][0] - V["p0"][0]
print(f"\n최선 {best}: Δ합산 {d:+.4f} · Δ저관심 {V[best][1]-V['p0'][1]:+.4f} · ΔILS {V[best][2]-V['p0'][2]:+.4f}")
if d >= 0.03:
    if V[best][2] >= V["p0"][2] + 0.03: print("판정: (E) ILS 안전장치 → 채택하지 않음, pop_boost=0.0 으로 되돌린다")
    elif V[best][1] < V["p0"][1] - 0.05: print("판정: (F) 저관심 안전장치 → 채택하지 않음, pop_boost=0.0 으로 되돌린다")
    else: print(f"판정: (A) 채택 → pop_boost = {'0.03' if best=='p03' else '0.10'}")
elif d < 0.02: print("판정: (C) → pop_boost 를 0.0 으로 **철회**한다 (D-66 은 파일럿 코퍼스의 결론이었다)")
else: print("판정: 보류 → 0.03 유지하되 문서에 미확인 표시")
