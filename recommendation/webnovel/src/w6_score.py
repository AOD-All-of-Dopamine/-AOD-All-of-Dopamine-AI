"""W-6 집계 — 사전등록 eval/w6_preregister.md. 가드 = 3역할 다수결 적합(E) − 적합(Ctl)."""
import json, glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
K = json.load(open(ROOT / "eval/w6_key.json")); rows, todo = K["rows"], K["todo"]
key = {t["id"]: t for t in todo}
G = {x: {} for x in "ABC"}
for x in "ABC":
    for f in sorted(glob.glob(str(ROOT / f"eval/w6/{x}_*.json"))):
        for i, g in json.load(open(f)).items():
            if i in key: G[x][(key[i]["pid"], key[i]["item"])] = int(g)
for x in "ABC": print(f"역할 {x} 채점 {len(G[x]):,}/{len(todo):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
print(f"미채점 (항목×역할) {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")
M = lambda r: sum(G[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2
maj = lambda sel: float(np.mean([M(r) for r in sel])) if sel else float("nan")
fitx = lambda x, sel: float(np.mean([G[x][(r["pid"], r["item"])] >= 2 for r in sel])) if sel else float("nan")
S = {s: [r for r in rows if r["set"] == s] for s in ("E", "Ctl", "Lv")}
print("\n### 판정 2 — 가드 (3역할 다수결)")
for s, sel in S.items():
    print(f"  {s:3} n={len(sel):3} 다수결 {maj(sel):.4f} · A {fitx('A',sel):.3f} · B {fitx('B',sel):.3f} · C {fitx('C',sel):.3f}")
d = maj(S["E"]) - maj(S["Ctl"])
print(f"\nΔ_guard {d:+.4f}")
print("\n### 부수 관측 (판정 아님)")
ds = pd.read_parquet(ROOT / "artifacts/wn_v6/dataset.parquet").set_index("item_id")
norm = lambda s: re.sub(r"[\s\W_]+", "", str(s).lower())
TA = lambda i: (norm(ds.loc[i, "name"]), str(ds.loc[i, "author"] or "").strip())
P = pd.read_parquet(ROOT / "artifacts/wn_v6/profiles_v6.parquet"); seeds = {r.profile_id: [int(s) for s in r.seed_ids] for r in P.itertuples(index=False)}
PG = json.load(open(ROOT / "eval/w6_pages.json"))
echo, rep, push = [], [], []
for r in S["Lv"]:
    flat = [i for x in PG[f"n0|{r['pid']}"] for i in x]
    if TA(r["item"]) in {TA(s) for s in seeds[r["pid"]]}: echo.append(r)
    elif TA(r["item"]) in {TA(j) for j in flat[:flat.index(r["item"])]}: rep.append(r)
    else: push.append(r)
print(f"  Lv 분해: 시드 에코 {len(echo)} 다수결 {maj(echo):.3f} · 앞 페이지 판본 재등장 {len(rep)} {maj(rep):.3f} · 밀려남 {len(push)} {maj(push):.3f}")
print(f"  Lv 에코 위치: {sorted(r['pos'] for r in echo)}")
print("\n기계 판정: 통과 (n1 에코 0 · 페이지 안 0 · 페이지 간 0 — w6_build)")
if d >= -0.10: print("판정: (A) 채택 — PRODUCTION['drop_excluded_series'] = True")
elif d >= -0.20: print("판정: (B) 보류 — 현행 유지, 좁은 규칙 다음 라운드")
else: print("판정: (C) 기각 — 현행 유지")
