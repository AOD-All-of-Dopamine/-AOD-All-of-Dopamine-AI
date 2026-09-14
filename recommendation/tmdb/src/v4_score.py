"""V-4 집계 — 사전등록 eval/v4_preregister.md. 주 지표 3역할 다수결 합산 · 프로필 52 · 팔 2개."""
import json, glob, sys
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
K, G = load("v4"); rows = K["rows"]
for x in "ABC": print(f"역할 {x} 채점 {len(G[x]):,}/{len(K['todo']):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
print(f"미채점 (슬롯×역할) {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")
P = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")
ds0 = pd.read_parquet(ROOT / "artifacts/tmdb_v1/dataset.parquet")
idx = pd.read_parquet(ROOT / "artifacts/tmdb_v1/corpus_index.parquet").sort_values("embedding_row")
ds = ds0.set_index("item_id").loc[idx["item_id"].to_numpy()].reset_index()
sv = {r.profile_id: float(np.median([ds.iloc[int(x)]["vote_count"] for x in r.seed_rows])) for r in P.itertuples(index=False)}
q25 = np.quantile(list(sv.values()), 0.25); longtail = {p for p, v in sv.items() if v <= q25}
Mj = lambda G_, r: sum(G_[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2
maj = lambda sel: float(np.mean([Mj(G, r) for r in sel])) if sel else float("nan")
fitx = lambda x, sel: float(np.mean([G[x][(r["pid"], r["item"])] >= 2 for r in sel])) if sel else float("nan")
V, LT = {}, {}
print(f"\n### 주 지표 — 3역할 다수결 · 프로필 52 · k=50  (롱테일 {len(longtail)})")
for a in ("h0", "h4"):
    s = [r for r in rows if r["arm"] == a]
    V[a] = maj(s); LT[a] = maj([r for r in s if r["pid"] in longtail])
    print(f"  {a} 합산 {V[a]:.4f} · 1-10 {maj([r for r in s if r['part']=='top10']):.4f} · 11-50 {maj([r for r in s if r['part']=='tail']):.4f} · 롱테일 {LT[a]:.4f}")
d = V["h4"] - V["h0"]; dl = LT["h4"] - LT["h0"]; safe = dl >= -0.03
per = {}
for p in P["profile_id"]:
    per[p] = maj([r for r in rows if r["arm"] == "h4" and r["pid"] == p]) - maj([r for r in rows if r["arm"] == "h0" and r["pid"] == p])
worst = min(per, key=per.get); collapse = per[worst] <= -0.30
print(f"\nΔ {d:+.4f} · 롱테일 Δ {dl:+.4f} ({'OK' if safe else 'NG'}) · 최악 프로필 {worst} {per[worst]:+.2f} ({'붕괴' if collapse else 'OK'})")
print("\n### 부수 관측 (판정 아님)")
for a in ("h0", "h4"):
    s = [r for r in rows if r["arm"] == a]
    print(f"  {a} 역할 B {fitx('B',s):.4f} · A {fitx('A',s):.4f} · C {fitx('C',s):.4f}")
vals = np.array(list(per.values()))
print(f"  프로필 Δ 부호 +{(vals>0).sum()} / 0 {(vals==0).sum()} / −{(vals<0).sum()} · 중앙 {np.median(vals):+.2f} · five_horror {per.get('five_horror', float('nan')):+.2f}")
print("  하위 5:", sorted(per.items(), key=lambda kv: kv[1])[:5])
K3, G3 = load("v3")
common = [k for k in G["B"] if k in G3["B"]]
if common:
    print(f"  V-3 과 같은 쌍 {len(common)}건 · 다수결 일치 {np.mean([(sum(G[x][k]>=2 for x in 'ABC')>=2) == (sum(G3[x][k]>=2 for x in 'ABC')>=2) for k in common]):.3f} · "
          f"역할 B 적합 일치 {np.mean([(G['B'][k]>=2) == (G3['B'][k]>=2) for k in common]):.3f}")
if not safe: print("\n판정: (C) 재현 실패(롱테일 위반) — genre_w = 0 으로 되돌린다")
elif d >= 0.03 and not collapse: print("\n판정: (A) 재현 성공 — genre_w = 0.40 유지·확정")
elif d >= 0.02: print("\n판정: (B') 약한 재현 — genre_w = 0.40 유지, 크기 불확실로 기록" + (" (붕괴 안전장치로 A→B')" if d >= 0.03 else ""))
else: print("\n판정: (C) 재현 실패 — genre_w = 0 으로 되돌린다")
