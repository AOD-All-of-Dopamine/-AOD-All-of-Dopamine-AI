"""V-3 집계 — 사전등록 eval/v3_preregister.md 판정 규칙. 역할 B 단독 · 대상군 50 · 팔 2개."""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(tag):
    K = json.load(open(ROOT / f"eval/{tag}_key.json")); key = {t["id"]: t for t in K["todo"]}
    G = {x: {} for x in "ABC"}
    for x in "ABC":
        for f in sorted(glob.glob(str(ROOT / f"eval/{tag}/{x}_*.json"))):
            for i, g in json.load(open(f)).items():
                if i in key:
                    G[x][(key[i]["pid"], key[i]["item"])] = int(g)
    return K, G


K, G = load("v3"); rows, target = K["rows"], set(K["target"])
for x in "ABC":
    print(f"역할 {x} 채점 {len(G[x]):,}/{len(K['todo']):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in G[x])
print(f"미채점 (슬롯×역할) {miss}")
if miss:
    sys.exit("미채점이 남아 비교하지 않는다")

P = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")
ds0 = pd.read_parquet(ROOT / "artifacts/tmdb_v1/dataset.parquet")
idx = pd.read_parquet(ROOT / "artifacts/tmdb_v1/corpus_index.parquet").sort_values("embedding_row")
ds = ds0.set_index("item_id").loc[idx["item_id"].to_numpy()].reset_index()
sv = {r.profile_id: float(np.median([ds.iloc[int(x)]["vote_count"] for x in r.seed_rows])) for r in P.itertuples(index=False)}
q25 = np.quantile(list(sv.values()), 0.25); longtail = {p for p, v in sv.items() if v <= q25}

fit = lambda Gx, sel: float(np.mean([Gx[(r["pid"], r["item"])] >= 2 for r in sel])) if sel else float("nan")
maj = lambda G_, sel: float(np.mean([sum(G_[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2 for r in sel])) if sel else float("nan")

B, A = {}, {}
print(f"\n### 주 지표 — 역할 B · 대상군 {len(target)} · k=50  (롱테일 {len(longtail)}, 시드 투표 중앙 ≤ {q25:,.0f})")
for a in ("e0", "e10"):
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]
    B[a] = fit(G["B"], s)
    A[a] = fit(G["A"], [r for r in rows if r["arm"] == a and r["pid"] in longtail])
    print(f"  {a:4} B {B[a]:.4f} · 1-10 {fit(G['B'],[r for r in s if r['part']=='top10']):.4f} · "
          f"11-50 {fit(G['B'],[r for r in s if r['part']=='tail']):.4f} · 롱테일 A {A[a]:.4f}")
d = B["e10"] - B["e0"]; dA = A["e10"] - A["e0"]; safe = dA >= -0.03
print(f"\nΔ_B {d:+.4f} · 롱테일 Δ_A {dA:+.4f} ({'OK' if safe else 'NG'})")

print("\n### 부수 관측 (판정 아님)")
for a in ("e0", "e10"):
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]; s52 = [r for r in rows if r["arm"] == a]
    print(f"  {a:4} 3역할다수 {maj(G, s):.4f} · A {fit(G['A'],s):.4f} · C {fit(G['C'],s):.4f} · B 52전체 {fit(G['B'],s52):.4f}")
K2, G2 = load("v2")
pool = {}
for a2, a3 in (("d0", "e0"), ("d10", "e10")):
    s = [(G2["B"], r) for r in K2["rows"] if r["arm"] == a2 and r["pid"] in target] + \
        [(G["B"], r) for r in rows if r["arm"] == a3 and r["pid"] in target]
    pool[a3] = float(np.mean([Gx[(r["pid"], r["item"])] >= 2 for Gx, r in s]))
print(f"  V-2+V-3 합산 역할 B (프로필당 20슬롯): e0 {pool['e0']:.4f} · e10 {pool['e10']:.4f} · Δ {pool['e10']-pool['e0']:+.4f}  (V-2 단독 +0.0560)")
common = [k for k in G["B"] if k in G2["B"]]
if common:
    print(f"  V-2·V-3 같은 쌍 재채점 {len(common)}건 · 역할 B 등급 일치 {np.mean([G['B'][k]==G2['B'][k] for k in common]):.3f} · "
          f"적합 여부 일치 {np.mean([(G['B'][k]>=2)==(G2['B'][k]>=2) for k in common]):.3f}  (웹툰 T-8: 0.614 / 0.827)")

if not safe:
    print("\n판정: (C) 재현 실패(롱테일 안전 위반) — director_w = 0 으로 되돌린다")
elif d >= 0.04:
    print("\n판정: (A) 재현 성공 — director_w = 0.10 유지 · 확정")
elif d >= 0.02:
    print("\n판정: (B') 약한 재현 — director_w = 0.10 유지, 효과 크기 불확실로 기록")
else:
    print("\n판정: (C) 재현 실패 — director_w = 0 으로 되돌린다")
