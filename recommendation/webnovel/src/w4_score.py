"""W-4 집계 — 사전등록 b64f2540… 판정 규칙. 주 지표 역할 B 단독 · 대상군 51."""
import json, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
K = json.load(open(ROOT / "eval/w4_key.json")); rows, todo, target = K["rows"], K["todo"], set(K["target"])
key = {t["id"]: t for t in todo}
grade = {x: {} for x in "ABC"}
for role in "ABC":
    for f in sorted(glob.glob(str(ROOT / f"eval/w4/{role}_*.json"))):
        for i, g in json.load(open(f)).items():
            if i in key: grade[role][(key[i]["pid"], key[i]["item"])] = int(g)
for role in "ABC": print(f"역할 {role} 채점 {len(grade[role]):,}/{len(todo):,}")
miss = sum(1 for r in rows for x in "ABC" if (r["pid"], r["item"]) not in grade[x])
print(f"미채점 (슬롯×역할) {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다")
os.environ["AOD_ARTIFACTS"] = str(ROOT / "artifacts/wn_v6")
from src.wn_eval import Engine
eng = Engine(); L = json.load(open(ROOT / "eval/w4_lists.json"))
prof = pd.read_parquet(ROOT / "artifacts/wn_v6/profiles_v6.parquet")
FAV = dict(zip([int(x) for x in eng.ds["item_id"]], eng.ds["interest_count"].fillna(0)))
sf = {r.profile_id: np.median([FAV.get(int(s), 0) for s in r.seed_ids]) for r in prof.itertuples(index=False)}
q25 = np.quantile(list(sf.values()), 0.25); low = {p for p, v in sf.items() if v <= q25}
print(f"저관심 {len(low)} (≤ {q25:,.0f})")
ARMS = ["a0", "a10", "a25"]
fit = lambda role, sel: float(np.mean([grade[role][(r["pid"], r["item"])] >= 2 for r in sel])) if sel else float("nan")
maj = lambda sel: float(np.mean([sum(grade[x][(r["pid"], r["item"])] >= 2 for x in "ABC") >= 2 for r in sel])) if sel else float("nan")
def ils(arm):
    v = []
    for p in prof["profile_id"]:
        ids = L[f"{arm}|{p}"][:50]; M = eng.emb[[eng.id_to_row[i] for i in ids]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); v.append(S[iu].mean())
    return float(np.mean(v))
B, A, IL = {}, {}, {}
print(f"\n### 주 지표 — 역할 B · 대상군 {len(target)} · k=50")
for a in ARMS:
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]
    B[a] = fit("B", s); IL[a] = ils(a)
    print(f"  {a:4} B {B[a]:.4f} · 1-10 {fit('B',[r for r in s if r['part']=='top10']):.4f} · 11-50 {fit('B',[r for r in s if r['part']=='tail']):.4f} · Δ_B {B[a]-B['a0']:+.4f} · ILS {IL[a]:.3f}")
print(f"\n### 안전 — 역할 A · 저관심 {len(low)}")
for a in ARMS:
    A[a] = fit("A", [r for r in rows if r["arm"] == a and r["pid"] in low]); print(f"  {a:4} A 저관심 {A[a]:.4f} Δ {A[a]-A['a0']:+.4f}")
print("\n### 부수 관측 (판정 아님)")
for a in ARMS:
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]; s52 = [r for r in rows if r["arm"] == a]
    print(f"  {a:4} 3역할다수 {maj(s):.4f} · A {fit('A',s):.4f} · C {fit('C',s):.4f} · B 52전체 {fit('B',s52):.4f}")
best = max(("a10", "a25"), key=lambda a: B[a]); d = B[best] - B["a0"]
mono = B["a10"] > B["a0"] and B["a25"] > B["a0"]; safe = A[best] - A["a0"] >= -0.03; e_ok = IL[best] < IL["a0"] + 0.03
print(f"\n최고 {best} · Δ_B {d:+.4f} · 단조성 {'OK' if mono else 'NG'} · 저관심 안전 {'OK' if safe else 'NG'} · (E) {'OK' if e_ok else 'NG'}")
if d >= 0.04 and mono and safe and e_ok: print(f"판정: (A) 채택 — author_w = {'0.10' if best=='a10' else '0.25'}")
elif d >= 0.04 and not (safe and e_ok): print("판정: (C) 기각 — 안전장치 위반. Δ_B 와 무관하게 채택하지 않는다")
elif d >= 0.02 or (d >= 0.04 and not mono): print("판정: (B) 보류 — author_w = 0 유지, 재현 필요. 문턱을 낮추지 않는다")
else: print("판정: (C) 기각 — author_w = 0 유지")
