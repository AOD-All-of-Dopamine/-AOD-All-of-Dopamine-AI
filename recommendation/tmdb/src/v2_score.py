"""V-2 집계 — 사전등록 cd9687a9d807c6ef395f52fec94a090a 의 판정 규칙.

주 지표는 **역할 B 단독 · 대상군 50 · k=50 합산**이다. 역할 A 는 롱테일 안전 확인용,
역할 C 는 부수 관측(3역할 다수결이 판정을 뒤집는지)용이며 **채택 판정에 쓰지 않는다.**
"""
import json, glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

K = json.load(open(ROOT / "eval/v2_key.json"))
rows, todo, target = K["rows"], K["todo"], set(K["target"])
key = {t["id"]: t for t in todo}

grade = {r: {} for r in "ABC"}
for role in "ABC":
    for f in sorted(glob.glob(str(ROOT / f"eval/v2/{role}_*.json"))):
        for i, g in json.load(open(f)).items():
            if i in key:
                grade[role][(key[i]["pid"], key[i]["item"])] = int(g)
for role in "ABC":
    print(f"역할 {role} 채점 {len(grade[role]):,}/{len(todo):,}")

miss = [r for r in rows if (r["pid"], r["item"]) not in grade["B"]]
print(f"미채점 슬롯(B) {len(miss)}")
if miss:
    sys.exit("미채점이 남아 비교하지 않는다")

P = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")
ds_meta = pd.read_parquet(ROOT / "artifacts/tmdb_v1/dataset.parquet")
idx = pd.read_parquet(ROOT / "artifacts/tmdb_v1/corpus_index.parquet").sort_values("embedding_row")
ds = ds_meta.set_index("item_id").loc[idx["item_id"].to_numpy()].reset_index()
seed_vote = {r.profile_id: float(np.median([ds.iloc[int(x)]["vote_count"] for x in r.seed_rows]))
             for r in P.itertuples(index=False)}
q25 = np.quantile(list(seed_vote.values()), 0.25)
longtail = {p for p, v in seed_vote.items() if v <= q25}
print(f"롱테일 프로필 {len(longtail)}개 (시드 투표수 중앙 ≤ {q25:,.0f})")

ARMS = ["d0", "d10", "d25"]


def fit(role, sel):
    v = [grade[role][(r["pid"], r["item"])] >= 2 for r in sel]
    return float(np.mean(v)) if v else float("nan")


def maj(sel):
    out = []
    for r in sel:
        k = (r["pid"], r["item"])
        out.append(sum(grade[x][k] >= 2 for x in "ABC") >= 2)
    return float(np.mean(out)) if out else float("nan")


print(f"\n### 주 지표 — 역할 B · 대상군 {len(target)} · k=50")
print(f"{'팔':5} {'B 합산':>8} {'B 1-10':>8} {'B 11-50':>8} {'Δ_B':>8}")
B = {}
for a in ARMS:
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]
    B[a] = fit("B", s)
    h = fit("B", [r for r in s if r["part"] == "top10"])
    t = fit("B", [r for r in s if r["part"] == "tail"])
    print(f"{a:5} {B[a]:8.4f} {h:8.4f} {t:8.4f} {B[a]-B['d0']:+8.4f}")

print(f"\n### 안전 — 역할 A · 롱테일 {len(longtail)}개")
A = {}
for a in ARMS:
    s = [r for r in rows if r["arm"] == a and r["pid"] in longtail]
    A[a] = fit("A", s)
    print(f"{a:5} A 롱테일 {A[a]:.4f}  Δ {A[a]-A['d0']:+.4f}")

print("\n### 부수 관측 (판정 아님)")
print(f"{'팔':5} {'3역할다수':>10} {'A 전체':>8} {'C 전체':>8} {'B 52전체':>9}")
for a in ARMS:
    s = [r for r in rows if r["arm"] == a and r["pid"] in target]
    s52 = [r for r in rows if r["arm"] == a]
    print(f"{a:5} {maj(s):10.4f} {fit('A', s):8.4f} {fit('C', s):8.4f} {fit('B', s52):9.4f}")

# ── 판정 ─────────────────────────────────────────────────────────────────────
best = max(("d10", "d25"), key=lambda a: B[a])
d = B[best] - B["d0"]
mono = B["d10"] > B["d0"] and B["d25"] > B["d0"]
safe = A[best] - A["d0"] >= -0.03
print(f"\n최고 팔 {best} · Δ_B {d:+.4f} · 단조성 {'OK' if mono else 'NG'} · "
      f"롱테일 안전 {'OK' if safe else 'NG'} (Δ_A {A[best]-A['d0']:+.4f})")
if d >= 0.04 and mono and safe:
    print(f"판정: (A) 채택 — director_w = {best}")
elif not safe:
    print("판정: (C) 기각 — 롱테일 안전 조건 위반. Δ_B 와 무관하게 채택하지 않는다")
elif d >= 0.02:
    print("판정: (B) 보류 — director_w = 0 유지, 재현 라운드 필요. 문턱을 낮추지 않는다")
else:
    print("판정: (C) 기각 — director_w = 0 유지. 문턱을 낮춰 다시 재지 않는다")
