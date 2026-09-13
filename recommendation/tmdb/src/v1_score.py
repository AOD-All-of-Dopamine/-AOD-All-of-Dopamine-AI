"""V-1 집계 — 사전등록 309e7d0d… 판정 규칙."""
import json, glob, collections, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

K = json.load(open(ROOT / "eval/v1_key.json"))
rows, todo = K["rows"], K["todo"]
key = {t["id"]: t for t in todo}
votes = collections.defaultdict(list)
for f in sorted(glob.glob(str(ROOT / "eval/v1/[ABC]_*.json"))):
    for i, g in json.load(open(f)).items():
        if i in key:
            votes[i].append(int(g) >= 2)
maj, unan = {}, 0
for i, v in votes.items():
    if len(v) >= 3:
        maj[i] = sum(v) >= 2
        unan += (len(set(v)) == 1)
print(f"신규 판정 {len(maj)}/{len(todo)} · 만장일치 {unan/max(len(maj),1):.2f}")

pair = {(key[i]["pid"], key[i]["item"]): m for i, m in maj.items()}
miss = 0
for r in rows:
    if "maj" in r:
        continue
    k = (r["pid"], r["item"])
    if k in pair:
        r["maj"] = pair[k]
    else:
        miss += 1
print(f"미채점 슬롯 {miss}")
if miss:
    sys.exit("미채점이 남아 비교하지 않는다")
json.dump(dict(rows=rows), open(ROOT / "eval/v1_graded.json", "w"), ensure_ascii=False)

P_ = lambda rs: float(np.mean([r["maj"] for r in rs])) if rs else float("nan")
prof = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")

from src.personalized_retrieve import build_components
from src.config import PRODUCTION
FIX = {k: v for k, v in PRODUCTION.items() if k != "strategy"}
_, ret, _, ranker = build_components(**{**FIX, "require_korean": False})
ds = ranker.dataset
L = json.load(open(ROOT / "eval/v1_lists.json"))
vote_of = dict(zip(ds["item_id"], ds["vote_count"]))
has_ko = dict(zip(ds["item_id"], ds["overview"].fillna("").str.contains(r"[가-힣]")))

# 롱테일 프로필 = 시드 투표수 중앙이 하위 25%
seed_vote = {r.profile_id: float(np.median([ds.iloc[int(x)]["vote_count"] for x in r.seed_rows]))
             for r in prof.itertuples(index=False)}
q25 = np.quantile(list(seed_vote.values()), 0.25)
longtail = {p for p, v in seed_vote.items() if v <= q25}
print(f"롱테일 프로필 {len(longtail)}개 (시드 투표수 중앙 ≤ {q25:,.0f})")

emb = np.asarray(ret.embeddings, dtype=np.float32)
row_of = dict(zip(ds["item_id"], ds["row"]))


def ils(arm):
    v = []
    for r in prof.itertuples(index=False):
        ids = L[f"{arm}|{r.profile_id}"][:50]
        M = emb[[row_of[i] for i in ids]]
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        S = M @ M.T
        iu = np.triu_indices(len(M), 1)
        v.append(S[iu].mean())
    return float(np.mean(v))


V = {}
print("\n### 팔별 (프로필 52 · 3역할 다수결)")
for a in ("k_on", "k_off"):
    s = [r for r in rows if r["arm"] == a]
    h = [r for r in s if r["part"] == "top10"]
    t = [r for r in s if r["part"] == "tail"]
    lt = [r for r in s if r["pid"] in longtail]
    ko = np.mean([has_ko.get(i, False) for r in prof.itertuples(index=False)
                  for i in L[f"{a}|{r.profile_id}"][:50]])
    uniq = len({i for r in prof.itertuples(index=False) for i in L[f"{a}|{r.profile_id}"]})
    V[a] = (P_(s), P_(lt), ils(a))
    print(f"  {a:6} 1-10 {P_(h):.3f} · 11-50 {P_(t):.3f} · 합산 {P_(s):.4f} · "
          f"롱테일 {P_(lt):.3f} · ILS {ils(a):.3f} · 한글 {ko*100:.0f}% · 고유 {uniq:,}")

d = V["k_off"][0] - V["k_on"][0]
print(f"\nΔ (k_off − k_on) 합산 {d:+.4f} · 롱테일 {V['k_off'][1]-V['k_on'][1]:+.4f} · "
      f"ILS {V['k_off'][2]-V['k_on'][2]:+.4f}")
if d >= -0.02:
    print("판정: (A) 비용 없음 → 새 기준선 채택, 나머지 축 재판정으로")
elif d > -0.05:
    print("판정: (B) 비용 있음 → 기준선 채택하되 문서에 비용 명기")
else:
    print("판정: (C) 비용 큼 → 값은 유지하되 사용자에게 선택지를 되묻는다")
