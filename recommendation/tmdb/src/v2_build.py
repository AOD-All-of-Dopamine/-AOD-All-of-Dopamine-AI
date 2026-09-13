"""V-2 빌드: director_w 0 / 0.10 / 0.25. 다른 축은 PRODUCTION 고정. seed 12. 은행 재사용 없음.

사전등록 cd9687a9d807c6ef395f52fec94a090a.
"""
import json, random, sys, collections
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI"
           "/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")

from src.personalized_retrieve import build_components, recommend
from src.config import PRODUCTION

ARMS = {"d0": 0.0, "d10": 0.10, "d25": 0.25}
P = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")
FIX = {k: v for k, v in PRODUCTION.items() if k != "strategy"}

# director_w 를 켠 컴포넌트를 한 번만 만들고 랭커의 계수만 바꾼다 — 임베딩 233MB 를
# 팔마다 다시 읽지 않는다. `_dirs` 는 director_w 가 켜져 있어야 적재되므로 0.001 로 연다.
comp = build_components(**{**FIX, "director_w": 0.001})
loader, ret, agg, ranker = comp
ds = ranker.dataset
dirs_of = ranker._dirs
row_of = dict(zip(ds["item_id"], ds["row"]))

lists = {}
for arm, w in ARMS.items():
    ranker.director_w = w
    for r in P.itertuples(index=False):
        out = recommend([int(x) for x in r.seed_rows], components=comp,
                        strategy=PRODUCTION["strategy"], top_n=50)
        lists[f"{arm}|{r.profile_id}"] = [str(x) for x in out["item_id"]]
json.dump(lists, open(ROOT / "eval/v2_lists.json", "w"))

# ── 대상군: 시드 감독의 다른 작품이 코퍼스에 1편 이상 (등급 무관, 사전등록대로) ──
by_dir = collections.Counter()
for s in dirs_of:
    for p in s:
        by_dir[p] += 1
target = []
for r in P.itertuples(index=False):
    sr = [int(x) for x in r.seed_rows]
    sd = set().union(*[dirs_of[i] for i in sr]) if sr else set()
    extra = sum(by_dir[p] for p in sd) - sum(1 for i in sr for p in dirs_of[i] if p in sd)
    if extra > 0:
        target.append(r.profile_id)
print(f"대상군 {len(target)}/52 · 비대상 "
      f"{[p for p in P['profile_id'] if p not in target]}")

emb = np.asarray(ret.embeddings, dtype=np.float32)
name_of = dict(zip(ds["item_id"], ds["name"]))
vote_of = dict(zip(ds["item_id"], ds["vote_count"]))


def ils(ids):
    M = emb[[row_of[i] for i in ids[:50]]]
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    iu = np.triu_indices(len(M), 1)
    return float((M @ M.T)[iu].mean())


for arm in ARMS:
    v = [ils(lists[f"{arm}|{p}"]) for p in P["profile_id"]]
    ov = np.mean([len(set(lists[f"{arm}|{p}"]) & set(lists[f"d0|{p}"])) / 50
                  for p in P["profile_id"]])
    hit = np.mean([sum(1 for i in lists[f"{arm}|{p}"][:10]
                       if dirs_of[row_of[i]] & set().union(
                           *[dirs_of[int(x)] for x in P.set_index('profile_id').loc[p, 'seed_rows']]))
                   for p in P["profile_id"]])
    uniq = len({i for p in P["profile_id"] for i in lists[f"{arm}|{p}"]})
    print(f"  {arm:4} ILS@50 {np.mean(v):.3f} · d0 과 겹침 {ov*100:.0f}% · "
          f"감독일치@10 {hit:.2f} · 고유 {uniq:,}")

# ── 채점 시트 ────────────────────────────────────────────────────────────────
kw_of = dict(zip(ds["item_id"], ds["keywords"]))
g_of = dict(zip(ds["item_id"], ds["genres"]))
ov_of = dict(zip(ds["item_id"], ds["overview"]))


def _lst(v):
    """parquet 의 list 열은 numpy 배열로 온다 — `or []` 가 모호성 오류를 낸다."""
    if v is None:
        return []
    return list(v) if hasattr(v, "__len__") else []


def card(i):
    g = _lst(g_of.get(i))
    kw = _lst(kw_of.get(i))[:6]
    syn = str(ov_of.get(i) or "")[:80].replace("\n", " ")
    return dict(name=str(name_of.get(i, i)),
                meta=f"{'영화' if str(i).startswith('movie') else '드라마'} · "
                     f"{', '.join(map(str, g)) or '-'} · {', '.join(map(str, kw)) or '-'} | {syn}")


SN = {r.profile_id: ", ".join(str(ds.iloc[int(x)]["name"]) for x in r.seed_rows)
      for r in P.itertuples(index=False)}

rng = random.Random(12)
rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i + 1) for i, x in enumerate(L[:10])]
    tail = [(x, i + 11) for i, x in enumerate(L[10:50])]
    pick = ([(x, p, "top10") for x, p in rng.sample(head, min(5, len(head)))] +
            [(x, p, "tail") for x, p in rng.sample(tail, min(5, len(tail)))])
    for x, p, part in pick:
        rows.append(dict(arm=arm, pid=pid, item=x, pos=p, part=part))

need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k not in seen:
        seen.add(k)
        need.append(k)
rng.shuffle(need)
todo = [dict(id=f"w{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]],
             cand=card(k[1])["name"], meta=card(k[1])["meta"])
        for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo, target=target),
          open(ROOT / "eval/v2_key.json", "w"), ensure_ascii=False)

nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"v2_b{b}.json").write_text(json.dumps(
        [{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]],
        ensure_ascii=False))
print("슬롯:", dict(collections.Counter(r["arm"] for r in rows)))
print(f"신규 채점 {len(todo)} · 배치 {nb} · 3역할이면 {len(todo)*3:,}건")
