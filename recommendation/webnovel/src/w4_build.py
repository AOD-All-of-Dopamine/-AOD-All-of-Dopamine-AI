"""W-4 빌드: author_w 0 / 0.10 / 0.25 on wn_v6. 다른 축 PRODUCTION 고정. seed 10. 은행 재사용 없음(역할별 등급이 필요).

사전등록 b64f2540eb50784fc891fe1151173fae.
"""
import json, os, random, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_ARTIFACTS"] = str(ROOT / "artifacts/wn_v6")
from src.wn_eval import Engine
eng = Engine()
P = pd.read_parquet(ROOT / "artifacts/wn_v6/profiles_v6.parquet")
ds = pd.read_parquet(ROOT / "artifacts/wn_v6/dataset.parquet").set_index("item_id")
ARMS = {"a0": 0.0, "a10": 0.10, "a25": 0.25}
lists = {}
for arm, w in ARMS.items():
    for _, r in P.iterrows():
        seeds = [int(s) for s in r["seed_ids"]]
        lists[f"{arm}|{r.profile_id}"] = [int(x) for x in eng.recommend(seeds, k=50, pop_boost=0.0, author_w=w)["item_id"]]
for _, r in P.head(5).iterrows():
    seeds = [int(s) for s in r["seed_ids"]]
    assert lists[f"a0|{r.profile_id}"] == [int(x) for x in eng.recommend(seeds, k=50)["item_id"]], "기준선 누수"
json.dump(lists, open(ROOT / "eval/w4_lists.json", "w"))
A = dict(zip([int(x) for x in eng.ds["item_id"]], eng._author))
cnt = collections.Counter(a for a in A.values() if a)
target = []
for _, r in P.iterrows():
    seeds = [int(s) for s in r["seed_ids"]]; sa = {A[s] for s in seeds} - {""}
    other = sum(cnt[a] for a in sa) - sum(1 for s in seeds if A[s] in sa)
    if other > 0: target.append(r.profile_id)
print(f"대상군 {len(target)}/{len(P)} · 비대상 {[p for p in P['profile_id'] if p not in target]}")
E, row = eng.emb, eng.id_to_row
def ils(ids):
    M = E[[row[i] for i in ids[:50]]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); return float(S[iu].mean())
for arm in ARMS:
    v = [ils(lists[f"{arm}|{p}"]) for p in P["profile_id"]]
    ov = np.mean([len(set(lists[f"{arm}|{p}"]) & set(lists[f"a0|{p}"])) / 50 for p in P["profile_id"]])
    hit = np.mean([sum(1 for i in lists[f"{arm}|{r.profile_id}"][:10] if A[i] and A[i] in {A[int(s)] for s in r["seed_ids"]}) for _, r in P.iterrows()])
    print(f"  {arm}: ILS@50 {np.mean(v):.3f} · a0 와 겹침 {ov*100:.0f}% · 작가일치@10 {hit:.2f} · 고유 {len({i for p in P['profile_id'] for i in lists[f'{arm}|{p}']}):,}")
meta = {}
for iid, r in ds.iterrows():
    g = list(r["genres"]) if r["genres"] is not None else []
    syn = str(r["synopsis"])[:70].replace("\n", " ")
    meta[int(iid)] = dict(name=str(r["name"]), meta=f"웹소설 · {', '.join(map(str,g)) or '-'} · {int(r.get('episode_count') or 0)}화 | {syn}")
SN = {r.profile_id: ", ".join(str(ds.loc[int(s), "name"]) for s in r["seed_ids"]) for _, r in P.iterrows()}
rng = random.Random(10); rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i + 1) for i, x in enumerate(L[:10])]; tail = [(x, i + 11) for i, x in enumerate(L[10:50])]
    pick = [(x, p, "top10") for x, p in rng.sample(head, min(5, len(head)))] + [(x, p, "tail") for x, p in rng.sample(tail, min(5, len(tail)))]
    for x, p, part in pick: rows.append(dict(arm=arm, pid=pid, item=x, pos=p, part=part))
need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k not in seen: seen.add(k); need.append(k)
rng.shuffle(need)
todo = [dict(id=f"n{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=meta[k[1]]["name"], meta=meta[k[1]]["meta"]) for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo, target=target), open(ROOT / "eval/w4_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"w4_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
print("슬롯:", dict(collections.Counter(r["arm"] for r in rows))); print(f"신규 채점 {len(todo)} · 배치 {nb}")
