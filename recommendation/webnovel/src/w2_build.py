"""W-2: strategy top2_mean / mean / max on wn_v6, pop 0. s_t2 = W-1 p0 목록. seed 8. 은행 = W-1."""
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
lists = {}
for arm, st in (("s_t2", "top2_mean"), ("s_mean", "mean"), ("s_max", "max")):
    for _, r in P.iterrows():
        seeds = [int(s) for s in r["seed_ids"]]
        lists[f"{arm}|{r.profile_id}"] = [int(x) for x in eng.recommend(seeds, k=50, pop_boost=0.0, strategy=st)["item_id"]]
json.dump(lists, open(ROOT / "eval/w2_lists.json", "w"))
E, row = eng.emb, eng.id_to_row
FAV = ds["interest_count"].fillna(0).to_dict()
def ils(ids):
    M = E[[row[i] for i in ids[:50]]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); return float(S[iu].mean())
for arm in ("s_t2", "s_mean", "s_max"):
    v = [ils(lists[f"{arm}|{r.profile_id}"]) for _, r in P.iterrows()]
    ov = np.mean([len(set(lists[f"{arm}|{r.profile_id}"]) & set(lists[f"s_t2|{r.profile_id}"])) / 50 for _, r in P.iterrows()])
    fav = np.median([FAV.get(i, 0) for _, r in P.iterrows() for i in lists[f"{arm}|{r.profile_id}"][:50]])
    uniq = len({i for _, r in P.iterrows() for i in lists[f"{arm}|{r.profile_id}"]})
    print(f"  {arm}: ILS@50 {np.mean(v):.3f} · s_t2 와 겹침 {ov*100:.0f}% · 관심수 중앙 {fav:,.0f} · 고유 {uniq:,}")
meta = {}
for iid, r in ds.iterrows():
    g = list(r["genres"]) if r["genres"] is not None else []
    syn = str(r["synopsis"])[:70].replace("\n", " ")
    meta[int(iid)] = dict(name=str(r["name"]), meta=f"웹소설 · {', '.join(map(str,g)) or '-'} · {int(r.get('episode_count') or 0)}화 | {syn}")
SN = {r.profile_id: ", ".join(str(ds.loc[int(s), "name"]) for s in r["seed_ids"]) for _, r in P.iterrows()}
rng = random.Random(8); rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i + 1) for i, x in enumerate(L[:10])]; tail = [(x, i + 11) for i, x in enumerate(L[10:50])]
    pick = [(x, p, "top10") for x, p in rng.sample(head, min(5, len(head)))] + [(x, p, "tail") for x, p in rng.sample(tail, min(5, len(tail)))]
    for x, p, part in pick: rows.append(dict(arm=arm, pid=pid, item=x, pos=p, part=part))
BANK = {(r["pid"], r["item"]): r["maj"] for r in json.load(open(ROOT / "eval/w1_graded.json"))["rows"] if "maj" in r}
need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k in BANK: r["maj"] = BANK[k]
    elif k not in seen: seen.add(k); need.append(k)
rng.shuffle(need)
todo = [dict(id=f"w{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=meta[k[1]]["name"], meta=meta[k[1]]["meta"]) for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/w2_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"w2_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
by = collections.Counter(r["arm"] for r in rows); hv = collections.Counter(r["arm"] for r in rows if "maj" in r)
print("슬롯:", {a: f"{by[a]} (은행 {hv[a]})" for a in by}); print(f"신규 채점 {len(todo)} · 배치 {nb}")
