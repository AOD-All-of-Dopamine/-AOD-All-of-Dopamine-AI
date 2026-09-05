"""T-3: tag_w 0 / 0.2 / 0.4 on rep_v2, pop_boost 0. w0 = T-1 rep_v2 목록. seed 3. 은행 = T-1 ∪ T-2."""
import json, random, os, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
eng = Engine(); P = json.load(open(ROOT / "eval/profiles.json"))
OFF = dict(pop_boost=0.0, star_boost=0.0, hub_lambda=0.0)
lists = {f"w0|{k.split('|')[1]}": v for k, v in json.load(open(ROOT / "eval/t1_lists.json")).items() if k.startswith("rep_v2|")}
for arm, tw in (("w2", 0.2), ("w4", 0.4)):
    for p in P:
        lists[f"{arm}|{p['pid']}"] = [int(x) for x in eng.recommend(p["seeds"], k=50, tag_w=tw, **OFF)["item_id"]]
json.dump(lists, open(ROOT / "eval/t3_lists.json", "w"))
E = eng.emb; row = eng.row
def ils(ids):
    M = E[[row[i] for i in ids[:50]]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); return float(S[iu].mean())
for arm in ("w0", "w2", "w4"):
    v = [ils(lists[f"{arm}|{p['pid']}"]) for p in P]
    ov = np.mean([len(set(lists[f"{arm}|{p['pid']}"]) & set(lists[f"w0|{p['pid']}"])) / 50 for p in P])
    print(f"  {arm}: ILS@50 {np.mean(v):.3f} · w0 와 겹침 {ov*100:.0f}% · 고유 {len({i for p in P for i in lists[f'{arm}|{p['pid']}']}):,}")
ds = eng.ds.set_index("item_id"); meta = {}
for iid, r in ds.iterrows():
    t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
    g = list(r["genres"]) if r["genres"] is not None else []
    meta[int(iid)] = dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str,g)) or '-'} · {', '.join(map(str,t)) or '-'} | {int(r.get('episode_count',0))}화")
SN = {p["pid"]: ", ".join(p["seed_names"]) for p in P}
bank = {}
for f in ("t1_graded.json", "t2_graded.json"):
    for r in json.load(open(ROOT / "eval" / f))["rows"]:
        if "maj" in r: bank[(r["pid"], r["item"])] = r["maj"]
rng = random.Random(3); rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i+1) for i, x in enumerate(L[:10])]; tail = [(x, i+11) for i, x in enumerate(L[10:50])]
    pick = [(x, pos, "top10") for x, pos in rng.sample(head, min(5, len(head)))] + [(x, pos, "tail") for x, pos in rng.sample(tail, min(5, len(tail)))]
    for x, pos, part in pick: rows.append(dict(arm=arm, pid=pid, item=x, pos=pos, part=part))
need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k in bank: r["maj"] = bank[k]
    elif k not in seen: seen.add(k); need.append(k)
rng.shuffle(need)
todo = [dict(id=f"v{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=meta[k[1]]["name"], meta=meta[k[1]]["meta"]) for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/t3_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"t3_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id","seed","cand","meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
by = collections.Counter(r["arm"] for r in rows); hv = collections.Counter(r["arm"] for r in rows if "maj" in r)
print("슬롯:", {a: f"{by[a]} (은행 {hv[a]})" for a in by}); print(f"신규 채점 {len(todo)} · 배치 {nb}")
