"""T-6: star_boost 0 / 0.10 / 0.20 on rep_v2, pop 0, tag_w 0.2, top2_mean, hub 0. b0 = T-3 w2 목록. seed 6. 은행 = T-1 ∪ T-5."""
import json, random, os, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
eng = Engine(); P = json.load(open(ROOT / "eval/profiles.json"))
FIX = dict(pop_boost=0.0, hub_lambda=0.0, tag_w=0.2, strategy="top2_mean")
lists = {f"b0|{k.split('|')[1]}": v for k, v in json.load(open(ROOT / "eval/t3_lists.json")).items() if k.startswith("w2|")}
chk = [int(x) for x in eng.recommend(P[0]["seeds"], k=50, star_boost=0.0, **FIX)["item_id"]]
assert chk == lists[f"b0|{P[0]["pid"]}"][:50], "기준선 재현 실패 — 설정 기본값이 새어 들어갔다"
for arm, sb in (("b10", 0.10), ("b20", 0.20)):
    for p in P:
        lists[f"{arm}|{p['pid']}"] = [int(x) for x in eng.recommend(p["seeds"], k=50, star_boost=sb, **FIX)["item_id"]]
json.dump(lists, open(ROOT / "eval/t6_lists.json", "w"))
E = eng.emb; row = eng.row
FAV = dict(zip(eng.ds["item_id"], pd.to_numeric(eng.ds["favorite_count"], errors="coerce").fillna(0)))
def ils(ids):
    M = E[[row[i] for i in ids[:50]]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); return float(S[iu].mean())
for arm in ("b0", "b10", "b20"):
    v = [ils(lists[f"{arm}|{p['pid']}"]) for p in P]
    ov = np.mean([len(set(lists[f"{arm}|{p['pid']}"]) & set(lists[f"b0|{p["pid"]}"])) / 50 for p in P])
    same = sum(lists[f"{arm}|{p['pid']}"] == lists[f"b0|{p["pid"]}"] for p in P)
    import collections as _c; cnt = _c.Counter(i for p in P for i in lists[f"{arm}|{p['pid']}"][:50]); hubs = sum(1 for c in cnt.values() if c >= 20); favm = float(np.median([FAV[i] for p in P for i in lists[f"{arm}|{p['pid']}"][:50]]))
    print(f"  {arm}: 관심수 중앙 {favm:,.0f} · ILS@50 {np.mean(v):.3f} · s_t2 와 겹침 {ov*100:.0f}% · 동일 목록 {same} · 고유 {len({i for p in P for i in lists[f'{arm}|{p['pid']}']}):,}")
ds = eng.ds.set_index("item_id"); meta = {}
for iid, r in ds.iterrows():
    t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
    g = list(r["genres"]) if r["genres"] is not None else []
    meta[int(iid)] = dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str,g)) or '-'} · {', '.join(map(str,t)) or '-'} | {int(r.get('episode_count',0))}화")
SN = {p["pid"]: ", ".join(p["seed_names"]) for p in P}
bank = {}
for f in ("t1_graded.json", "t2_graded.json", "t3_graded.json", "t4_graded.json", "t5_graded.json"):
    for r in json.load(open(ROOT / "eval" / f))["rows"]:
        if "maj" in r: bank[(r["pid"], r["item"])] = r["maj"]
rng = random.Random(6); rows = []
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
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/t6_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"t6_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id","seed","cand","meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
by = collections.Counter(r["arm"] for r in rows); hv = collections.Counter(r["arm"] for r in rows if "maj" in r)
print("슬롯:", {a: f"{by[a]} (은행 {hv[a]})" for a in by}); print(f"신규 채점 {len(todo)} · 배치 {nb}")
