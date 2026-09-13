"""T-7 빌드: creator_w 0 / 0.10 / 0.25. 다른 축 PRODUCTION 고정. seed 7. 은행 재사용 없음(역할별 등급이 필요).

사전등록 8454695cbbf78c62772f916622d429b0.
"""
import json, os, random, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
from src.personalization.personalized_ranker import creators_of
eng = Engine(); P = json.load(open(ROOT / "eval/profiles.json"))
ARMS = {"c0": 0.0, "c10": 0.10, "c25": 0.25}
lists = {}
for arm, w in ARMS.items():
    for p in P:
        lists[f"{arm}|{p['pid']}"] = [int(x) for x in eng.recommend(p["seeds"], k=50, creator_w=w)["item_id"]]
# 기준선 재현: c0 는 프로덕션 기본값 호출과 같아야 한다
for p in P[:5]:
    assert lists[f"c0|{p['pid']}"] == [int(x) for x in eng.recommend(p["seeds"], k=50)["item_id"]], "기준선 누수"
json.dump(lists, open(ROOT / "eval/t7_lists.json", "w"))

IDS = [int(x) for x in eng.ds["item_id"]]
Cid = dict(zip(IDS, [creators_of(r) for r in eng.ds.itertuples(index=False)]))
target = []
for p in P:
    sd = frozenset().union(*[Cid[int(s)] for s in p["seeds"]]); ss = {int(s) for s in p["seeds"]}
    if any(cs & sd for i, cs in Cid.items() if i not in ss): target.append(p["pid"])
print(f"대상군 {len(target)}/{len(P)} · 비대상 {[p['pid'] for p in P if p['pid'] not in target]}")
E, row = eng.emb, eng.row
def ils(ids):
    M = E[[row[i] for i in ids[:50]]]; S = M @ M.T; iu = np.triu_indices(len(M), 1); return float(S[iu].mean())
for arm in ARMS:
    v = [ils(lists[f"{arm}|{p['pid']}"]) for p in P]
    ov = np.mean([len(set(lists[f"{arm}|{p['pid']}"]) & set(lists[f"c0|{p['pid']}"])) / 50 for p in P])
    hit = np.mean([sum(1 for i in lists[f"{arm}|{p['pid']}"][:10] if Cid[i] & frozenset().union(*[Cid[int(s)] for s in p["seeds"]])) for p in P])
    print(f"  {arm}: ILS@50 {np.mean(v):.3f} · c0 과 겹침 {ov*100:.0f}% · 작가일치@10 {hit:.2f} · 고유 {len({i for p in P for i in lists[f'{arm}|{p['pid']}']}):,}")
ds = eng.ds.set_index("item_id"); meta = {}
for iid, r in ds.iterrows():
    t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
    g = list(r["genres"]) if r["genres"] is not None else []
    meta[int(iid)] = dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str,g)) or '-'} · {', '.join(map(str,t)) or '-'} | {int(r.get('episode_count',0))}화")
SN = {p["pid"]: ", ".join(p["seed_names"]) for p in P}
rng = random.Random(7); rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i+1) for i, x in enumerate(L[:10])]; tail = [(x, i+11) for i, x in enumerate(L[10:50])]
    pick = [(x, pos, "top10") for x, pos in rng.sample(head, min(5, len(head)))] + [(x, pos, "tail") for x, pos in rng.sample(tail, min(5, len(tail)))]
    for x, pos, part in pick: rows.append(dict(arm=arm, pid=pid, item=x, pos=pos, part=part))
need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k not in seen: seen.add(k); need.append(k)
rng.shuffle(need)
todo = [dict(id=f"t{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=meta[k[1]]["name"], meta=meta[k[1]]["meta"]) for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo, target=target), open(ROOT / "eval/t7_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"t7_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id","seed","cand","meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
print("슬롯:", dict(collections.Counter(r["arm"] for r in rows))); print(f"신규 채점 {len(todo)} · 배치 {nb}")
