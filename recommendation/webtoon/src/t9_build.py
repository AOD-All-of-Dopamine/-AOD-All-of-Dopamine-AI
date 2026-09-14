"""T-9 빌드: tag_w 0 / 0.2 재현. g2 = 현행 서빙(T-8 r0). seed 9. 채점 재사용 없음. 사전등록 eval/t9_preregister.md."""
import json, os, random, sys, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
eng = Engine(); P = json.load(open(ROOT / "eval/profiles.json"))
ARMS = {"g0": 0.0, "g2": 0.2}
lists = {f"{arm}|{p['pid']}": [int(x) for x in eng.recommend(p["seeds"], k=50, tag_w=w)["item_id"]]
         for arm, w in ARMS.items() for p in P}
T8 = json.load(open(ROOT / "eval/t8_lists.json"))
same = sum(lists[f"g2|{p['pid']}"] == T8[f"r0|{p['pid']}"] for p in P)
print(f"g2 vs T-8 r0(현행 서빙): 완전 일치 {same}/{len(P)}")
assert same == len(P), "g2 가 현행 서빙 목록이 아니다"
T3 = json.load(open(ROOT / "eval/t3_lists.json"))
for new, old in (("g0", "w0"), ("g2", "w2")):
    s = sum(lists[f"{new}|{p['pid']}"] == [int(x) for x in T3[f"{old}|{p['pid']}"]] for p in P)
    ov = np.mean([len(set(lists[f"{new}|{p['pid']}"]) & set(int(x) for x in T3[f"{old}|{p['pid']}"][:50])) / 50 for p in P])
    print(f"  (부수) {new} vs T-3 {old}: 완전 일치 {s}/{len(P)} · 겹침 {ov*100:.1f}%")
json.dump(lists, open(ROOT / "eval/t9_lists.json", "w"))
target = [p["pid"] for p in P]
ds = eng.ds.set_index("item_id"); meta = {}
for iid, r in ds.iterrows():
    t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
    g = list(r["genres"]) if r["genres"] is not None else []
    meta[int(iid)] = dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str,g)) or '-'} · {', '.join(map(str,t)) or '-'} | {int(r.get('episode_count',0))}화")
SN = {p["pid"]: ", ".join(p["seed_names"]) for p in P}
rng = random.Random(9); rows = []
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
todo = [dict(id=f"g{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=meta[k[1]]["name"], meta=meta[k[1]]["meta"]) for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo, target=target), open(ROOT / "eval/t9_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"t9_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id","seed","cand","meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
print("슬롯:", dict(collections.Counter(r["arm"] for r in rows)), f"· 신규 채점 {len(todo)} · 배치 {nb}")
