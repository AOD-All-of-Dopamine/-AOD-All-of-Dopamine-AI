"""T-8 빌드: creator_w 0 / 0.10 재현. T-7 c0·c10 과 같은 목록. seed 8. 채점 재사용 없음.

사전등록 eval/t8_preregister.md.
"""
import json, os, random, sys, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.personalized_retrieve import Engine
eng = Engine(); P = json.load(open(ROOT / "eval/profiles.json"))
ARMS = {"r0": 0.0, "r10": 0.10}
lists = {f"{arm}|{p['pid']}": [int(x) for x in eng.recommend(p["seeds"], k=50, creator_w=w)["item_id"]]
         for arm, w in ARMS.items() for p in P}
# 재현 라운드의 전제: T-7 과 같은 목록
T7 = json.load(open(ROOT / "eval/t7_lists.json"))
for new, old in (("r0", "c0"), ("r10", "c10")):
    same = sum(lists[f"{new}|{p['pid']}"] == T7[f"{old}|{p['pid']}"] for p in P)
    print(f"{new} vs T-7 {old}: 완전 일치 {same}/{len(P)}")
    assert same == len(P), "T-7 목록과 다르다 — 재현 라운드가 아니다"
json.dump(lists, open(ROOT / "eval/t8_lists.json", "w"))
target = json.load(open(ROOT / "eval/t7_key.json"))["target"]   # 같은 정의, 같은 64
ds = eng.ds.set_index("item_id"); meta = {}
for iid, r in ds.iterrows():
    t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
    g = list(r["genres"]) if r["genres"] is not None else []
    meta[int(iid)] = dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str,g)) or '-'} · {', '.join(map(str,t)) or '-'} | {int(r.get('episode_count',0))}화")
SN = {p["pid"]: ", ".join(p["seed_names"]) for p in P}
rng = random.Random(8); rows = []
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
todo = [dict(id=f"u{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=meta[k[1]]["name"], meta=meta[k[1]]["meta"]) for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo, target=target), open(ROOT / "eval/t8_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"t8_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id","seed","cand","meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
t7pairs = {(t["pid"], t["item"]) for t in json.load(open(ROOT / "eval/t7_key.json"))["todo"]}
print("슬롯:", dict(collections.Counter(r["arm"] for r in rows)), f"· 대상군 {len(target)}")
print(f"신규 채점 {len(todo)} · 배치 {nb} · T-7 과 겹치는 쌍 {sum(1 for k in need if k in t7pairs)} (재사용 안 함, 새로 채점)")
