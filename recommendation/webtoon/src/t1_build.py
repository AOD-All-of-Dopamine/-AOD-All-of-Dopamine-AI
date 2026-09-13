"""T-1: rep_v1 vs rep_v2 목록 + 눈가림 슬롯. 사전등록 2f06edaa… 대로 — 보정 전부 0, seed 1, 5+5."""
import json, random, os, sys, collections
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
OFF = dict(pop_boost=0.0, star_boost=0.0, tag_w=0.0, hub_lambda=0.0)
P = json.load(open(ROOT / "eval/profiles.json"))

def engine(art):
    os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / art)
    for m in [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]: del sys.modules[m]
    from src.personalized_retrieve import Engine
    return Engine()

lists, meta = {}, {}
for arm, art in (("rep_v1", "artifacts/rep_v1"), ("rep_v2", "artifacts/wt_v1")):
    eng = engine(art)
    for p in P:
        df = eng.recommend(p["seeds"], k=50, **OFF)
        lists[f"{arm}|{p['pid']}"] = [int(x) for x in df["item_id"]]
    ds = eng.ds.set_index("item_id")
    if not meta:
        for iid, r in ds.iterrows():
            t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
            g = list(r["genres"]) if r["genres"] is not None else []
            meta[int(iid)] = dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str, g)) or '-'} · {', '.join(map(str, t)) or '-'} | {int(r.get('episode_count', 0))}화")
    print(arm, "완료", flush=True)
json.dump(lists, open(ROOT / "eval/t1_lists.json", "w"))
SN = {p["pid"]: ", ".join(p["seed_names"]) for p in P}

rng = random.Random(1); rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i + 1) for i, x in enumerate(L[:10])]; tail = [(x, i + 11) for i, x in enumerate(L[10:50])]
    pick = [(x, pos, "top10") for x, pos in rng.sample(head, min(5, len(head)))]
    pick += [(x, pos, "tail") for x, pos in rng.sample(tail, min(5, len(tail)))]
    for x, pos, part in pick: rows.append(dict(arm=arm, pid=pid, item=x, pos=pos, part=part))
need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k not in seen: seen.add(k); need.append(k)
rng.shuffle(need)
todo = [dict(id=f"t{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=meta[k[1]]["name"], meta=meta[k[1]]["meta"]) for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/t1_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"t1_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
ov = [len(set(lists[f"rep_v1|{p['pid']}"]) & set(lists[f"rep_v2|{p['pid']}"])) / 50 for p in P]
print(f"슬롯 {len(rows)} · 고유 쌍 {len(todo)} · 배치 {nb} · 팔 간 top-50 겹침 중앙 {sorted(ov)[len(ov)//2]*100:.0f}%")
