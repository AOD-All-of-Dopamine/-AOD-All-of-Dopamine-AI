"""T-11 빌드: 싫어요 감점 d0/w1/w2/w3 · 표적·무관 시나리오 → 들어온/빠진 작품 눈가림 시트. 사전등록 eval/t11_preregister.md."""
import json, os, random, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.config import PRODUCTION
from src.personalized_retrieve import Engine
assert PRODUCTION["dislike_w"] == 0.0 and PRODUCTION["dislike_floor"] == 0.61
eng = Engine(); P = json.load(open(ROOT / "eval/profiles.json"))
ARMS = {"d0": 0.0, "w1": 1.0, "w2": 2.0, "w3": 3.0}
ids = eng.ds["item_id"].astype(int).to_numpy()
nx = lambda seeds, k, seen, dis, w: [int(x) for x in eng.next_page(seeds, k=k, seen=seen, disliked_ids=dis, dislike_w=w)["item_id"]]

T10 = json.load(open(ROOT / "eval/t10_pages.json"))
page1, pool = {}, {}
for p in P:
    page1[p["pid"]] = nx(p["seeds"], 10, [], None, 0.0)
    pool[p["pid"]] = [int(x) for x in eng.recommend(p["seeds"], k=100, exclude=page1[p["pid"]])["item_id"]]
same = sum(page1[p["pid"]] == T10[f"n0|{p['pid']}"][0] for p in P)
print(f"page1 == T-10 n0 1페이지: {same}/{len(P)}")
assert same == len(P), "page1 이 현행 서빙이 아니다"

out = {}
for i, p in enumerate(P):
    pid, seeds = p["pid"], [int(x) for x in p["seeds"]]
    d = page1[pid][0]
    q = next(o for o in P[i + 1:] + P[:i] if o.get("genre") != p.get("genre")
             and page1[o["pid"]][0] not in set(page1[pid]) | set(seeds) | set(pool[pid]))
    du = page1[q["pid"]][0]
    s = eng.emb @ eng.emb[eng.row[d]]
    nn = [int(ids[t]) for t in np.argsort(-s) if int(ids[t]) != d][:50]
    rec = dict(d=d, du=du, nn50=nn, page1=page1[pid], t={}, u={})
    for arm, w in ARMS.items():
        rec["t"][arm] = nx(seeds, 20, page1[pid], [d], w)
        rec["u"][arm] = nx(seeds, 20, page1[pid], [du], w)
        assert len(rec["t"][arm]) == 20 and len(rec["u"][arm]) == 20, (pid, arm)
    out[pid] = rec
json.dump(out, open(ROOT / "eval/t11_pages.json", "w"))

rows = []   # 팔마다 들어온(in) / 빠진(out) 작품
for pid, r in out.items():
    base = r["t"]["d0"]
    for arm in ("w1", "w2", "w3"):
        cur = r["t"][arm]
        rows += [dict(arm=arm, pid=pid, item=i, kind="in") for i in cur if i not in base]
        rows += [dict(arm=arm, pid=pid, item=i, kind="out") for i in base if i not in cur]
ds = eng.ds.set_index("item_id")
def card(i):   # T-10 시트와 같은 형식
    r = ds.loc[int(i)]
    t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
    g = list(r["genres"]) if r["genres"] is not None else []
    return dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str,g)) or '-'} · {', '.join(map(str,t)) or '-'} | {int(r.get('episode_count',0))}화")
SN = {p["pid"]: ", ".join(p["seed_names"]) for p in P}
rng = random.Random(11)
need = list(dict.fromkeys((r["pid"], r["item"]) for r in rows)); rng.shuffle(need)
todo = [dict(id=f"k{n:04d}", pid=p, item=i, seed=SN[p], cand=card(i)["name"], meta=card(i)["meta"]) for n, (p, i) in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/t11_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"t11_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
(ROOT / "eval/t11").mkdir(exist_ok=True)
for arm in ("w1", "w2", "w3"):
    print(f"  {arm}: 들어온 {sum(1 for r in rows if r['arm']==arm and r['kind']=='in')} · 빠진 {sum(1 for r in rows if r['arm']==arm and r['kind']=='out')}")
print(f"채점 행 {len(rows)} · 신규 채점 {len(todo)} · 배치 {nb}")
