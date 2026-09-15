"""T-10 빌드: tag_drop_genre False(n0) / True(n1). next_page 5회(seen 누적) → 5페이지 전수 + 1페이지 5슬롯. 사전등록 eval/t10_preregister.md."""
import json, os, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_WT_ARTIFACTS"] = str(ROOT / "artifacts/wt_v1")
from src.config import PRODUCTION
from src.personalized_retrieve import Engine
assert PRODUCTION["tag_drop_genre"] is False and PRODUCTION["tag_w"] == 0.2
eng = Engine(); P = json.load(open(ROOT / "eval/profiles.json"))
ARMS = {"n0": False, "n1": True}
pages = {}
for arm, tdg in ARMS.items():
    for p in P:
        seen, pg = [], []
        for _ in range(5):
            ids = [int(x) for x in eng.next_page(p["seeds"], k=10, seen=seen, tag_drop_genre=tdg)["item_id"]]
            seen += ids; pg.append(ids)
        pages[f"{arm}|{p['pid']}"] = pg
T9 = json.load(open(ROOT / "eval/t9_lists.json"))
same = sum(pages[f"n0|{p['pid']}"][0] == T9[f"g2|{p['pid']}"][:10] for p in P)
print(f"n0 1페이지 == T-9 g2 top-10: {same}/{len(P)}")
assert same == len(P), "n0 가 현행 서빙이 아니다"
json.dump(pages, open(ROOT / "eval/t10_pages.json", "w"))
for pn in (0, 4):
    ov = sum(len(set(pages[f"n0|{p['pid']}"][pn]) & set(pages[f"n1|{p['pid']}"][pn])) for p in P) / (10 * len(P))
    print(f"  {pn + 1}페이지 팔 간 겹침 {ov:.3f}")
rng = random.Random(10); rows = []
for arm in ARMS:
    for p in P:
        pg = pages[f"{arm}|{p['pid']}"]
        for j, i in enumerate(pg[4]): rows.append(dict(arm=arm, pid=p["pid"], item=i, page=5, pos=41 + j))
        for j in sorted(rng.sample(range(len(pg[0])), min(5, len(pg[0])))): rows.append(dict(arm=arm, pid=p["pid"], item=pg[0][j], page=1, pos=j + 1))
ds = eng.ds.set_index("item_id")
def card(i):   # T-9 시트와 같은 형식
    r = ds.loc[int(i)]
    t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
    g = list(r["genres"]) if r["genres"] is not None else []
    return dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str,g)) or '-'} · {', '.join(map(str,t)) or '-'} | {int(r.get('episode_count',0))}화")
SN = {p["pid"]: ", ".join(p["seed_names"]) for p in P}
need = list(dict.fromkeys((r["pid"], r["item"]) for r in rows)); rng.shuffle(need)
todo = [dict(id=f"u{n:04d}", pid=p, item=i, seed=SN[p], cand=card(i)["name"], meta=card(i)["meta"]) for n, (p, i) in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/t10_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"t10_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
(ROOT / "eval/t10").mkdir(exist_ok=True)
print(f"슬롯 {len(rows)} · 신규 채점 {len(todo)} · 배치 {nb}")
