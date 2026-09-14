"""V-3 빌드: director_w 0 / 0.10 재현. V-2 d0·d10 과 같은 목록. seed 14. 채점 재사용 없음.

사전등록 eval/v3_preregister.md.
"""
import json, random, sys, collections
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI"
           "/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")

from src.personalized_retrieve import build_components, recommend
from src.config import PRODUCTION

ARMS = {"e0": 0.0, "e10": 0.10}
P = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")
FIX = {k: v for k, v in PRODUCTION.items() if k != "strategy"}
comp = build_components(**{**FIX, "director_w": 0.001})     # _dirs 적재용, 계수는 아래서 바꾼다
ranker = comp[3]
lists = {}
for arm, w in ARMS.items():
    ranker.director_w = w                                   # PRODUCTION 은 0.10 — e0 은 명시적 0
    for r in P.itertuples(index=False):
        out = recommend([int(x) for x in r.seed_rows], components=comp,
                        strategy=PRODUCTION["strategy"], top_n=50)
        lists[f"{arm}|{r.profile_id}"] = [str(x) for x in out["item_id"]]

V2 = json.load(open(ROOT / "eval/v2_lists.json"))
for new, old in (("e0", "d0"), ("e10", "d10")):
    same = sum(lists[f"{new}|{p}"] == V2[f"{old}|{p}"] for p in P["profile_id"])
    print(f"{new} vs V-2 {old}: 완전 일치 {same}/{len(P)}")
    assert same == len(P), "V-2 목록과 다르다 — 재현 라운드가 아니다"
json.dump(lists, open(ROOT / "eval/v3_lists.json", "w"))
target = json.load(open(ROOT / "eval/v2_key.json"))["target"]

ds = ranker.dataset
name_of = dict(zip(ds["item_id"], ds["name"]))
kw_of = dict(zip(ds["item_id"], ds["keywords"]))
g_of = dict(zip(ds["item_id"], ds["genres"]))
ov_of = dict(zip(ds["item_id"], ds["overview"]))


def _lst(v):
    if v is None:
        return []
    return list(v) if hasattr(v, "__len__") else []


def card(i):
    g = _lst(g_of.get(i)); kw = _lst(kw_of.get(i))[:6]
    syn = str(ov_of.get(i) or "")[:80].replace("\n", " ")
    return dict(name=str(name_of.get(i, i)),
                meta=f"{'영화' if str(i).startswith('movie') else '드라마'} · "
                     f"{', '.join(map(str, g)) or '-'} · {', '.join(map(str, kw)) or '-'} | {syn}")


SN = {r.profile_id: ", ".join(str(ds.iloc[int(x)]["name"]) for x in r.seed_rows) for r in P.itertuples(index=False)}
rng = random.Random(14)
rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i + 1) for i, x in enumerate(L[:10])]
    tail = [(x, i + 11) for i, x in enumerate(L[10:50])]
    pick = ([(x, p, "top10") for x, p in rng.sample(head, min(5, len(head)))] +
            [(x, p, "tail") for x, p in rng.sample(tail, min(5, len(tail)))])
    for x, p, part in pick:
        rows.append(dict(arm=arm, pid=pid, item=x, pos=p, part=part))
need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k not in seen:
        seen.add(k); need.append(k)
rng.shuffle(need)
todo = [dict(id=f"x{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=card(k[1])["name"], meta=card(k[1])["meta"])
        for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo, target=target), open(ROOT / "eval/v3_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"v3_b{b}.json").write_text(json.dumps(
        [{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
v2pairs = {(t["pid"], t["item"]) for t in json.load(open(ROOT / "eval/v2_key.json"))["todo"]}
print("슬롯:", dict(collections.Counter(r["arm"] for r in rows)), f"· 대상군 {len(target)}")
print(f"신규 채점 {len(todo)} · 배치 {nb} · V-2 와 겹치는 쌍 {sum(1 for k in need if k in v2pairs)} (재사용 안 함)")
