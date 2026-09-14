"""V-4 빌드: genre_w 0 / 0.40. h4 = 현행 서빙(V-3 e0). seed 15. 채점 재사용 없음. 사전등록 eval/v4_preregister.md."""
import json, random, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
from src.personalized_retrieve import build_components, recommend
from src.config import PRODUCTION
assert PRODUCTION["genre_w"] == 0.40 and PRODUCTION["director_w"] == 0.0
ARMS = {"h0": 0.0, "h4": 0.40}
P = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")
comp = build_components(**{k: v for k, v in PRODUCTION.items() if k != "strategy"})   # genre 집합 적재
ranker = comp[3]
lists = {}
for arm, w in ARMS.items():
    ranker.genre_w = w
    for r in P.itertuples(index=False):
        out = recommend([int(x) for x in r.seed_rows], components=comp, strategy=PRODUCTION["strategy"], top_n=50)
        lists[f"{arm}|{r.profile_id}"] = [str(x) for x in out["item_id"]]
V3 = json.load(open(ROOT / "eval/v3_lists.json"))
same = sum(lists[f"h4|{p}"] == V3[f"e0|{p}"] for p in P["profile_id"])
print(f"h4 vs V-3 e0(현행 서빙): 완전 일치 {same}/{len(P)}")
assert same == len(P), "h4 가 현행 서빙 목록이 아니다"
ov = np.mean([len(set(lists[f"h0|{p}"]) & set(lists[f"h4|{p}"])) / 50 for p in P["profile_id"]])
print(f"  h0 vs h4 top-50 겹침 {ov*100:.1f}%")
json.dump(lists, open(ROOT / "eval/v4_lists.json", "w"))
ds = ranker.dataset
name_of = dict(zip(ds["item_id"], ds["name"])); kw_of = dict(zip(ds["item_id"], ds["keywords"]))
g_of = dict(zip(ds["item_id"], ds["genres"])); ov_of = dict(zip(ds["item_id"], ds["overview"]))
def _lst(v):
    if v is None: return []
    return list(v) if hasattr(v, "__len__") else []
def card(i):
    g = _lst(g_of.get(i)); kw = _lst(kw_of.get(i))[:6]
    syn = str(ov_of.get(i) or "")[:80].replace("\n", " ")
    return dict(name=str(name_of.get(i, i)), meta=f"{'영화' if str(i).startswith('movie') else '드라마'} · {', '.join(map(str, g)) or '-'} · {', '.join(map(str, kw)) or '-'} | {syn}")
SN = {r.profile_id: ", ".join(str(ds.iloc[int(x)]["name"]) for x in r.seed_rows) for r in P.itertuples(index=False)}
rng = random.Random(15); rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i + 1) for i, x in enumerate(L[:10])]; tail = [(x, i + 11) for i, x in enumerate(L[10:50])]
    pick = [(x, p, "top10") for x, p in rng.sample(head, min(5, len(head)))] + [(x, p, "tail") for x, p in rng.sample(tail, min(5, len(tail)))]
    for x, p, part in pick: rows.append(dict(arm=arm, pid=pid, item=x, pos=p, part=part))
need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k not in seen: seen.add(k); need.append(k)
rng.shuffle(need)
todo = [dict(id=f"y{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]], cand=card(k[1])["name"], meta=card(k[1])["meta"]) for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo, target=list(P["profile_id"])), open(ROOT / "eval/v4_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"v4_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
print("슬롯:", dict(collections.Counter(r["arm"] for r in rows)), f"· 신규 채점 {len(todo)} · 배치 {nb}")
