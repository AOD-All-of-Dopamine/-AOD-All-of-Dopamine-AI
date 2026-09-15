"""W-6 빌드: next_page drop_excluded_series False(n0) / True(n1). 기계 판정 + 진입·대조·이탈 시트. 사전등록 eval/w6_preregister.md."""
import json, os, random, re, sys, collections
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_ARTIFACTS"] = str(ROOT / "artifacts/wn_v6")
from src.config import PRODUCTION
from src.personalized_retrieve import next_page, build_components
from src.wn_eval import Engine
assert PRODUCTION["series_by"] == "author"
comps = build_components(); eng = Engine()
P = pd.read_parquet(ROOT / "artifacts/wn_v6/profiles_v6.parquet")
ds = pd.read_parquet(ROOT / "artifacts/wn_v6/dataset.parquet").set_index("item_id")
norm = lambda s: re.sub(r"[\s\W_]+", "", str(s).lower())
TA = {int(i): (norm(r["name"]), str(r["author"] or "").strip()) for i, r in ds.iterrows()}
seeds = {r.profile_id: [int(s) for s in r.seed_ids] for r in P.itertuples(index=False)}
ARMS = {"n0": False, "n1": True}
pages = {}
for arm, dx in ARMS.items():
    for p in P.profile_id:
        seen = set(); pg = []
        for _ in range(3):
            ids = [int(x) for x in next_page(seeds[p], seen_ids=seen, page_size=10, components=comps, drop_excluded_series=dx)["item_id"]]
            seen |= set(ids); pg.append(ids)
        pages[f"{arm}|{p}"] = pg
json.dump(pages, open(ROOT / "eval/w6_pages.json", "w"))

def mech(arm):
    echo = within = cross = p1eq = 0
    for p in P.profile_id:
        pg = pages[f"{arm}|{p}"]; st = {TA[s] for s in seeds[p]}; prev = set()
        echo += sum(TA[i] in st for x in pg for i in x)
        for x in pg:
            c = collections.Counter(TA[i] for i in x); within += sum(v - 1 for v in c.values())
            cross += sum(TA[i] in prev for i in x); prev |= {TA[i] for i in x}
        p1eq += pg[0] == [int(i) for i in eng.recommend(seeds[p], k=10)["item_id"]]
    return echo, within, cross, p1eq
print("### 판정 1 — 기계 (52프로필 × 3페이지)")
R = {a: mech(a) for a in ARMS}
for a, (e, w, c, q) in R.items():
    print(f"  {a}: 에코 {e} · 페이지 안 중복 {w} · 페이지 간 재등장 {c} · 1페이지 == Engine top-10 {q}/52")
assert R["n0"][:3] == (33, 0, 72), f"n0 가 사전등록 직전 측정을 재현하지 않는다: {R['n0'][:3]}"
mech_ok = R["n1"][:3] == (0, 0, 0)
print(f"  기계 판정: {'통과 (0/0/0)' if mech_ok else '실패 — (C). 채점하지 않는다'}")
if not mech_ok: sys.exit(0)

E, Lv, Ctl = [], [], []
rng = random.Random(61)
for p in P.profile_id:
    a = [i for x in pages[f"n0|{p}"] for i in x]; b = pages[f"n1|{p}"]; bf = [i for x in b for i in x]
    sa, sb = set(a), set(bf)
    ent = [(p, i, bf.index(i) + 1) for i in bf if i not in sa]; lv = [(p, i, a.index(i) + 1) for i in a if i not in sb]
    E += ent; Lv += lv
    pool = [(p, i, 20 + b[2].index(i) + 1) for i in b[2] if i in sa]
    Ctl += rng.sample(pool, min(len(ent), len(pool)))
print(f"\n|E| {len(E)} · |Lv| {len(Lv)} · |Ctl| {len(Ctl)} · 프로필(E>0) {len({x[0] for x in E})}")
print(f"E 페이지: 1 {sum(pos <= 10 for *_, pos in E)} · 2 {sum(10 < pos <= 20 for *_, pos in E)} · 3 {sum(pos > 20 for *_, pos in E)}")

meta = {}
def m(i):
    if i not in meta:
        r = ds.loc[i]; g = list(r["genres"]) if r["genres"] is not None else []
        syn = str(r["synopsis"])[:70].replace("\n", " ")
        meta[i] = dict(name=str(r["name"]), meta=f"웹소설 · {', '.join(map(str,g)) or '-'} · {int(r.get('episode_count') or 0)}화 | {syn}")
    return meta[i]
SN = {p: ", ".join(str(ds.loc[s, "name"]) for s in seeds[p]) for p in P.profile_id}
rows = [dict(set=s, pid=p, item=i, pos=pos) for s, X in (("E", E), ("Ctl", Ctl), ("Lv", Lv)) for p, i, pos in X]
need = list(dict.fromkeys((r["pid"], r["item"]) for r in rows)); rng.shuffle(need)
todo = [dict(id=f"q{n:04d}", pid=p, item=i, seed=SN[p], cand=m(i)["name"], meta=m(i)["meta"]) for n, (p, i) in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/w6_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"w6_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
(ROOT / "eval/w6").mkdir(exist_ok=True)
print(f"신규 채점 {len(todo)} · 배치 {nb}")
