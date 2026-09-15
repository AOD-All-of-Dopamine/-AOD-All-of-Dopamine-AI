"""W-5 빌드: 시리즈 키 publisher(s0) / author(s1). 기계 판정 + 진입·대조·이탈 시트. 사전등록 eval/w5_preregister.md."""
import json, os, random, re, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.environ["AOD_ARTIFACTS"] = str(ROOT / "artifacts/wn_v6")
from src.wn_eval import Engine
eng = Engine()
P = pd.read_parquet(ROOT / "artifacts/wn_v6/profiles_v6.parquet")
ds = pd.read_parquet(ROOT / "artifacts/wn_v6/dataset.parquet").set_index("item_id")
ARMS = {"s0": "publisher", "s1": "author"}
seeds = {r.profile_id: [int(s) for s in r.seed_ids] for r in P.itertuples(index=False)}
lists = {f"{a}|{p}": [int(x) for x in eng.recommend(seeds[p], k=50, series_by=by)["item_id"]] for a, by in ARMS.items() for p in P.profile_id}
W4 = json.load(open(ROOT / "eval/w4_lists.json"))
same = sum(lists[f"s0|{p}"] == [int(x) for x in W4[f"a0|{p}"]] for p in P.profile_id)
print(f"s0 vs W-4 a0(현행 서빙): 완전 일치 {same}/{len(P)}")
assert same == len(P), "s0 가 현행 서빙 목록이 아니다"
json.dump(lists, open(ROOT / "eval/w5_lists.json", "w"))

norm = lambda s: re.sub(r"[\s\W_]+", "", str(s).lower())
TA = {int(i): (norm(r["name"]), str(r["author"] or "").strip()) for i, r in ds.iterrows()}
def mech(arm, k):
    echo = dup = 0; prof = 0
    for p in P.profile_id:
        l = lists[f"{arm}|{p}"][:k]; st = {TA[s] for s in seeds[p]}
        e = sum(TA[i] in st for i in l); c = collections.Counter(TA[i] for i in l); d = sum(v - 1 for v in c.values())
        echo += e; dup += d; prof += (e + d) > 0
    return echo, dup, prof
print("\n### 판정 1 — 기계 (52프로필)")
for a in ARMS:
    e10, d10, p10 = mech(a, 10); e50, d50, _ = mech(a, 50)
    print(f"  {a}: M-echo @10 {e10} · @50 {e50} · M-dup @10 {d10} · @50 {d50} · 해당 프로필@10 {p10}")
e50, d50, _ = mech("s1", 50); mech_ok = e50 == 0 and d50 == 0
print(f"  기계 판정: {'통과 (0/0)' if mech_ok else '실패 — (C). 채점하지 않는다'}")
if not mech_ok: sys.exit(0)

E, Lv, Ctl, diff_drop = [], [], [], []
rng = random.Random(51)
for p in P.profile_id:
    a, b = lists[f"s0|{p}"], lists[f"s1|{p}"]; sa, sb = set(a), set(b)
    ent = [(p, i, b.index(i) + 1) for i in b if i not in sa]; lv = [(p, i, a.index(i) + 1) for i in a if i not in sb]
    E += ent; Lv += lv
    pool = [(p, i, b.index(i) + 1) for i in b[30:50] if i in sa]
    Ctl += rng.sample(pool, min(len(ent), len(pool)))
    keep_ta = {TA[i] for i in b} | {TA[s] for s in seeds[p]}
    for _, i, pos in lv:
        if TA[i] not in keep_ta: diff_drop.append((p, pos, ds.loc[i, "name"], ds.loc[i, "author"]))
print(f"\n|E| {len(E)} · |Lv| {len(Lv)} · |Ctl| {len(Ctl)} · 프로필(E>0) {len({x[0] for x in E})}")
lv_echo = sum(TA[i] in {TA[s] for s in seeds[p]} for p, i, _ in Lv)
print(f"이탈 중 시드 에코 {lv_echo} · 제목이 다른데 빠진 작품 {len(diff_drop)}")
for x in diff_drop: print("   ", x)
print(f"E 위치: 1-10 {sum(pos <= 10 for _, _, pos in E)} · 11-50 {sum(pos > 10 for _, _, pos in E)}")

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
todo = [dict(id=f"p{n:04d}", pid=p, item=i, seed=SN[p], cand=m(i)["name"], meta=m(i)["meta"]) for n, (p, i) in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/w5_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"w5_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
(ROOT / "eval/w5").mkdir(exist_ok=True)
print(f"신규 채점 {len(todo)} · 배치 {nb}")
