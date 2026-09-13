"""X-18 val: 페르소나 v2 에서 kw_w=0.2 로 TMDB 만 재생성 → M6 통합 → TMDB 슬롯 표본.
Steam·웹소설 목록은 r1_top100.json 에 저장된 확정 목록을 그대로 쓴다(랭커 미로드).
"""
import os, sys, json, random, collections, numpy as np, pandas as pd
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
KW=0.2
TOP=json.load(open("crossdomain/r1_top100.json"))
PERS=json.load(open("crossdomain/personas_v2.json"))
wn=pd.read_parquet("/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4/dataset.parquet",
                   columns=["item_id","episode_count"])
EPS=dict(zip(wn["item_id"].astype(int), wn["episode_count"].fillna(0).astype(int)))
sys.path.insert(0,"crossdomain"); from mix import RULES
ROOT=os.getcwd()
sys.path.insert(0,os.path.join(ROOT,"tmdb")); os.chdir("tmdb")
os.environ.setdefault("AOD_ARTIFACTS","artifacts/tmdb_v1")
from src.personalized_retrieve import build_components, recommend
from src.config import PRODUCTION
comps=build_components(**{k:v for k,v in PRODUCTION.items() if k!="strategy"}, kw_w=KW)
os.chdir(ROOT)
mixed={}
for e in PERS:
    if not e["seeds"].get("tmdb"): continue
    lists={}; seeds={}
    for pl in ("steam","wn"):
        k=f"val|{pl}|{e['pid']}"
        if k in TOP and TOP[k]["top100"]:
            lists[pl]=TOP[k]["top100"][:50]; seeds[pl]=len(TOP[k]["seeds"])
    df=recommend([int(x) for x in e["seeds"]["tmdb"]], components=comps, top_n=50,
                 strategy=PRODUCTION["strategy"], postprocess_on=True)
    lists["tmdb"]=[int(x) for x in df["row"].head(50)]; seeds["tmdb"]=len(e["seeds"]["tmdb"])
    mixed[e["pid"]]=[dict(plat=p,item=str(it),rank=rk) for p,it,rk in
                     RULES["M6"](lists,seeds,{p:0.6 for p in lists},k=50,episodes=EPS)]
    print(e["pid"],flush=True)
json.dump(mixed,open("crossdomain/x18_val_mixed.json","w"))
# 은행
bank={}
for r in json.load(open("eval/x16_graded.json"))["rows"]:
    if "maj" in r: bank[(r["pid"],r["plat"],str(r["item"]))]=r["maj"]
for f in ("eval/x14_graded.json","eval/x15_graded.json","eval/r0_graded.json"):
    for r in json.load(open(f)):
        if "maj" in r: bank.setdefault((r["pid"],r["plat"],str(r["item"])),bool(r["maj"]))
for r in json.load(open("eval/r2wn_graded.json"))["rows"]:
    if "maj" in r: bank.setdefault((r["pid"],"wn",str(r["item"])),r["maj"])
rng=random.Random(20260831003); rows=[]
for pid,slots in mixed.items():
    sel=[(s,i+1) for i,s in enumerate(slots) if s["plat"]=="tmdb"]
    pick=[(s,p) for s,p in sel if p<=10]
    tail=[(s,p) for s,p in sel if p>10]
    pick+=rng.sample(tail,min(10,len(tail)))
    for s,pos in pick:
        r=dict(pid=pid,plat="tmdb",item=s["item"],rank=s["rank"],pos=pos)
        k=(pid,"tmdb",s["item"])
        if k in bank: r["maj"]=bank[k]
        rows.append(r)
SN={r["pid"]:r["seed"] for r in json.load(open("eval/x15_graded.json"))}
cards=json.load(open("crossdomain/x16_cards.json")); ds=comps[3].dataset
def card(item):
    if f"tmdb|{item}" in cards: return cards[f"tmdb|{item}"]
    r=ds.iloc[int(item)]; g=r["genres"]; g=list(g) if g is not None and len(g) else []
    return dict(name=str(r["name"]), meta=("영화" if r["media"]=="movie" else "TV")+" · "+(", ".join(map(str,g)) or "-"),
                stat=f"평점 {float(r['vote_average'] or 0):.1f} · 투표 {int(r['vote_count'] or 0):,}")
need=[]; seen=set()
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["item"])
    if k in seen: continue
    seen.add(k); need.append(r)
todo=[]
for i,r in enumerate(need):
    c=card(r["item"])
    todo.append(dict(id=f"v{i:03d}",pid=r["pid"],item=r["item"],seed=SN[r["pid"]],cand=c["name"],meta=f"영상 · {c['meta'][:40]} | {c['stat']}"))
json.dump(dict(rows=rows,todo=todo,kw_w=KW),open("eval/x18_val_key.json","w"),ensure_ascii=False)
S="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
nb=(len(todo)+59)//60
for b in range(nb): open(f"{S}/x18v_b{b}.json","w").write(json.dumps([{k:t[k] for k in ("id","seed","cand","meta")} for t in todo[b*60:(b+1)*60]],ensure_ascii=False))
print(f"TMDB 슬롯 {len(rows)} · 은행 {sum(1 for r in rows if 'maj' in r)} · 신규 {len(todo)} · 배치 {nb}")
