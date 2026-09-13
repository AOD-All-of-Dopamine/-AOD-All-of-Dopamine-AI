"""R-2wn: 페르소나 v2 중 웹소설 시드가 있는 26명의 M6 통합 목록을 wn_v4/wn_v5 로 만들고
웹소설 슬롯 표본을 뽑는다. v4 채점은 은행 재사용, v5 신규만 눈가림 배치로.
"""
import json, random, os, sys, collections
import numpy as np, pandas as pd
ROOT="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation"; os.chdir(ROOT); sys.path.insert(0,"crossdomain")
import xseed
from pathlib import Path
SCR="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
xseed.load_all()
xseed._load_platform("wn5", xseed.WN, xseed.WN/"artifacts/wn_v5")

def rank_wn_with(name, seeds, k=50):
    own=[int(x) for x in seeds.get("wn",[])]
    with xseed._swap(name) as m:
        eng=m["eng"]
        sd={int(s): eng.emb[eng.id_to_row[int(s)]].copy() for s in own}
        keys=list(sd); V=np.array([sd[key] for key in keys],dtype=np.float32)
        sim=V@eng.emb.T
        s=sim.mean(axis=0) if sim.shape[0]<=2 else np.sort(sim,axis=0)[-2:].mean(axis=0)
        final=s*(1+eng.pop_pct*m["PROD"]["pop_boost"])
        df=eng.ds[["item_id","name"]].copy(); df["seed_similarity"]=s; df["final_score"]=final
        df["dominant_seed"]=[keys[i] for i in sim.argmax(axis=0)]
        df=df[~df["item_id"].isin(set(own))].sort_values("final_score",ascending=False).reset_index(drop=True)
        df["rank"]=range(1,len(df)+1)
        out=m["drop_seed_series"](df.head(k*8),eng.ds,own); out=eng.mmr_by_seed(out,1.0)
        return [int(x) for x in m["pp"](out,eng.ds,top_n=k).head(k)["item_id"]]

PERS=[e for e in json.load(open("crossdomain/personas_v2.json")) if e["seeds"].get("wn")]
EPS=xseed.wn_episodes(); X16=json.load(open("crossdomain/x16_lists.json"))
mixed={}; wnlists={}
for e in PERS:
    pid=e["pid"]
    pl={p: xseed.RANK[p](e["seeds"],"M6","a",50) for p in ("steam","tmdb") if e["seeds"].get(p)}
    for ver,name in (("v4","wn"),("v5","wn5")):
        wl=rank_wn_with(name,e["seeds"]); wnlists[f"{ver}|{pid}"]=wl
        mixed[f"{ver}|{pid}"]=xseed.mix_m6({**pl,"wn":wl},e["seeds"],50,EPS)
    # v4 재현 확인: x16 val M6 와 일치해야 한다
    a={(s["plat"],s["item"]) for s in mixed[f"v4|{pid}"]}; b={(s["plat"],s["item"]) for s in X16[f"val|M6|{pid}"]}
    if len(a&b)/max(len(b),1)<0.999: print("경고: v4 재현 불일치",pid,len(a&b)/len(b))
    print(pid,flush=True)
json.dump(mixed,open("crossdomain/r2wn_mixed.json","w"))

# ILS@50 (v4 임베딩 공통 잣대) · 태그 0개 노출률
eng4=xseed._MODS["wn"]["eng"]
def ils(ids):
    rows=[eng4.id_to_row[i] for i in ids if i in eng4.id_to_row]; E=eng4.emb[rows]
    E=E/np.linalg.norm(E,axis=1,keepdims=True); S=E@E.T; n=len(rows)
    return (S.sum()-n)/(n*(n-1)) if n>1 else float("nan")
sys.path.insert(0,str(xseed.WN)); os.environ["AOD_ARTIFACTS"]="artifacts/wn_v5"
d5=pd.read_parquet(xseed.WN/"artifacts/wn_v5/dataset.parquet").set_index("item_id")
no_tag=set(d5[~d5["semantic_text"].str.contains("태그:")].index.astype(int))
for ver in ("v4","v5"):
    ilss=[ils(wnlists[f"{ver}|{e['pid']}"]) for e in PERS]
    exp=[sum(1 for i in wnlists[f"{ver}|{e['pid']}"] if i in no_tag)/len(wnlists[f"{ver}|{e['pid']}"]) for e in PERS]
    print(f"{ver}: ILS@50 평균 {np.nanmean(ilss):.3f} · 태그0 노출 {np.mean(exp):.2f}")

# 슬롯 표본: 웹소설 슬롯만, 통합 1-10 전수 + 11-50 프로필당 5
bank={}
K=json.load(open("eval/x16_graded.json"))
for r in K["rows"]:
    if "maj" in r: bank[(r["pid"],r["plat"],str(r["item"]))]=r["maj"]
for f in ("eval/x14_graded.json","eval/x15_graded.json","eval/r0_graded.json"):
    for r in json.load(open(f)):
        if "maj" in r: bank.setdefault((r["pid"],r["plat"],str(r["item"])),bool(r["maj"]))
rng=random.Random(20260831001); rows=[]
for key,slots in mixed.items():
    ver,pid=key.split("|")
    wn=[(s,i+1) for i,s in enumerate(slots) if s["plat"]=="wn"]
    pick=[(s,p) for s,p in wn if p<=10]
    tail=[(s,p) for s,p in wn if p>10]
    pick+=rng.sample(tail,min(5,len(tail)))
    for s,pos in pick:
        r=dict(ver=ver,pid=pid,plat="wn",item=s["item"],rank=s["rank"],pos=pos)
        k=(pid,"wn",s["item"])
        if k in bank: r["maj"]=bank[k]
        rows.append(r)
SN={}
for r in json.load(open("eval/x15_graded.json")): SN[r["pid"]]=r["seed"]
cards=json.load(open("crossdomain/x16_cards.json"))
def card5(item):
    if f"wn|{item}" in cards: return cards[f"wn|{item}"]
    r=d5.loc[int(item)]
    g=r["genres"]; g=list(g) if g is not None and len(g) else []
    rt=float(r["rating"]) if r["rating"] and r["rating"]>0 else 0
    return dict(name=str(r["name"]),meta=(", ".join(map(str,g)) or "-")+f" · {r['author']}",
                stat=(f"평점 {rt:.1f} · " if rt>0 else "평점 없음 · ")+f"관심 {int(r['interest_count'] or 0):,} · {int(r['episode_count'] or 0)}화")
need=[]; seen=set()
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["item"])
    if k in seen: continue
    seen.add(k); need.append(r)
rng.shuffle(need); todo=[]
for i,r in enumerate(need):
    c=card5(r["item"])
    todo.append(dict(id=f"w{i:03d}",pid=r["pid"],plat="wn",item=r["item"],seed=SN[r["pid"]],cand=c["name"],meta=f"소설 · {c['meta'][:40]} | {c['stat']}"))
json.dump(dict(rows=rows,todo=todo),open("eval/r2wn_key.json","w"),ensure_ascii=False)
nb=(len(todo)+59)//60
for b in range(nb): open(f"{SCR}/r2wn_b{b}.json","w").write(json.dumps([{k:t[k] for k in ("id","seed","cand","meta")} for t in todo[b*60:(b+1)*60]],ensure_ascii=False))
by=collections.Counter(r["ver"] for r in rows); have=collections.Counter(r["ver"] for r in rows if "maj" in r)
print(f"슬롯 v4 {by['v4']}(은행 {have['v4']}) · v5 {by['v5']}(은행 {have['v5']}) · 신규 {len(todo)} · 배치 {nb}")
