"""X-25: 4팔 통합 목록 + 채점 슬롯. 사전등록 80a3fbc4… 대로."""
import json, random, sys, os, collections
import pandas as pd
ROOT="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation"; os.chdir(ROOT); sys.path.insert(0,"crossdomain")
import xseed; xseed.load_all(); EPS=xseed.wn_episodes()
SCR="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
T=pd.read_parquet("tmdb/artifacts/tmdb_v1/dataset.parquet")
ARMS=["k5","k1","k3_div","k3_red"]
S25=json.load(open("crossdomain/x25_seeds.json"))

lists={}
if os.path.exists("crossdomain/x25_lists.json"): lists=json.load(open("crossdomain/x25_lists.json"))
else:
    for e in S25:
        for arm in ARMS:
            seeds={k:[int(x) for x in v] for k,v in e["arms"][arm].items() if v}
            pl=xseed.platform_lists(seeds,"M6","a",50)
            lists[f"{arm}|{e['pid']}"]=xseed.mix_m6(pl,seeds,50,EPS)
        print(e["pid"],flush=True)
    json.dump(lists,open("crossdomain/x25_lists.json","w"))

S=pd.read_parquet("steam/artifacts/tags_full/dataset.parquet").set_index("steam_appid")
W=xseed._MODS["wn"]["eng"].ds.set_index("item_id")
def num(v,d=0):
    try:
        if v is None or pd.isna(v): return d
    except Exception: pass
    return v
def card(plat,item):
    if plat=="steam":
        r=S.loc[int(item)]; r=r.iloc[0] if isinstance(r,pd.DataFrame) else r
        tg=r.get("tags"); tg=list(tg)[:5] if tg is not None and len(tg) else []
        n=num(r.get("recommendations_total"))
        return dict(name=str(r["name"]),meta=" · ".join(map(str,tg)) or "-",stat=f"리뷰 {int(n):,}" if n else "리뷰 수 미보고")
    if plat=="tmdb":
        r=T.iloc[int(item)]; g=r["genres"]; g=list(g) if g is not None and len(g) else []
        return dict(name=str(r["name"]),meta=("영화" if r["media"]=="movie" else "TV")+" · "+(", ".join(map(str,g)) or "-"),
                    stat=f"평점 {float(num(r['vote_average'],0.0)):.1f} · 투표 {int(num(r['vote_count'])):,}")
    r=W.loc[int(item)]; r=r.iloc[0] if isinstance(r,pd.DataFrame) else r
    g=r["genres"]; g=list(g) if g is not None and len(g) else []; rt=float(num(r["rating"],0.0))
    return dict(name=str(r["name"]),meta=(", ".join(map(str,g)) or "-")+f" · {r['author']}",
                stat=(f"평점 {rt:.1f} · " if rt>0 else "평점 없음 · ")+f"관심 {int(num(r['interest_count'])):,} · {int(num(r['episode_count']))}화")
cards={}
for slots in lists.values():
    for s in slots:
        k=f"{s['plat']}|{s['item']}"
        if k not in cards: cards[k]=card(s["plat"],s["item"])

# 시드 문자열: **표시는 k5 기준으로 통일**하지 않는다 — 팔마다 실제 준 시드를 보여야 채점이 정직하다
# 부록(x25_addendum.md) 결정 (b): **모든 팔에서 v2 전체 5개 시드를 보여준다.**
# 사용자의 취향은 팔에 따라 변하지 않는다. 우리가 재는 것은 "적게 알려줬을 때도 그 취향을 맞히는가"다.
SN={}
for e in S25:
    nm=e["names"]
    SN[e["pid"]]=" | ".join(f"{c}: {', '.join(nm[plat])}"
                           for plat,c in (("steam","S"),("tmdb","T"),("wn","W")) if nm.get(plat))

rng=random.Random(25); rows=[]
for key,slots in sorted(lists.items()):
    arm,pid=key.split("|")
    head=[(s,i+1) for i,s in enumerate(slots[:10])]; tail=[(s,i+11) for i,s in enumerate(slots[10:50])]
    pick=[(s,p,"top10") for s,p in rng.sample(head,min(5,len(head)))]
    pick+=[(s,p,"tail") for s,p in rng.sample(tail,min(5,len(tail)))]
    for s,pos,part in pick:
        rows.append(dict(arm=arm,pid=pid,plat=s["plat"],item=str(s["item"]),pos=pos,part=part))

# 은행: X-24 (같은 프롬프트·같은 날). 시드 표시가 팔마다 다르므로 **은행 키에 팔을 넣지 않는다** —
# 은행은 (pid, plat, item) 단위이고, X-24 참조군은 k5 시드(=v2 전체)로 채점됐다.
bank={}
K24=json.load(open("eval/x24_key.json")); key24={t["id"]:t for t in K24["todo"]}
import glob
votes=collections.defaultdict(list)
for f in sorted(glob.glob("eval/x24/[ABC]_*.json")):
    for i,g in json.load(open(f)).items():
        if i in key24: votes[i].append(int(g)>=2)
for i,v in votes.items():
    if len(v)>=3:
        t=key24[i]; bank[(t["pid"],t["plat"],t["item"])]=sum(v)>=2
print(f"X-24 은행 {len(bank)} 쌍")

need=[]; seen=set()
for r in rows:
    k=(r["pid"],r["plat"],r["item"])
    if k in bank: r["maj"]=bank[k]
    elif k not in seen: seen.add(k); need.append((k,r["arm"]))
rng.shuffle(need)
PC={"steam":"게임","tmdb":"영상","wn":"소설"}
todo=[]
for i,(k,arm) in enumerate(need):
    c=cards[f"{k[1]}|{k[2]}"]
    todo.append(dict(id=f"z{i:04d}",pid=k[0],plat=k[1],item=k[2],seed=SN[k[0]],
                     cand=c["name"],meta=f"{PC[k[1]]} · {c['meta'][:40]} | {c['stat']}"))
json.dump(dict(rows=rows,todo=todo),open("eval/x25_key.json","w"),ensure_ascii=False)
nb=(len(todo)+59)//60
for b in range(nb):
    open(f"{SCR}/x25_b{b}.json","w").write(json.dumps(
        [{k:r[k] for k in ("id","seed","cand","meta")} for r in todo[b*60:(b+1)*60]],ensure_ascii=False))
by=collections.Counter(r["arm"] for r in rows); hv=collections.Counter(r["arm"] for r in rows if "maj" in r)
for a in ARMS: print(f"  {a:<7} 슬롯 {by[a]} · 은행 {hv[a]}")
print(f"신규 채점 {len(todo)} 쌍 · 배치 {nb}")
