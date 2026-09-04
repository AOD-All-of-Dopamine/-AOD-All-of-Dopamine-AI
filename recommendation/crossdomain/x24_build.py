"""X-24: 근거 페르소나 50(M6·C1a) + 참조군 페르소나 v2 32(M6) 목록·채점 슬롯 생성.
사전등록 59cf944f… 대로: 프로필·변형마다 1-10 중 5 + 11-50 중 5, RNG seed 24, 은행 재사용 없음."""
import json, random, sys, os, collections
import pandas as pd
ROOT="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation"; os.chdir(ROOT); sys.path.insert(0,"crossdomain")
import xseed; xseed.load_all(); EPS=xseed.wn_episodes()
SCR="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"

T=pd.read_parquet("tmdb/artifacts/tmdb_v1/dataset.parquet")
TROW={v:i for i,v in enumerate(T["item_id"].tolist())}

G=json.load(open("crossdomain/x24_personas.json"))
profs=[]
for p in G:
    seeds={}
    if p["steam"]: seeds["steam"]=[int(x) for x in p["steam"]]
    if p["tmdb"]:  seeds["tmdb"] =[TROW[x] for x in p["tmdb"]]
    if p["wn"]:    seeds["wn"]   =[int(x) for x in p["wn"]]
    sn=p["seed_names"]
    lab=" | ".join(f"{c}: {', '.join(sn[k])}" for k,c in (("steam","S"),("tmdb","T"),("wn","W")) if sn.get(k))
    profs.append(dict(split="근거",pid=p["pid"],seeds=seeds,seedname=lab))
for e in json.load(open("crossdomain/personas_v2.json")):
    s={k:[int(x) for x in v] for k,v in e["seeds"].items() if v}
    n=e["names"]
    lab=" | ".join(f"{c}: {', '.join(n[k][:5])}" for k,c in (("steam","S"),("tmdb","T"),("wn","W")) if n.get(k))
    profs.append(dict(split="참조",pid=e["pid"],seeds=s,seedname=lab))

VAR={"근거":[("M6","M6"),("C1a","C1")],"참조":[("M6","M6")]}
lists={}
if os.path.exists("crossdomain/x24_lists.json"): lists=json.load(open("crossdomain/x24_lists.json"))
else:
    for e in profs:
        for tag,variant in VAR[e["split"]]:
            pl=xseed.platform_lists(e["seeds"],variant,"a",50)
            lists[f"{e['split']}|{tag}|{e['pid']}"]=xseed.mix_m6(pl,e["seeds"],50,EPS)
        print(e["split"],e["pid"],flush=True)
    json.dump(lists,open("crossdomain/x24_lists.json","w"))

# ── 카드 ──
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

# ── 슬롯 표본 (1-10 중 5 · 11-50 중 5, seed 24) ──
SN={e["pid"]:e["seedname"] for e in profs}
rng=random.Random(24); rows=[]
for key,slots in sorted(lists.items()):
    split,tag,pid=key.split("|")
    head=[(s,i+1) for i,s in enumerate(slots[:10])]
    tail=[(s,i+11) for i,s in enumerate(slots[10:50])]
    pick=[(s,p,"top10") for s,p in rng.sample(head,min(5,len(head)))]
    pick+=[(s,p,"tail")  for s,p in rng.sample(tail,min(5,len(tail)))]
    for s,pos,part in pick:
        rows.append(dict(split=split,variant=tag,pid=pid,plat=s["plat"],item=str(s["item"]),pos=pos,part=part))
need=[]; seen=set()
for r in rows:
    k=(r["pid"],r["plat"],r["item"])
    if k not in seen: seen.add(k); need.append(k)
rng.shuffle(need)
PC={"steam":"게임","tmdb":"영상","wn":"소설"}
todo=[]
for i,k in enumerate(need):
    c=cards[f"{k[1]}|{k[2]}"]
    todo.append(dict(id=f"y{i:04d}",pid=k[0],plat=k[1],item=k[2],seed=SN[k[0]],
                     cand=c["name"],meta=f"{PC[k[1]]} · {c['meta'][:40]} | {c['stat']}"))
json.dump(dict(rows=rows,todo=todo),open("eval/x24_key.json","w"),ensure_ascii=False)
nb=(len(todo)+59)//60
for b in range(nb):
    open(f"{SCR}/x24_b{b}.json","w").write(json.dumps(
        [{k:r[k] for k in ("id","seed","cand","meta")} for r in todo[b*60:(b+1)*60]],ensure_ascii=False))
by=collections.Counter((r["split"],r["variant"]) for r in rows)
print("\n슬롯:",dict(by)); print(f"신규 채점 {len(todo)} 쌍 · 배치 {nb}")
