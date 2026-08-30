"""X-16: dev(무작위 60) · val(페르소나 v2) 에 대해 M6 · C1a · C1b · C2 통합 목록을 만들고,
눈가림 채점 슬롯(1-10 전수 + 11-50 프로필당 10)을 뽑는다. 기존 채점은 은행에서 재사용.
    python crossdomain/x16_build.py           → crossdomain/x16_lists.json, x16_cards.json, eval/x16_key.json, 배치
"""
import json, random, sys, os, glob, collections
import pandas as pd
ROOT="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation"; os.chdir(ROOT); sys.path.insert(0,"crossdomain")
import xseed; xseed.load_all(); EPS=xseed.wn_episodes()
SCR="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
VARIANTS=[("M6","M6","a"),("C1a","C1","a"),("C1b","C1","b"),("C2","C2","a")]

# ── 프로필 ──────────────────────────────────────────────────────────────────
SI=json.load(open("crossdomain/seed_index.json"))["seeds"]
DEV_PIDS=set(json.load(open("crossdomain/mixed_x11.json"))["M6"])   # X-7~X-13 의 dev 60 (r2_st_* + r3_000~041)
dev=[]
for e in json.load(open("crossdomain/profiles_random100.json")):
    if e["pid"] not in DEV_PIDS: continue
    seeds={p:SI[p][e[p]] for p in ("steam","tmdb","wn") if e.get(p)}
    dev.append(dict(pid=e["pid"],seeds=seeds))
val=[dict(pid=e["pid"],seeds=e["seeds"]) for e in json.load(open("crossdomain/personas_v2.json"))]
SN={}
for r in json.load(open("eval/x9_all_600.json")): SN[r["pid"]]=r["seed"]
for r in json.load(open("eval/x15_graded.json")): SN[r["pid"]]=r["seed"]

# ── 목록 ────────────────────────────────────────────────────────────────────
lists={}
if os.path.exists("crossdomain/x16_lists.json"): lists=json.load(open("crossdomain/x16_lists.json"))
for split,profs in (("dev",dev),("val",val)) if not lists else ():
    for e in profs:
        for tag,variant,comb in VARIANTS:
            pl=xseed.platform_lists(e["seeds"],variant,comb,50)
            lists[f"{split}|{tag}|{e['pid']}"]=xseed.mix_m6(pl,e["seeds"],50,EPS)
        print(split,e["pid"],flush=True)
json.dump(lists,open("crossdomain/x16_lists.json","w"))

# M6 dev 재현 확인 (mixed_x11 M6 와 비교)
ref=json.load(open("crossdomain/mixed_x11.json"))["M6"]; ov=[]
for e in dev:
    a={(s["plat"],s["item"]) for s in lists[f"dev|M6|{e['pid']}"]}; b={(s["plat"],s["item"]) for s in ref[e["pid"]]}
    ov.append(len(a&b)/max(len(b),1))
print(f"dev M6 재현 일치 평균 {sum(ov)/len(ov):.3f} 최소 {min(ov):.2f}")

# ── 카드 ────────────────────────────────────────────────────────────────────
def num(v,d=0):
    try:
        if v is None or pd.isna(v): return d
    except Exception: pass
    return v
S=xseed._MODS["steam"]["comps"][3].dataset.reset_index().set_index("steam_appid") if "steam_appid" not in xseed._MODS["steam"]["comps"][3].dataset.index.names else xseed._MODS["steam"]["comps"][3].dataset
S=pd.read_parquet("steam/artifacts/tags_full/dataset.parquet").set_index("steam_appid")
T=xseed._MODS["tmdb"]["comps"][1].dataset
W=xseed._MODS["wn"]["eng"].ds.set_index("item_id")
def card(plat,item):
    if plat=="steam":
        r=S.loc[int(item)]; r=r.iloc[0] if isinstance(r,pd.DataFrame) else r
        tags=r.get("tags"); tags=list(tags)[:5] if tags is not None and len(tags) else []
        n=num(r.get("recommendations_total")); return dict(name=str(r["name"]),meta=" · ".join(map(str,tags)) or "-",stat=f"리뷰 {int(n):,}" if n else "리뷰 수 미보고")
    if plat=="tmdb":
        r=T.iloc[int(item)]; g=r["genres"]; g=list(g) if g is not None and len(g) else []
        return dict(name=str(r["name"]),meta=("영화" if r["media"]=="movie" else "TV")+" · "+(", ".join(map(str,g)) or "-"),stat=f"평점 {float(num(r['vote_average'],0.0)):.1f} · 투표 {int(num(r['vote_count'])):,}")
    r=W.loc[int(item)]; r=r.iloc[0] if isinstance(r,pd.DataFrame) else r
    g=r["genres"]; g=list(g) if g is not None and len(g) else []; rt=float(num(r["rating"],0.0))
    return dict(name=str(r["name"]),meta=(", ".join(map(str,g)) or "-")+f" · {r['author']}",stat=(f"평점 {rt:.1f} · " if rt>0 else "평점 없음 · ")+f"관심 {int(num(r['interest_count'])):,} · {int(num(r['episode_count']))}화")
cards={}
for slots in lists.values():
    for s in slots:
        k=f"{s['plat']}|{s['item']}"
        if k not in cards: cards[k]=card(s["plat"],s["item"])
json.dump(cards,open("crossdomain/x16_cards.json","w"),ensure_ascii=False)

# ── 은행 ────────────────────────────────────────────────────────────────────
bank={}
for f in ("eval/x9_all_600.json","eval/x14_graded.json","eval/x15_graded.json","eval/r0_graded.json"):
    for r in json.load(open(f)):
        if "maj" in r: bank[(r["pid"],r["plat"],str(r["item"]))]=bool(r["maj"])
for x in ("x10","x11","x12"):
    kd=json.load(open(f"eval/{x}_key.json")); kd=kd["rows"] if isinstance(kd,dict) else kd
    key={r["id"]:r for r in kd if "id" in r}; votes=collections.defaultdict(list)
    for role in "ABC":
        for line in open(f"eval/{x}/{role}.jsonl"):
            line=line.strip()
            if line:
                for i,g in json.loads(line).items(): votes[i].append(int(g)>=2)
    for i,v in votes.items():
        if len(v)>=3 and i in key: r=key[i]; bank[(r["pid"],r["plat"],str(r["item"]))]=sum(v)>=2
print(f"은행 {len(bank)} 쌍")

# ── 슬롯 표본 ───────────────────────────────────────────────────────────────
rng=random.Random(20260830016); rows=[]; seen=set()
for key,slots in lists.items():
    split,tag,pid=key.split("|")
    pick=[(s,i+1,"top10") for i,s in enumerate(slots[:10])]
    tail=[(s,i+11,"tail") for i,s in enumerate(slots[10:50])]
    pick+= [(s,pos,"tail") for s,pos in rng.sample([(s,p) for s,p,_ in tail],min(10,len(tail)))]
    for s,pos,part in pick:
        rows.append(dict(split=split,variant=tag,pid=pid,plat=s["plat"],item=s["item"],rank=s["rank"],pos=pos,part=part))
need=[]; 
for r in rows:
    k=(r["pid"],r["plat"],r["item"])
    if k in bank: r["maj"]=bank[k]
    elif k not in seen: seen.add(k); need.append(k)
rng.shuffle(need); todo=[]
for i,k in enumerate(need):
    c=cards[f"{k[1]}|{k[2]}"]
    todo.append(dict(id=f"x{i:04d}",pid=k[0],plat=k[1],item=k[2],seed=SN[k[0]],cand=c["name"],meta=f"{ {'steam':'게임','tmdb':'영상','wn':'소설'}[k[1]] } · {c['meta'][:40]} | {c['stat']}"))
json.dump(dict(rows=rows,todo=todo),open("eval/x16_key.json","w"),ensure_ascii=False)
nb=(len(todo)+59)//60
for b in range(nb): open(f"{SCR}/x16_b{b}.json","w").write(json.dumps([{k:r[k] for k in ("id","seed","cand","meta")} for r in todo[b*60:(b+1)*60]],ensure_ascii=False))
by=collections.Counter((r["split"],r["variant"]) for r in rows); have=collections.Counter((r["split"],r["variant"]) for r in rows if "maj" in r)
for k in sorted(by): print(k, f"슬롯 {by[k]} · 은행 {have[k]}")
print(f"신규 채점 {len(todo)} 쌍 · 배치 {nb}")
