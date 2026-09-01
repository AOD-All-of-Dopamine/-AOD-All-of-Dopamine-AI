"""X-21: 웹소설 단독 top-50 을 필터 없음/있음 두 팔로 만들고 채점 슬롯을 뽑는다."""
import os, sys, json, random, collections, numpy as np
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation"); sys.path.insert(0,"crossdomain")
S="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
import xseed; xseed._load_platform("wn", xseed.WN, xseed.ART["wn"])
eng=xseed._MODS["wn"]["eng"]; ds=eng.ds.set_index("item_id"); ep=ds["episode_count"].fillna(0)
PERS=[e for e in json.load(open("crossdomain/personas_v2.json")) if e["seeds"].get("wn")]
arms={}
for e in PERS:
    seeds=[int(x) for x in e["seeds"]["wn"]]
    deep=xseed.rank_wn({"wn":seeds},"M6","a",250)
    arms[("A_현행",e["pid"])]=deep[:50]
    arms[("B_20화필터",e["pid"])]=[i for i in deep if ep.get(i,0)>=20][:50]
    print(e["pid"],flush=True)
json.dump({f"{a}|{p}":v for (a,p),v in arms.items()}, open("crossdomain/x21_lists.json","w"))
for a in ("A_현행","B_20화필터"):
    L=[arms[(a,e["pid"])] for e in PERS]
    print(f"{a}: 길이 중앙 {int(np.median([len(x) for x in L]))} 최소 {min(len(x) for x in L)} · "
          f"20화미만 {np.mean([np.mean([ep.get(i,0)<20 for i in x]) for x in L]):.1%} · "
          f"화수중앙 {np.median([ep.loc[[i for i in x if i in ds.index]].median() for x in L]):.0f} · "
          f"관심중앙 {np.median([ds['interest_count'].fillna(0).loc[[i for i in x if i in ds.index]].median() for x in L]):,.0f}")
bank={}
for f in ("eval/x16_graded.json","eval/r1_val_graded.json","eval/x18_val_graded.json","eval/r1_dev_a_graded.json","eval/r1_dev_b_graded.json","eval/r2wn_graded.json","eval/x20_graded.json"):
    for r in json.load(open(f))["rows"]:
        if "maj" in r: bank.setdefault((r["pid"],r.get("plat","tmdb"),str(r["item"])),r["maj"])
for f in ("eval/x14_graded.json","eval/x15_graded.json","eval/r0_graded.json","eval/x9_all_600.json"):
    for r in json.load(open(f)):
        if "maj" in r: bank.setdefault((r["pid"],r["plat"],str(r["item"])),bool(r["maj"]))
rng=random.Random(20260831020); rows=[]
for (a,pid),ids in arms.items():
    sel=[(i,j+1) for j,i in enumerate(ids)]
    pick=[x for x in sel if x[1]<=10]
    tail=[x for x in sel if x[1]>10]
    pick+=rng.sample(tail,min(10,len(tail)))
    for i,pos in pick:
        r=dict(arm=a,pid=pid,plat="wn",item=str(i),pos=pos)
        k=(pid,"wn",str(i))
        if k in bank: r["maj"]=bank[k]
        rows.append(r)
cards=json.load(open("crossdomain/x16_cards.json"))
for r in rows:
    k=f"wn|{r['item']}"
    if "maj" in r or k in cards: continue
    d=ds.loc[int(r["item"])]; g=d["genres"]; g=list(g) if g is not None and len(g) else []
    rt=float(d["rating"] or 0)
    cards[k]=dict(name=str(d["name"]),meta=(", ".join(map(str,g)) or "-")+f" · {d['author']}",
        stat=(f"평점 {rt:.1f} · " if rt>0 else "평점 없음 · ")+f"관심 {int(d['interest_count'] or 0):,} · {int(d['episode_count'] or 0)}화")
json.dump(cards,open("crossdomain/x16_cards.json","w"),ensure_ascii=False)
SN={r["pid"]:r["seed"] for r in json.load(open("eval/x15_graded.json"))}
need=[]; seen=set()
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["item"])
    if k in seen: continue
    seen.add(k); need.append(r)
rng.shuffle(need); todo=[]
for i,r in enumerate(need):
    c=cards[f"wn|{r['item']}"]
    todo.append(dict(id=f"e{i:04d}",pid=r["pid"],plat="wn",item=r["item"],seed=SN[r["pid"]],cand=c["name"],meta=f"소설 · {c['meta'][:40]} | {c['stat']}"))
json.dump(dict(rows=rows,todo=todo),open("eval/x21_key.json","w"),ensure_ascii=False)
nb=(len(todo)+59)//60
for b in range(nb): open(f"{S}/x21_b{b}.json","w").write(json.dumps([{k:t[k] for k in ("id","seed","cand","meta")} for t in todo[b*60:(b+1)*60]],ensure_ascii=False))
c=collections.Counter(r["arm"] for r in rows); h=collections.Counter(r["arm"] for r in rows if "maj" in r)
for k in sorted(c): print(f"{k}: 슬롯 {c[k]} · 은행 {h[k]}")
print(f"신규 {len(todo)} · 배치 {nb}")
