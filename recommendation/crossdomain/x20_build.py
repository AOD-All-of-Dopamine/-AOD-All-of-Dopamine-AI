"""X-20: 제품 경로 검증 슬롯·배치 생성 (랭커 미로드, parquet 만)."""
import os, sys, json, random, collections, pandas as pd
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
S="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
BANDS=[0,2,4]
st=json.load(open(f"{S}/np_steam.json")); wn=json.load(open(f"{S}/np_wn.json")); wt=json.load(open(f"{S}/np_wn_t2.json"))
bank={}
for f in ("eval/x16_graded.json","eval/r1_val_graded.json","eval/x18_val_graded.json","eval/r1_dev_a_graded.json","eval/r1_dev_b_graded.json","eval/r2wn_graded.json"):
    for r in json.load(open(f))["rows"]:
        if "maj" in r: bank.setdefault((r["pid"],r.get("plat","tmdb"),str(r["item"])),r["maj"])
for f in ("eval/x14_graded.json","eval/x15_graded.json","eval/r0_graded.json","eval/x9_all_600.json"):
    for r in json.load(open(f)):
        if "maj" in r: bank.setdefault((r["pid"],r["plat"],str(r["item"])),bool(r["maj"]))
rows=[]
for pid,v in st.items():
    for b in BANDS:
        for arm,ids in (("제품",v["pages"][b]),("평가경로",v["run_multi"][b*10:(b+1)*10])):
            for j,i in enumerate(ids):
                r=dict(axis="steam",arm=arm,pid=pid,plat="steam",item=str(i),page=b+1,pos=b*10+j+1)
                k=(pid,"steam",str(i))
                if k in bank: r["maj"]=bank[k]
                rows.append(r)
for pid in wn:
    for b in BANDS:
        for arm,ids in (("제품(max)",wn[pid]["pages"][b]),("확정(top2_mean)",wt[pid][b])):
            for j,i in enumerate(ids):
                r=dict(axis="wn",arm=arm,pid=pid,plat="wn",item=str(i),page=b+1,pos=b*10+j+1)
                k=(pid,"wn",str(i))
                if k in bank: r["maj"]=bank[k]
                rows.append(r)
# 카드
cards=json.load(open("crossdomain/x16_cards.json"))
missing={"steam":set(),"wn":set()}
for r in rows:
    if "maj" not in r and f"{r['plat']}|{r['item']}" not in cards: missing[r["plat"]].add(int(r["item"]))
if missing["steam"]:
    d=pd.read_parquet("steam/artifacts/tags_full/dataset.parquet",columns=["steam_appid","name","tags","recommendations_total"])
    d=d[d["steam_appid"].astype(int).isin(missing["steam"])]
    for a,n,t,rc in zip(d["steam_appid"],d["name"],d["tags"],d["recommendations_total"]):
        tg=list(t)[:5] if t is not None and len(t) else []
        try: rc=int(rc) if rc==rc and rc is not None else 0
        except Exception: rc=0
        cards[f"steam|{int(a)}"]=dict(name=str(n),meta=" · ".join(map(str,tg)) or "-",stat=f"리뷰 {rc:,}" if rc else "리뷰 수 미보고")
if missing["wn"]:
    d=pd.read_parquet("/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4/dataset.parquet",
                      columns=["item_id","name","genres","author","rating","interest_count","episode_count"]).set_index("item_id")
    for i in missing["wn"]:
        if i not in d.index: continue
        r=d.loc[i]; r=r.iloc[0] if hasattr(r,"iloc") and getattr(r,"ndim",1)>1 else r
        g=r["genres"]; g=list(g) if g is not None and len(g) else []
        rt=float(r["rating"] or 0)
        cards[f"wn|{i}"]=dict(name=str(r["name"]),meta=(", ".join(map(str,g)) or "-")+f" · {r['author']}",
            stat=(f"평점 {rt:.1f} · " if rt>0 else "평점 없음 · ")+f"관심 {int(r['interest_count'] or 0):,} · {int(r['episode_count'] or 0)}화")
json.dump(cards,open("crossdomain/x16_cards.json","w"),ensure_ascii=False)
SN={}
for f in ("eval/x9_all_600.json","eval/x15_graded.json"):
    for r in json.load(open(f)): SN[r["pid"]]=r["seed"]
LBL={"steam":"게임","wn":"소설"}
need=[]; seen=set()
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["plat"],r["item"])
    if k in seen: continue
    seen.add(k); need.append(r)
random.Random(20260831010).shuffle(need)
todo=[]
for i,r in enumerate(need):
    c=cards.get(f"{r['plat']}|{r['item']}")
    if not c: continue
    todo.append(dict(id=f"n{i:04d}",pid=r["pid"],plat=r["plat"],item=r["item"],seed=SN[r["pid"]],
                     cand=c["name"],meta=f"{LBL[r['plat']]} · {c['meta'][:40]} | {c['stat']}"))
json.dump(dict(rows=rows,todo=todo),open("eval/x20_key.json","w"),ensure_ascii=False)
nb=(len(todo)+59)//60
for b in range(nb): open(f"{S}/x20_b{b}.json","w").write(json.dumps([{k:t[k] for k in ("id","seed","cand","meta")} for t in todo[b*60:(b+1)*60]],ensure_ascii=False))
c=collections.Counter((r["axis"],r["arm"]) for r in rows); h=collections.Counter((r["axis"],r["arm"]) for r in rows if "maj" in r)
for k in sorted(c): print(f"{k}: 슬롯 {c[k]} · 은행 {h[k]}")
print(f"신규 {len(todo)} · 배치 {nb}")
