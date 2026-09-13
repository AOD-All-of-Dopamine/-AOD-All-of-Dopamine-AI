"""R-1 신규 슬롯 카드 생성 + 눈가림 배치 (랭커 미로드, parquet 만 읽는다)."""
import os, sys, json, random, pandas as pd
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
split=sys.argv[1]; variants=sys.argv[2].split(",")
SCR="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
cards=json.load(open("crossdomain/x16_cards.json"))
need={"steam":set(),"tmdb":set()}
keys={}
for v in variants:
    K=json.load(open(f"eval/r1_{split}_{v}_key.json"))
    keys[v]=K
    for r in K["need"]:
        if f"{r['plat']}|{r['item']}" not in cards: need[r["plat"]].add(int(r["item"]))
if need["steam"]:
    d=pd.read_parquet("steam/artifacts/tags_full/dataset.parquet",
                      columns=["steam_appid","name","tags","recommendations_total"])
    d=d[d["steam_appid"].astype(int).isin(need["steam"])]
    for a,n,t,rc in zip(d["steam_appid"],d["name"],d["tags"],d["recommendations_total"]):
        tg=list(t)[:5] if t is not None and len(t) else []
        try: rc=int(rc) if rc==rc and rc is not None else 0
        except Exception: rc=0
        cards[f"steam|{int(a)}"]=dict(name=str(n), meta=" · ".join(map(str,tg)) or "-",
                                      stat=f"리뷰 {rc:,}" if rc else "리뷰 수 미보고")
if need["tmdb"]:
    idx=pd.read_parquet("tmdb/artifacts/tmdb_v1/corpus_index.parquet", columns=["item_id","embedding_row"]).sort_values("embedding_row")
    ds=pd.read_parquet("tmdb/artifacts/tmdb_v1/dataset.parquet",
                       columns=["item_id","name","media","genres","vote_average","vote_count"]).set_index("item_id")
    ds=ds.loc[idx["item_id"].to_numpy()].reset_index(drop=True)
    for row in need["tmdb"]:
        if row>=len(ds): continue
        r=ds.iloc[row]; g=r["genres"]; g=list(g) if g is not None and len(g) else []
        cards[f"tmdb|{row}"]=dict(name=str(r["name"]),
            meta=("영화" if r["media"]=="movie" else "TV")+" · "+(", ".join(map(str,g)) or "-"),
            stat=f"평점 {float(r['vote_average'] or 0):.1f} · 투표 {int(r['vote_count'] or 0):,}")
json.dump(cards,open("crossdomain/x16_cards.json","w"),ensure_ascii=False)
SN={}
for f in ("eval/x9_all_600.json","eval/x15_graded.json"):
    for r in json.load(open(f)): SN[r["pid"]]=r["seed"]
LBL={"steam":"게임","tmdb":"영상","wn":"소설"}
allneed=[]; seen=set()
for v in variants:
    for r in keys[v]["need"]:
        k=(r["pid"],r["plat"],r["item"])
        if k in seen: continue
        seen.add(k); allneed.append(r)
random.Random(20260831004).shuffle(allneed)
todo=[]
for i,r in enumerate(allneed):
    c=cards.get(f"{r['plat']}|{r['item']}")
    if not c: continue
    todo.append(dict(id=f"r{i:04d}",pid=r["pid"],plat=r["plat"],item=r["item"],seed=SN[r["pid"]],
                     cand=c["name"], meta=f"{LBL[r['plat']]} · {c['meta'][:40]} | {c['stat']}"))
json.dump(todo,open(f"eval/r1_{split}_todo.json","w"),ensure_ascii=False)
nb=(len(todo)+59)//60
for b in range(nb): open(f"{SCR}/r1{split}_b{b}.json","w").write(json.dumps([{k:t[k] for k in ("id","seed","cand","meta")} for t in todo[b*60:(b+1)*60]],ensure_ascii=False))
print(f"신규 합집합 {len(todo)} · 배치 {nb}")
