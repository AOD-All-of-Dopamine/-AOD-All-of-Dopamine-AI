import os, sys, json, random, collections, pandas as pd
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
S="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
lists=json.load(open("crossdomain/x22_lists.json"))
ds=pd.read_parquet("steam/artifacts/tags_full/dataset.parquet",columns=["steam_appid","name","tags","recommendations_total"]).set_index("steam_appid")
prof=pd.read_parquet("steam/artifacts/p1/profiles.parquet").set_index("profile_id")
ns={}; exec(open("steam/artifacts/p1/grades.py",encoding="utf-8").read(), ns) if os.path.exists("steam/artifacts/p1/grades.py") else None
BANK={}
if ns.get("G"):
    for k,v in ns["G"].items():
        pid,a=k.split("\t"); BANK[(pid,int(a))]=v>=2
print(f"steam p1 은행 {len(BANK)} 쌍 (설계자 채점 — 참고용, 판정엔 독립 채점만 쓴다)")
rng=random.Random(20260903001); rows=[]
for key,ids in lists.items():
    qw,grp,pid=key.split("|")
    sel=[(i,j+1) for j,i in enumerate(ids)]
    pick=[x for x in sel if x[1]<=10]+rng.sample([x for x in sel if x[1]>10],10)
    for i,pos in pick: rows.append(dict(qw=qw,grp=grp,pid=pid,item=str(i),pos=pos))
def card(a):
    r=ds.loc[int(a)]; r=r.iloc[0] if hasattr(r,"ndim") and getattr(r,"ndim",1)>1 else r
    t=r["tags"]; tg=list(t)[:5] if t is not None and len(t) else []
    rc=r["recommendations_total"]
    try: rc=int(rc) if rc==rc and rc is not None else 0
    except Exception: rc=0
    return str(r["name"]), " · ".join(map(str,tg)) or "-", (f"리뷰 {rc:,}" if rc else "리뷰 수 미보고")
SEEDNAME={}
for pid in {r["pid"] for r in rows}:
    sd=[int(x) for x in prof.loc[pid,"liked_appids"]]
    SEEDNAME[pid]="S: "+" / ".join(card(a)[0][:22] for a in sd)
need=[]; seen=set()
for r in rows:
    k=(r["pid"],r["item"])
    if k in seen: continue
    seen.add(k); need.append(r)
rng.shuffle(need); todo=[]
for i,r in enumerate(need):
    n,m,st=card(r["item"])
    todo.append(dict(id=f"q{i:04d}",pid=r["pid"],item=r["item"],seed=SEEDNAME[r["pid"]],cand=n,meta=f"게임 · {m[:44]} | {st}"))
json.dump(dict(rows=rows,todo=todo),open("eval/x22_key.json","w"),ensure_ascii=False)
nb=(len(todo)+59)//60
for b in range(nb): open(f"{S}/x22_b{b}.json","w").write(json.dumps([{k:t[k] for k in ("id","seed","cand","meta")} for t in todo[b*60:(b+1)*60]],ensure_ascii=False))
c=collections.Counter((r["qw"],r["grp"]) for r in rows)
for k in sorted(c): print(f"{k}: 슬롯 {c[k]}")
print(f"고유 신규 {len(todo)} · 배치 {nb}")
