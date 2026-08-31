"""R-1 dev: 변형 (a) vs (b) 비교 → 하나 선택 (사전등록: 플랫폼 슬롯 P@50 최고)."""
import os, sys, json, glob, collections, numpy as np
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
split=sys.argv[1] if len(sys.argv)>1 else "dev"
todo={t["id"]:t for t in json.load(open(f"eval/r1_{split}_todo.json"))}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in glob.glob(f"eval/x16/{role}_r1{split}_*.json"):
        for i,g in json.load(open(f)).items(): votes[i][role]=int(g)
new={}
for i,v in votes.items():
    if i in todo and len(v)>=3:
        t=todo[i]; new[(t["pid"],t["plat"],str(t["item"]))]=sum(g>=2 for g in v.values())>=2
unan=sum(1 for v in votes.values() if len(v)==3 and len({g>=2 for g in v.values()})==1)
print(f"신규 판정 {len(new)} · 만장일치 {unan/max(len(votes),1):.2f}")
def P(rs): return np.mean([r["maj"] for r in rs]) if rs else float("nan")
out={}
for v in ("a","b"):
    K=json.load(open(f"eval/r1_{split}_{v}_key.json")); rows=K["rows"]; miss=0
    for r in rows:
        if "maj" in r: continue
        k=(r["pid"],r["plat"],str(r["item"]))
        if k in new: r["maj"]=new[k]
        else: miss+=1
    json.dump(K,open(f"eval/r1_{split}_{v}_graded.json","w"),ensure_ascii=False)
    g=[r for r in rows if "maj" in r]
    out[v]=dict(p10=P([r for r in g if r["pos"]<=10]), ptail=P([r for r in g if r["pos"]>10]), n=len(g), miss=miss)
    per={pl: P([r for r in g if r["plat"]==pl]) for pl in ("steam","tmdb")}
    print(f"  ({v}) 1-10 {out[v]['p10']:.3f} · 11-50 {out[v]['ptail']:.3f} · n={len(g)} 미채점 {miss} · steam {per['steam']:.3f} tmdb {per['tmdb']:.3f}")
win=max(("a","b"), key=lambda v: (out[v]["p10"]*10+out[v]["ptail"]*40)/50)
print(f"\n[사전등록: dev 에서 이긴 변형 하나만 val 로]  선택 = ({win})")
