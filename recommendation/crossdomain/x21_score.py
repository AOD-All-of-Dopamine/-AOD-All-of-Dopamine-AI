"""X-21 판정: 웹소설 단독 20화 필터. 사전등록 문턱 적용."""
import os, json, glob, collections, numpy as np, pandas as pd, sys
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
K=json.load(open("eval/x21_key.json")); rows=K["rows"]; todo={t["id"]:t for t in K["todo"]}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in glob.glob(f"eval/x16/{role}_x21_*.json"):
        for i,g in json.load(open(f)).items(): votes[i][role]=int(g)
new={}
for i,v in votes.items():
    if i in todo and len(v)>=3:
        t=todo[i]; new[(t["pid"],str(t["item"]))]=sum(g>=2 for g in v.values())>=2
unan=sum(1 for v in votes.values() if len(v)==3 and len({g>=2 for g in v.values()})==1)
print(f"신규 판정 {len(new)} · 만장일치 {unan/max(len(votes),1):.2f}")
miss=collections.Counter()
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],str(r["item"]))
    if k in new: r["maj"]=new[k]
    else: miss[r["arm"]]+=1
print("미채점", dict(miss) or 0)
json.dump(K,open("eval/x21_graded.json","w"),ensure_ascii=False)
def P(rs): return np.mean([r["maj"] for r in rs]) if rs else float("nan")
res={}
print(f"\n{'팔':<14}{'1-10':>9}{'11-50':>9}{'합산':>9}{'n':>6}")
for a in ("A_현행","B_20화필터"):
    rs=[r for r in rows if r["arm"]==a and "maj" in r]
    res[a]=(P([r for r in rs if r["pos"]<=10]), P([r for r in rs if r["pos"]>10]), P(rs))
    print(f"{a:<14}{res[a][0]:>9.3f}{res[a][1]:>9.3f}{res[a][2]:>9.3f}{len(rs):>6}")
d10=res["B_20화필터"][0]-res["A_현행"][0]; dall=res["B_20화필터"][2]-res["A_현행"][2]
print(f"{'Δ':<14}{d10:>+9.3f}{res['B_20화필터'][1]-res['A_현행'][1]:>+9.3f}{dall:>+9.3f}")
print("\n판정:", "(A) 채택 — 플랫폼 층에 20화 필터 추가" if dall>=0.03 and d10>=-0.02
      else "(C) 기각 — 통합 층에만 유지" if dall<=-0.02 or d10<-0.05 else "보류 — 현행 유지")
# 페르소나별
per=collections.defaultdict(dict)
for a in ("A_현행","B_20화필터"):
    for pid in {r["pid"] for r in rows}:
        per[pid][a]=P([r for r in rows if r["arm"]==a and r["pid"]==pid and "maj" in r])
d=sorted(((v["B_20화필터"]-v["A_현행"],p) for p,v in per.items()))
print(f"\n개선 {sum(1 for x in d if x[0]>0.02)} · 악화 {sum(1 for x in d if x[0]<-0.02)} · 비슷 {sum(1 for x in d if abs(x[0])<=0.02)}")
print("  최악:", [(p,round(v,2)) for v,p in d[:3]], "· 최선:", [(p,round(v,2)) for v,p in d[-3:]])
