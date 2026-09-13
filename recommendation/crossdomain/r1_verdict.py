"""R-1 val 판정: 리랭커(b) vs 현행 M6. Steam·TMDB 슬롯만, 사전등록 문턱."""
import os, json, glob, collections, numpy as np
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
todo={t["id"]:t for t in json.load(open("eval/r1_val_todo.json"))}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in glob.glob(f"eval/x16/{role}_r1val_*.json"):
        for i,g in json.load(open(f)).items(): votes[i][role]=int(g)
new={}
for i,v in votes.items():
    if i in todo and len(v)>=3:
        t=todo[i]; new[(t["pid"],t["plat"],str(t["item"]))]=sum(g>=2 for g in v.values())>=2
unan=sum(1 for v in votes.values() if len(v)==3 and len({g>=2 for g in v.values()})==1)
print(f"신규 판정 {len(new)} · 만장일치 {unan/max(len(votes),1):.2f}")
K=json.load(open("eval/r1_val_b_key.json")); rows=K["rows"]; miss=0
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["plat"],str(r["item"]))
    if k in new: r["maj"]=new[k]
    else: miss+=1
print("미채점", miss)
json.dump(K,open("eval/r1_val_graded.json","w"),ensure_ascii=False)
base=[r for r in json.load(open("eval/x16_graded.json"))["rows"]
      if r["split"]=="val" and r["variant"]=="M6" and r["plat"] in ("steam","tmdb") and "maj" in r]
new_rows=[r for r in rows if "maj" in r]
def P(rs): return np.mean([r["maj"] for r in rs]) if rs else float("nan")
print(f"\n{'':14}{'1-10':>9}{'11-50':>9}{'n':>7}   steam / tmdb (전체)")
for nm,rs in (("현행 M6",base),("리랭커 (b)",new_rows)):
    pp="/".join(f"{P([r for r in rs if r['plat']==p]):.3f}" for p in ("steam","tmdb"))
    print(f"{nm:14}{P([r for r in rs if r['pos']<=10]):>9.3f}{P([r for r in rs if r['pos']>10]):>9.3f}{len(rs):>7}   {pp}")
d10=P([r for r in new_rows if r["pos"]<=10])-P([r for r in base if r["pos"]<=10])
d50=P([r for r in new_rows if r["pos"]>10])-P([r for r in base if r["pos"]>10])
print(f"{'Δ':14}{d10:>+9.3f}{d50:>+9.3f}")
print("\n플랫폼별 판정 (사전등록: 플랫폼별 독립)")
for p in ("steam","tmdb"):
    b=[r for r in base if r["plat"]==p]; n=[r for r in new_rows if r["plat"]==p]
    a10=P([r for r in n if r["pos"]<=10])-P([r for r in b if r["pos"]<=10])
    a50=P([r for r in n if r["pos"]>10])-P([r for r in b if r["pos"]>10])
    v=("(A) 채택" if a50>=0.05 and a10>=-0.02 else "(C) 기각" if a10<-0.05 else "보류")
    print(f"  {p:<6} Δ1-10 {a10:+.3f} · Δ11-50 {a50:+.3f} → {v}")
