"""X-18 val 판정: kw_w=0.2 vs 현행(kw_w=0). TMDB 슬롯만, 사전등록 문턱 적용."""
import os, json, glob, collections, numpy as np
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
K=json.load(open("eval/x18_val_key.json")); rows=K["rows"]; todo={t["id"]:t for t in K["todo"]}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in glob.glob(f"eval/x16/{role}_x18v_*.json"):
        for i,g in json.load(open(f)).items(): votes[i][role]=int(g)
new={}
for i,v in votes.items():
    if i in todo and len(v)>=3: new[(todo[i]["pid"],str(todo[i]["item"]))]=sum(g>=2 for g in v.values())>=2
unan=sum(1 for v in votes.values() if len(v)==3 and len({g>=2 for g in v.values()})==1)
print(f"신규 판정 {len(new)} · 만장일치 {unan/max(len(votes),1):.2f}")
miss=0
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],str(r["item"]))
    if k in new: r["maj"]=new[k]
    else: miss+=1
print("미채점", miss)
json.dump(K,open("eval/x18_val_graded.json","w"),ensure_ascii=False)
def P(rs): return sum(r["maj"] for r in rs)/len(rs) if rs else float("nan")
# 현행 기준선: X-16 val M6 의 TMDB 슬롯
base=[r for r in json.load(open("eval/x16_graded.json"))["rows"]
      if r["split"]=="val" and r["variant"]=="M6" and r["plat"]=="tmdb" and "maj" in r]
new_rows=[r for r in rows if "maj" in r]
b10,b50=[r for r in base if r["pos"]<=10],[r for r in base if r["pos"]>10]
n10,n50=[r for r in new_rows if r["pos"]<=10],[r for r in new_rows if r["pos"]>10]
print(f"\n{'':12}{'1-10':>10}{'11-50':>10}{'n':>7}")
print(f"{'현행 kw_w=0':12}{P(b10):>10.3f}{P(b50):>10.3f}{len(base):>7}")
print(f"{'kw_w=0.2':12}{P(n10):>10.3f}{P(n50):>10.3f}{len(new_rows):>7}")
d10,d50=P(n10)-P(b10), P(n50)-P(b50)
print(f"{'Δ':12}{d10:>+10.3f}{d50:>+10.3f}")
if d50>=0.05 and d10>=-0.02: print("\n판정: (A) 채택 — kw_w=0.2 확정")
elif d10<-0.05 or d50<=-0.02: print("\n판정: (C) 기각 — 현행 유지")
else: print("\n판정: 보류 — 현행 유지")
