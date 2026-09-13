"""R-2wn 판정: v4/v5 웹소설 슬롯 P(1-10 전수 · 11-50 표본), 사전등록 문턱 적용."""
import json, glob, collections, os
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
K=json.load(open("eval/r2wn_key.json")); rows=K["rows"]; todo={t["id"]:t for t in K["todo"]}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in glob.glob(f"eval/x16/{role}_r2wn*.json"):
        d=json.load(open(f))
        for i,g in d.items(): votes[i][role]=int(g)
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
json.dump(K,open("eval/r2wn_graded.json","w"),ensure_ascii=False)
def P(rs): return sum(r["maj"] for r in rs)/len(rs) if rs else float("nan")
res={}
for ver in ("v4","v5"):
    rs=[r for r in rows if r["ver"]==ver and "maj" in r]
    top=[r for r in rs if r["pos"]<=10]; tail=[r for r in rs if r["pos"]>10]
    res[ver]=dict(p10=P(top),ptail=P(tail),n10=len(top),nt=len(tail))
    print(f"{ver}: 웹소설 1-10 P {P(top):.3f} (n={len(top)}) · 11-50 P {P(tail):.3f} (n={len(tail)})")
d_tail=res["v5"]["ptail"]-res["v4"]["ptail"]; d_top=res["v5"]["p10"]-res["v4"]["p10"]
print(f"Δ 11-50 {d_tail:+.3f} · Δ 1-10 {d_top:+.3f}")
if d_tail>=0.05 and d_top>=-0.02: print("판정: (A) 채택 — wn_v5 확정")
elif d_top<-0.05: print("판정: (C) 기각 — 1-10 손상")
else: print("판정: 보류 — LLM 태그(변형 b)로")
# 페르소나별
per=collections.defaultdict(dict)
for ver in ("v4","v5"):
    for pid in {r["pid"] for r in rows}:
        rs=[r for r in rows if r["ver"]==ver and r["pid"]==pid and "maj" in r]
        per[pid][ver]=P(rs)
dl=sorted(per.items(), key=lambda x: x[1]["v5"]-x[1]["v4"])
print("\n페르소나별 웹소설 슬롯 P (v4→v5), 하위·상위 5")
for pid,v in dl[:5]+dl[-5:]: print(f"  {pid:<18} {v['v4']:.2f} → {v['v5']:.2f}  ({v['v5']-v['v4']:+.2f})")
