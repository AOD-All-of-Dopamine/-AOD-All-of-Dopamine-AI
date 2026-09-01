"""X-20 판정: 제품 경로(next_page) vs 비교 팔. 사전등록 문턱 적용."""
import os, json, glob, collections, numpy as np
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
K=json.load(open("eval/x20_key.json")); rows=K["rows"]; todo={t["id"]:t for t in K["todo"]}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in glob.glob(f"eval/x16/{role}_x20_*.json"):
        for i,g in json.load(open(f)).items(): votes[i][role]=int(g)
new={}
for i,v in votes.items():
    if i in todo and len(v)>=3:
        t=todo[i]; new[(t["pid"],t["plat"],str(t["item"]))]=sum(g>=2 for g in v.values())>=2
unan=sum(1 for v in votes.values() if len(v)==3 and len({g>=2 for g in v.values()})==1)
print(f"신규 판정 {len(new)} · 만장일치 {unan/max(len(votes),1):.2f}")
miss=collections.Counter()
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["plat"],str(r["item"]))
    if k in new: r["maj"]=new[k]
    else: miss[(r["axis"],r["arm"])]+=1
print("미채점", dict(miss) or 0)
json.dump(K,open("eval/x20_graded.json","w"),ensure_ascii=False)
def P(rs): return np.mean([r["maj"] for r in rs]) if rs else float("nan")
for axis,arms,label in (("wn",("제품(max)","확정(top2_mean)"),"X-20a 웹소설"),
                        ("steam",("제품","평가경로"),"X-20b Steam")):
    print(f"\n### {label}")
    print(f"{'팔':<18}{'p1':>8}{'p3':>8}{'p5':>8}{'합산':>8}{'n':>6}")
    agg={}
    for a in arms:
        rs=[r for r in rows if r["axis"]==axis and r["arm"]==a and "maj" in r]
        v=[P([r for r in rs if r["page"]==p]) for p in (1,3,5)]
        agg[a]=P(rs)
        print(f"{a:<18}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}{P(rs):>8.3f}{len(rs):>6}")
    d=agg[arms[1]]-agg[arms[0]]
    if axis=="wn":
        print(f"Δ(확정 − 제품) {d:+.3f}")
        print("판정:", "(A) 확정값으로 통일" if d>=0.02 else "(C) max 가 옳음 — PRODUCTION 재검토" if d<=-0.05 else "차이 없음 → 확정값으로 통일")
    else:
        d2=agg["제품"]-agg["평가경로"]
        print(f"Δ(제품 − 평가경로) {d2:+.3f}")
        print("판정:", "(A) 제품 경로 검증 완료" if d2>=-0.02 else "(C) 제품 경로 결함" if d2<-0.05 else "보류")
