"""X-22 판정: quality_w ∈ {0, 0.25, 0.50} 저리뷰 9 (판정) · 대작 5 (관찰)."""
import os, json, glob, collections, numpy as np
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
K=json.load(open("eval/x22_key.json")); rows=K["rows"]; todo={t["id"]:t for t in K["todo"]}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in glob.glob(f"eval/x16/{role}_x22_*.json"):
        for i,g in json.load(open(f)).items(): votes[i][role]=int(g)
maj={}
for i,v in votes.items():
    if i in todo and len(v)>=3:
        t=todo[i]; maj[(t["pid"],str(t["item"]))]=sum(g>=2 for g in v.values())>=2
unan=sum(1 for v in votes.values() if len(v)==3 and len({g>=2 for g in v.values()})==1)
print(f"신규 판정 {len(maj)} · 만장일치 {unan/max(len(votes),1):.2f}")
miss=collections.Counter()
for r in rows:
    k=(r["pid"],str(r["item"]))
    if k in maj: r["maj"]=maj[k]
    else: miss[(r["qw"],r["grp"])]+=1
print("미채점", dict(miss) or 0)
json.dump(K,open("eval/x22_graded.json","w"),ensure_ascii=False)
def P(rs): return np.mean([r["maj"] for r in rs]) if rs else float("nan")
out={}
for grp,label in (("low","저리뷰 9 (판정)"),("high","대작 5 (관찰)")):
    print(f"\n### {label}")
    print(f"{'quality_w':<11}{'1-10':>9}{'11-50':>9}{'합산':>9}{'n':>6}")
    for qw in ("0.0","0.25","0.5"):
        rs=[r for r in rows if r["qw"]==qw and r["grp"]==grp and "maj" in r]
        out[(grp,qw)]=(P([r for r in rs if r["pos"]<=10]), P([r for r in rs if r["pos"]>10]), P(rs))
        print(f"{qw:<11}{out[(grp,qw)][0]:>9.3f}{out[(grp,qw)][1]:>9.3f}{out[(grp,qw)][2]:>9.3f}{len(rs):>6}")
base=out[("low","0.5")][2]
alts={qw: out[("low",qw)][2]-base for qw in ("0.0","0.25")}
print(f"\n저리뷰 Δ(대안 − 현행 0.50): " + " · ".join(f"{k}: {v:+.3f}" for k,v in alts.items()))
best=max(alts.items(), key=lambda x:x[1])
if best[1]>=0.05: print(f"판정: (A) 저리뷰용 값 분리 근거 — quality_w={best[0]} 가 +{best[1]:.3f}. 서빙 분기는 별도 사전등록")
elif max(alts.values())<0.02: print("판정: (C) 현행 확정 — quality_w=0.50 이 저리뷰 취향에서도 옳다. 축 종료")
else: print(f"판정: 보류 — 최대 {best[0]} 가 +{best[1]:.3f} (문턱 +0.05 미달)")
hb=out[("high","0.5")][2]
print(f"대작 관찰 Δ: " + " · ".join(f"{qw}: {out[('high',qw)][2]-hb:+.3f}" for qw in ("0.0","0.25")))
