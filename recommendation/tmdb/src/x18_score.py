"""X-18 dev 재계산: 미채점을 독립 3역할로 채운 뒤 kw_w 를 고른다.
은행(설계자·관대)과 독립(3역할·엄격) 두 잣대가 섞이므로 혼합 비율을 함께 보고한다.
"""
import os, sys, json, glob, collections, numpy as np, pandas as pd
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
ns={}; exec(open("tmdb/artifacts/p1/grades.py",encoding="utf-8").read(), ns); G=ns["G"]
todo=json.load(open("tmdb/artifacts/p1/x18_todo.json")); byid={t["id"]:t for t in todo}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in glob.glob(f"eval/x16/{role}_x18_*.json"):
        for i,g in json.load(open(f)).items(): votes[i][role]=int(g)
IND={}
for i,v in votes.items():
    if i in byid and len(v)>=3:
        IND[(byid[i]["pid"], int(byid[i]["item"]))] = sum(g>=2 for g in v.values())>=2
unan=sum(1 for v in votes.values() if len(v)==3 and len({g>=2 for g in v.values()})==1)
print(f"독립 판정 {len(IND)} · 만장일치 {unan/max(len(votes),1):.2f} · 독립 적합률 {np.mean(list(IND.values())):.3f}")
lists=json.load(open("tmdb/artifacts/p1/x18_lists.json"))
print(f"\n{'kw_w':<7}{'은행만':>9}{'채운 뒤':>10}{'독립 비율':>10}{'미채점':>8}")
res={}
for kw in ("0.0","0.2","0.4","0.6"):
    bank_only=[]; filled=[]; nind=0; ntot=0; miss=0
    for k,rows in lists.items():
        if not k.startswith(kw+"|"): continue
        pid=k.split("|",1)[1]
        b=[]; f=[]
        for x in rows:
            g=G.get(f"{pid}\t{x}")
            if g is not None: b.append(g>=2); f.append(g>=2)
            elif (pid,x) in IND: f.append(IND[(pid,x)]); nind+=1
            else: miss+=1
            ntot+=1
        if b: bank_only.append(np.mean(b))
        if f: filled.append(np.mean(f))
    res[kw]=(np.mean(bank_only), np.mean(filled), nind/ntot, miss)
    print(f"{kw:<7}{np.mean(bank_only):>9.4f}{np.mean(filled):>10.4f}{nind/ntot:>9.1%}{miss:>8}")
base=res["0.0"][1]
cand={k:v[1] for k,v in res.items() if k!="0.0"}
best=max(cand.items(), key=lambda x:(x[1], -float(x[0])))
print(f"\n[사전등록 규칙: 미채점 0 을 채운 뒤 최고값, 동률이면 작은 값]")
print(f"  기준선(0) {base:.4f} · 최적 kw_w={best[0]} {best[1]:.4f} (Δ {best[1]-base:+.4f})")
