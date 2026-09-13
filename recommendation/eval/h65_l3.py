"""h65 L3 조사 — 왜 코사인 top-3 이 적합_전체에서 무작위보다 나쁜가."""
exec(open('/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad/h65_instrument.py').read().split("METRICS=['A'")[0])
import numpy as np
# 도메인 정보를 다시 붙인다
pools2={}
for ss in P:
    blk=P[ss]; prof=[(d,n) for d,n in blk['profile']]; out=[]
    for dom in ('wn','steam','tmdb'):
        for c in blk['deep'][dom][:3]+blk['sample'][dom]:
            n=c['n']; gs=[]; ok=True
            for pd_,pn in prof:
                if pn==n: continue
                g=PAIRS.get((pd_,pn,dom,n))
                if g is None: ok=False; break
                gs.append(g)
            if not ok or len(gs)<2: continue
            cs=max(float(np.dot(uv(EMB[pd_][ROW[pd_][pn]]), uv(EMB[dom][ROW[dom][n]]-LAM*CEN[dom])))
                   for pd_,pn in prof if pn!=n)
            out.append(dict(dom=dom,rank=c['rank'],gs=gs,mx=max(gs),
                            allfit=(sum(gs)/len(gs)>=1.5 and min(gs)>=1),cos=cs,
                            sharp=(max(gs)==3 and min(gs)==0)))
    pools2[ss]=out

def rate(sel,key):
    v=[c[key] for ss in pools2 for c in sel(pools2[ss])]
    return 100*np.mean(v), len(v)
def bycos(p,k): return sorted(p,key=lambda c:-c['cos'])[:k]
def perdom(p,k):  # 도메인별 코사인 상위 k (§8-1 쿼터)
    out=[]
    for d in ('wn','steam','tmdb'):
        out+=sorted([c for c in p if c['dom']==d],key=lambda c:-c['cos'])[:k]
    return out
rng=np.random.default_rng(20260904)
print(f"{'선택 방식':<28} {'n':>4} {'적합_전체':>9} {'적합_max':>9} {'날카(3&0)':>10} {'평균 등급':>9}")
def show(tag,sel):
    a,n=rate(sel,'allfit'); m,_=rate(sel,'mx' if False else 'allfit')
    mx=100*np.mean([c['mx']>=2 for ss in pools2 for c in sel(pools2[ss])])
    sh=100*np.mean([c['sharp'] for ss in pools2 for c in sel(pools2[ss])])
    av=np.mean([np.mean(c['gs']) for ss in pools2 for c in sel(pools2[ss])])
    print(f"{tag:<28} {n:>4} {a:>8.1f}% {mx:>8.1f}% {sh:>9.1f}% {av:>9.2f}")
show("전역 코사인 top-3",       lambda p: bycos(p,3))
show("전역 코사인 top-6",       lambda p: bycos(p,6))
show("도메인별 코사인 top-1 (3칸)", lambda p: perdom(p,1))
show("도메인별 코사인 top-2 (6칸)", lambda p: perdom(p,2))
show("풀 전체 (24건)",          lambda p: p)
show("무작위 3건 (1회 예시)",     lambda p: list(rng.choice(p,3,replace=False)))
# 코사인 순위와 '날카로움'의 관계
allc=[c for ss in pools2 for c in pools2[ss]]
allc_sorted=sorted(allc,key=lambda c:-c['cos'])
q=len(allc_sorted)//4
print(f"\n코사인 4분위별 (전체 {len(allc)}건):")
print(f"{'분위':<10} {'적합_전체':>9} {'적합_max':>9} {'날카(3&0)':>10} {'평균 등급':>9}")
for i,lab in enumerate(["최상위 25%","2분위","3분위","최하위 25%"]):
    g=allc_sorted[i*q:(i+1)*q] if i<3 else allc_sorted[3*q:]
    print(f"{lab:<10} {100*np.mean([c['allfit'] for c in g]):>8.1f}% "
          f"{100*np.mean([c['mx']>=2 for c in g]):>8.1f}% "
          f"{100*np.mean([c['sharp'] for c in g]):>9.1f}% "
          f"{np.mean([np.mean(c['gs']) for c in g]):>9.2f}")
