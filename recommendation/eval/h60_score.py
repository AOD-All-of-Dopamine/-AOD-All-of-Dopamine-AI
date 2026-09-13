"""h60 판정 — 등록 f6d8d1aaee88fb364a8c7f3d75dc01bd · 라벨 1607bea474f2ff3fdf4e7e74e2fd99d4"""
import json, math, random, sys, statistics as st, collections
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h60_labels import A as LA, C as LC
from h50_grades import G as G50
from h51_grades import G as G51
from h53_grades import G as G53
from h54_grades import G as G54
from h55_grades import G as G55
from h57_grades import G as G57
from h58_grades import G as G58
from h59_grades import G as G59
from h60_grades import G as G60
DOMS=('wn','steam','tmdb')
have={}
sn={s:x['wn'] for s,x in enumerate(json.load(open(f'{SP}/h50_seeds20.json')))}
for k in json.load(open(f'{SP}/h50_grade_key.json')): have[('wn',sn[k['seed']],k['dom'],k['cand_name'])]=G50[k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')): have[('wn',sn[k['seed']],'wn',k['cand_name'])]=G51[k['pid']]
for k in json.load(open(f'{SP}/h53_grade_key.json')): have[('wn',sn[k['seed']],k['dom'],k['cand_name'])]=G53[k['pid']]
for k in json.load(open(f'{SP}/h54_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G54[k['pid']]
for k in json.load(open(f'{SP}/h55_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G55[k['pid']]
for k in json.load(open(f'{SP}/h57_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G57[k['pid']]
for k in json.load(open(f'{SP}/h58_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G58[k['pid']]
for k in json.load(open(f'{SP}/h59_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G59[k['pid']]
for k in json.load(open(f'{SP}/h60_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G60[k['pid']]
tag={}
for k in json.load(open(f'{SP}/h60_key.json')): tag[(k['dom'],k['name'])]=(LA[k['id']], LC[k['id']])
def cos(a,b):
    X,Y=set(a),set(b); return len(X&Y)/math.sqrt(len(X)*len(Y)) if X and Y else 0.0
def fit_all(g): return (sum(g)/3.0>=1.5) and (min(g)>=1)
def fit_max(g): return max(g)>=2

P=json.load(open(f'{SP}/h60_pools.json'))
rng=random.Random(20260826)
res={}; tops={}
for sysid,idx in (('추상',0),('구체',1)):
    tot=0; cnt=0; mx=0; empty=0; isz=[]; picks={}
    for ss,blk in P.items():
        prof=blk['profile']
        T=[set(tag[(d,n)][idx]) for d,n in prof]
        U=set().union(*T); I=set.intersection(*T)
        if idx==0 or True:
            isz.append(len(I))
            if not I: empty+=1
        items=[]
        for d in DOMS:
            for c in blk['D'][d]:
                g=[have[(rd,rn,d,c['n'])] for rd,rn in prof]
                items.append(dict(n=c['n'], dom=d, tg=tag[(d,c['n'])][idx],
                                  fit=fit_all(g), mx=fit_max(g)))
        sc=[cos(it['tg'],U) for it in items]
        top=sorted(range(len(items)), key=lambda i:(-sc[i], i))[:5]
        picks[ss]={items[i]['n'] for i in top}
        tot+=sum(items[i]['fit'] for i in top); mx+=sum(items[i]['mx'] for i in top); cnt+=len(top)
    res[sysid]=dict(fit=100.0*tot/cnt, mx=100.0*mx/cnt, empty=empty, isz=sorted(isz), n=cnt)
    tops[sysid]=picks
# 무작위
rv=[]
for ss,blk in P.items():
    prof=blk['profile']
    f=[fit_all([have[(rd,rn,d,c['n'])] for rd,rn in prof]) for d in DOMS for c in blk['D'][d]]
    vals=[sum(f[i] for i in rng.sample(range(len(f)), 5))/5 for _ in range(2000)]
    rv.append(100.0*st.mean(vals))
RND=st.mean(rv)
print("="*78); print("h60 판정 — 축 체계: 추상 14축 vs 구체 트로프 14축"); print("="*78)
print(f"  {'체계':8s} {'적합_전체':>10s} {'적합_max':>10s} {'비율':>7s} {'교집합 빈 시드':>14s}")
for s in ('추상','구체'):
    r=res[s]
    print(f"  {s:8s} {r['fit']:9.1f}% {r['mx']:9.1f}% {r['fit']/r['mx'] if r['mx'] else 0:7.2f} {r['empty']:12d}/10")
print(f"  {'무작위':8s} {RND:9.1f}%   (2,000회 평균)")
print("\n[등록 판정]")
f3 = RND<=70
print(f"  F3 천장 가드(= 풀 적합률): 무작위 {RND:.1f}% (기준 ≤70%)   {'통과' if f3 else '탈락 — 배치 무효'}")
cA=collections.Counter(x for v in LA.values() for x in v); cC=collections.Counter(x for v in LC.values() for x in v)
mA=100*cA.most_common(1)[0][1]/len(LA); mC=100*cC.most_common(1)[0][1]/len(LC)
f4 = mA<=60 and mC<=60
print(f"  F4 퇴화 가드(양쪽): 추상 최빈 {mA:.1f}% · 구체 최빈 {mC:.1f}% (기준 ≤60%)   {'통과' if f4 else '탈락'}")
d=res['구체']['fit']-res['추상']['fit']
print(f"  F1 1차: 구체 − 추상 = {d:+.1f}%p (기준 ≥+10%p)   {'적중' if d>=10 else '빗나감'}")
print(f"  F2 기제: 구체 교집합 빈 시드 {res['구체']['empty']}/10 (기준 ≤3)   {'적중' if res['구체']['empty']<=3 else '빗나감'}")
print("\n[F5 기술 통계]")
print(f"  교집합 크기: 추상 {res['추상']['isz']} · 구체 {res['구체']['isz']}")
ov=[len(tops['추상'][ss] & tops['구체'][ss])/5 for ss in P]
print(f"  두 체계 top-5 겹침: 평균 {100*st.mean(ov):.1f}% · 시드별 {[int(100*x) for x in ov]}")
print(f"  무작위 대비: 추상 {res['추상']['fit']-RND:+.1f}%p · 구체 {res['구체']['fit']-RND:+.1f}%p")
print(f"  축 사용 최빈 3: 추상 {cA.most_common(3)} · 구체 {cC.most_common(3)}")
