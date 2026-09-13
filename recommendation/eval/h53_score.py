"""h53 판정 — 등록 md5 2feffc400bd886febe1fdd15ef7ba3d3. 라벨 5d2742bc… · 등급 479a1b95…"""
import json, math, collections, random, sys
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L as L50
from h50_grades import G as G50
from h51_labels import L as L51
from h51_grades import G as G51
from h53_labels import L as L53
from h53_grades import G as G53
DOMS=('wn','steam','tmdb')

# ---------- 라벨(태그) 지도 (dom,name) -> tags ----------
tags={}
for k in json.load(open(f'{SP}/h50_key.json')):
    if k['kind']=='cand': tags[(k['dom'],k['name'])]=L50[k['id']]
    else: tags[('wn',k['name'])]=L50[k['id']]
for k in json.load(open(f'{SP}/h51_key.json')): tags[('wn',k['name'])]=L51[k['id']]
for k in json.load(open(f'{SP}/h53_key.json')): tags[(k['dom'],k['name'])]=L53[k['id']]

# ---------- 등급 지도 (seed,dom,name) -> grade ----------
grade={}
for k in json.load(open(f'{SP}/h50_grade_key.json')): grade[(k['seed'],k['dom'],k['cand_name'])]=G50[k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')): grade[(k['seed'],'wn',k['cand_name'])]=G51[k['pid']]
for k in json.load(open(f'{SP}/h53_grade_key.json')): grade[(k['seed'],k['dom'],k['cand_name'])]=G53[k['pid']]

D=json.load(open(f'{SP}/h53_pools.json')); pools=D['pools']; profs=D['profiles']
seeds=sorted(int(s) for s in pools)

# 커버리지 검증
miss=[(s,a,d,c['n']) for s in pools for a in ('M','S') for d in DOMS for c in pools[s][a][d]
      if (d,c['n']) not in tags or (int(s),d,c['n']) not in grade]
print(f"커버리지 미충족 {len(miss)}건", miss[:5])
if miss: sys.exit("[중단] 라벨/등급 누락")

def cos(a,b):
    X,Y=set(a),set(b); return len(X&Y)/math.sqrt(len(X)*len(Y)) if X and Y else 0.0
def pct(v):
    n=len(v); o=sorted(range(n),key=lambda i:v[i]); r=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and v[o[j+1]]==v[o[i]]: j+=1
        for k in range(i,j+1): r[o[k]]=((i+j)/2)/(n-1)
        i=j+1
    return r

rng=random.Random(20260910)
res={a:dict(tri=0,wn=0,fit=0,rnd=0,Etri=0,Ewn=0,Efit=0,dom=collections.Counter(),
            poolfit=collections.Counter(), pooltot=collections.Counter()) for a in ('M','S')}
ov=[]
for s in seeds:
    ss=str(s)
    rsel=rng.sample(range(30),5)
    for arm in ('M','S'):
        items=[dict(dom=d, rank=i, n=c['n'], gr=grade[(s,d,c['n'])], tg=tags[(d,c['n'])])
               for d in DOMS for i,c in enumerate(pools[ss][arm][d])]
        ptags=set()
        for d,n in profs[ss][arm]: ptags |= set(tags[(d,n)])
        ft=[cos(ptags,it['tg']) for it in items]
        sim=[-it['rank'] for it in items]
        E=[0.643*a+0.357*b for a,b in zip(pct(ft),pct(sim))]
        top=sorted(range(30), key=lambda i:-ft[i])[:5]
        res[arm]['tri'] += len({items[i]['dom'] for i in top})==3
        res[arm]['wn']  += sum(1 for i in top if items[i]['dom']=='wn')
        res[arm]['fit'] += sum(1 for i in top if items[i]['gr']>=2)
        res[arm]['rnd'] += sum(1 for i in rsel if items[i]['gr']>=2)
        for i in top: res[arm]['dom'][items[i]['dom']]+=1
        te=sorted(range(30), key=lambda i:-E[i])[:5]
        res[arm]['Etri'] += len({items[i]['dom'] for i in te})==3
        res[arm]['Ewn']  += sum(1 for i in te if items[i]['dom']=='wn')
        res[arm]['Efit'] += sum(1 for i in te if items[i]['gr']>=2)
        for it in items:
            res[arm]['pooltot'][it['dom']]+=1
            res[arm]['poolfit'][it['dom']]+= it['gr']>=2
    for d in DOMS:
        A={c['n'] for c in pools[ss]['M'][d]}; B={c['n'] for c in pools[ss]['S'][d]}
        ov.append(len(A&B)/10)

print("\n"+"="*78); print("h53 판정 — 풀 수준. 혼합 프로필 M vs 단일 프로필 S (재검색)"); print("="*78)
M,S=res['M'],res['S']
print(f"  ft top-5 세 도메인 시드   M {M['tri']:>3d}/20   S {S['tri']:>3d}/20   {M['tri']-S['tri']:+d}")
print(f"  ft top-5 웹소설 칸        M {M['wn']:>3d}/100  S {S['wn']:>3d}/100  {M['wn']-S['wn']:+d}")
print(f"  ft top-5 적합률           M {M['fit']:>3d}/100  S {S['fit']:>3d}/100  {M['fit']-S['fit']:+d}%p")
print(f"  무작위 top-5 적합률       M {M['rnd']:>3d}/100  S {S['rnd']:>3d}/100")
for arm in ('M','S'):
    d=res[arm]['dom']; print(f"  {arm} ft top-5 도메인 구성: wn {d['wn']} / steam {d['steam']} / tmdb {d['tmdb']}")
print("\n  [풀 수준 적합률 — 후보 200건/갈래/도메인]")
for d in DOMS:
    a=100*M['poolfit'][d]/M['pooltot'][d]; b=100*S['poolfit'][d]/S['pooltot'][d]
    print(f"    {d:6s} M {a:5.1f}%  S {b:5.1f}%   {a-b:+5.1f}%p")

print("\n[등록 판정]")
v5 = all((res[a]['fit']-res[a]['rnd'])>=10 for a in ('M','S'))
print(f"  V5 건전성: ft − 무작위  M {M['fit']-M['rnd']:+d}%p · S {S['fit']-S['rnd']:+d}%p (기준 +10%p 둘 다)  {'통과' if v5 else '탈락'}")
if not v5: sys.exit("  ⇒ 무효. V1~V4 해석 금지.")
v1=(M['tri']-S['tri'])>=6; v2=(M['wn']-S['wn'])<=-15; v3=(M['fit']-S['fit'])>=-5
st=100*M['poolfit']['steam']/M['pooltot']['steam']-100*S['poolfit']['steam']/S['pooltot']['steam']
tm=100*M['poolfit']['tmdb']/M['pooltot']['tmdb']-100*S['poolfit']['tmdb']/S['pooltot']['tmdb']
v4=(st>=10) and (tm>=10)
print(f"  V1 세 도메인 시드 M−S = {M['tri']-S['tri']:+d} (기준 ≥+6)          {'적중' if v1 else '빗나감'}")
print(f"  V2 웹소설 칸 M−S = {M['wn']-S['wn']:+d} (기준 ≤−15)             {'적중' if v2 else '빗나감'}")
print(f"  V3 적합률 M−S = {M['fit']-S['fit']:+d}%p (기준 ≥−5%p)           {'적중' if v3 else '빗나감'}")
print(f"  V4 Steam {st:+.1f}%p · TMDB {tm:+.1f}%p (기준 둘 다 ≥+10%p)  {'적중' if v4 else '빗나감'}")
print(f"  V1·V2 방향 일치: {'예' if v1==v2 else '아니오 — 해석 보류'}")

print("\n[V6 · 기술 통계, 기각 조건 없음]")
print(f"  E(기획서 원안) 세 도메인 시드  M {M['Etri']}/20 · S {S['Etri']}/20")
print(f"  E 웹소설 칸                    M {M['Ewn']}/100 · S {S['Ewn']}/100")
print(f"  E 적합률                       M {M['Efit']}/100 · S {S['Efit']}/100")
print(f"  M/S 풀 겹침 평균 {100*sum(ov)/len(ov):.1f}%")
psz={a:sorted(len(set().union(*[set(tags[(d,n)]) for d,n in profs[str(s)][a]])) for s in seeds) for a in ('M','S')}
for a in ('M','S'): print(f"  프로필 태그 합집합 크기 {a}: 중앙 {psz[a][10]} · 최소 {psz[a][0]} · 최대 {psz[a][-1]}")
