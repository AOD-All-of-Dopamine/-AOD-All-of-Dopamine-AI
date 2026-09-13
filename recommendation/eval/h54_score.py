"""h54 판정 — 등록 md5 884cfa6e6b1c0fbfd200919400acd139. 신규 등급 bbc3527e…"""
import json, math, collections, random, sys
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L as L50
from h51_labels import L as L51
from h53_labels import L as L53
from h50_grades import G as G50
from h51_grades import G as G51
from h53_grades import G as G53
from h54_grades import G as G54
DOMS=('wn','steam','tmdb'); SEEDS=list(range(10))

tags={}
for k in json.load(open(f'{SP}/h50_key.json')):
    tags[(k['dom'],k['name'])]=L50[k['id']] if k['kind']=='cand' else None
    if k['kind']!='cand': tags[('wn',k['name'])]=L50[k['id']]
for k in json.load(open(f'{SP}/h51_key.json')): tags[('wn',k['name'])]=L51[k['id']]
for k in json.load(open(f'{SP}/h53_key.json')): tags[(k['dom'],k['name'])]=L53[k['id']]

seedgrade={}
for k in json.load(open(f'{SP}/h50_grade_key.json')): seedgrade[(k['seed'],k['dom'],k['cand_name'])]=G50[k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')): seedgrade[(k['seed'],'wn',k['cand_name'])]=G51[k['pid']]
for k in json.load(open(f'{SP}/h53_grade_key.json')): seedgrade[(k['seed'],k['dom'],k['cand_name'])]=G53[k['pid']]

# 프로필 항목 기준 등급 (rdom,rname,dom,cand) -> grade
refgrade={}
for k in json.load(open(f'{SP}/h54_grade_key.json')):
    refgrade[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G54[k['pid']]

D=json.load(open(f'{SP}/h53_pools.json')); pools=D['pools']; profs=D['profiles']

def fits(s, arm, dom, name):
    """새 정답: 프로필 3항목 중 최댓값 ≥ 2"""
    best=seedgrade[(s,dom,name)]
    if best>=2: return True, best
    for rdom, rname in profs[str(s)][arm][1:]:
        g=refgrade.get((rdom,rname,dom,name))
        if g is not None: best=max(best,g)
    return best>=2, best

def cos(a,b):
    X,Y=set(a),set(b); return len(X&Y)/math.sqrt(len(X)*len(Y)) if X and Y else 0.0

rng=random.Random(20260912)
res={a:dict(tri=0,wn=0,fit=0,rnd=0,fit_seed=0,dom=collections.Counter(),
            pf=collections.Counter(), pf_seed=collections.Counter(), pt=collections.Counter())
     for a in ('M','S')}
for s in SEEDS:
    ss=str(s); rsel=rng.sample(range(30),5)
    for arm in ('M','S'):
        items=[]
        for d in DOMS:
            for i,c in enumerate(pools[ss][arm][d]):
                ok,_=fits(s,arm,d,c['n'])
                items.append(dict(dom=d, rank=i, n=c['n'], ok=ok,
                                  ok_seed=seedgrade[(s,d,c['n'])]>=2, tg=tags[(d,c['n'])]))
        ptags=set()
        for d,n in profs[ss][arm]: ptags |= set(tags[(d,n)])
        ft=[cos(ptags,it['tg']) for it in items]
        top=sorted(range(30), key=lambda i:-ft[i])[:5]
        r=res[arm]
        r['tri'] += len({items[i]['dom'] for i in top})==3
        r['wn']  += sum(1 for i in top if items[i]['dom']=='wn')
        r['fit'] += sum(1 for i in top if items[i]['ok'])
        r['fit_seed'] += sum(1 for i in top if items[i]['ok_seed'])
        r['rnd'] += sum(1 for i in rsel if items[i]['ok'])
        for i in top: r['dom'][items[i]['dom']]+=1
        for it in items:
            r['pt'][it['dom']]+=1
            r['pf'][it['dom']]+= it['ok']
            r['pf_seed'][it['dom']]+= it['ok_seed']

M,S=res['M'],res['S']
print("="*78); print("h54 판정 — 정답 기준을 프로필 전체(max)로 바꾸면 h53 이 뒤집히는가"); print("="*78)
print(f"  시드 10개 · 갈래별 top-5 = 50칸 · 갈래·도메인별 풀 = 100후보\n")
print(f"  ft top-5 세 도메인 시드   M {M['tri']:>3d}/10   S {S['tri']:>3d}/10   {M['tri']-S['tri']:+d}")
print(f"  ft top-5 웹소설 칸        M {M['wn']:>3d}/50   S {S['wn']:>3d}/50   {M['wn']-S['wn']:+d}")
print(f"  ft top-5 적합률(프로필)   M {M['fit']:>3d}/50   S {S['fit']:>3d}/50   {M['fit']-S['fit']:+d}")
print(f"  무작위 top-5 적합률       M {M['rnd']:>3d}/50   S {S['rnd']:>3d}/50")
for arm in ('M','S'):
    d=res[arm]['dom']; print(f"  {arm} ft top-5 도메인 구성: wn {d['wn']} / steam {d['steam']} / tmdb {d['tmdb']}")

print("\n  [풀 적합률 — 새 기준(프로필 max) / 옛 기준(시드만)]")
for d in DOMS:
    a=100*M['pf'][d]/M['pt'][d]; b=100*S['pf'][d]/S['pt'][d]
    a0=100*M['pf_seed'][d]/M['pt'][d]; b0=100*S['pf_seed'][d]/S['pt'][d]
    print(f"    {d:6s} 새: M {a:5.1f}%  S {b:5.1f}%  ({a-b:+5.1f}%p)   |   옛: M {a0:5.1f}%  S {b0:5.1f}%  ({a0-b0:+5.1f}%p)")

print("\n[등록 판정]")
def pct(x,n): return 100.0*x/n
u4 = all((pct(res[a]['fit'],50)-pct(res[a]['rnd'],50))>=10 for a in ('M','S'))
print(f"  U4 건전성: ft − 무작위  M {pct(M['fit'],50)-pct(M['rnd'],50):+.1f}%p · S {pct(S['fit'],50)-pct(S['rnd'],50):+.1f}%p"
      f" (기준 +10%p 둘 다)  {'통과' if u4 else '탈락'}")
if not u4: sys.exit("  ⇒ 무효. U1~U3 해석 금지.")
st=pct(M['pf']['steam'],M['pt']['steam'])-pct(S['pf']['steam'],S['pt']['steam'])
tm=pct(M['pf']['tmdb'],M['pt']['tmdb'])-pct(S['pf']['tmdb'],S['pt']['tmdb'])
u1=(st>=10) and (tm>=10)
u2=(pct(M['fit'],50)-pct(S['fit'],50))>=-5
u3=(M['tri']-S['tri'])>=3
print(f"  U1 Steam {st:+.1f}%p · TMDB {tm:+.1f}%p (기준 둘 다 ≥+10%p)   {'적중' if u1 else '빗나감'}")
print(f"  U2 적합률 M−S = {pct(M['fit'],50)-pct(S['fit'],50):+.1f}%p (기준 ≥−5%p)      {'적중' if u2 else '빗나감'}")
print(f"  U3 세 도메인 시드 M−S = {M['tri']-S['tri']:+d} (기준 ≥+3)          {'적중' if u3 else '빗나감'}")

print("\n[U5 · 기술 통계 — D-12 의 크기]")
for arm in ('M','S'):
    r=res[arm]
    tot=sum(r['pt'].values()); f_new=sum(r['pf'].values()); f_old=sum(r['pf_seed'].values())
    print(f"  {arm} 풀 전체: 시드 기준 {100*f_old/tot:5.1f}% → 프로필 기준 {100*f_new/tot:5.1f}%  (+{100*(f_new-f_old)/tot:.1f}%p)")
    print(f"     top-5:   시드 기준 {2*r['fit_seed']:5.1f}% → 프로필 기준 {2*r['fit']:5.1f}%  (+{2*(r['fit']-r['fit_seed']):.1f}%p)")
print("\n  [U5 · 도메인 내 우위가 게임·영화 출발에서도 성립하는가 — 참조 도메인별 적합률]")
by=collections.defaultdict(lambda: [0,0])
for k in json.load(open(f'{SP}/h54_grade_key.json')):
    g=G54[k['pid']]; b=by[(k['rdom'],k['dom'])]
    b[0]+= g>=2; b[1]+=1
for rd in DOMS:
    row=" · ".join(f"→{d} {100*by[(rd,d)][0]/by[(rd,d)][1]:4.1f}% (n={by[(rd,d)][1]})" for d in DOMS if by[(rd,d)][1])
    print(f"    기준 {rd:6s} {row}")
