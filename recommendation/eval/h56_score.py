"""h56 판정 — 등록 md5 386765b2962e63ed3fcfc73c362f9328. 라벨 968936d6… · 신규 채점 0건."""
import json, math, collections, random, sys
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L as L50
from h51_labels import L as L51
from h53_labels import L as L53
from h56_labels import L as L56
from h50_grades import G as G50
from h51_grades import G as G51
from h53_grades import G as G53
from h54_grades import G as G54
from h55_grades import G as G55
DOMS=('wn','steam','tmdb')

tags={}
for k in json.load(open(f'{SP}/h50_key.json')): tags[(k['dom'] or 'wn', k['name'])]=L50[k['id']]
for k in json.load(open(f'{SP}/h51_key.json')): tags[('wn',k['name'])]=L51[k['id']]
for k in json.load(open(f'{SP}/h53_key.json')): tags[(k['dom'],k['name'])]=L53[k['id']]
for k in json.load(open(f'{SP}/h56_key.json')): tags[(k['dom'],k['name'])]=L56[k['id']]

have={}
seedname={s:x['wn'] for s,x in enumerate(json.load(open(f'{SP}/h50_seeds20.json')))}
for k in json.load(open(f'{SP}/h50_grade_key.json')): have[('wn',seedname[k['seed']],k['dom'],k['cand_name'])]=G50[k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')): have[('wn',seedname[k['seed']],'wn',k['cand_name'])]=G51[k['pid']]
for k in json.load(open(f'{SP}/h53_grade_key.json')): have[('wn',seedname[k['seed']],k['dom'],k['cand_name'])]=G53[k['pid']]
for k in json.load(open(f'{SP}/h54_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G54[k['pid']]
for k in json.load(open(f'{SP}/h55_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G55[k['pid']]

def cos(a,b):
    X,Y=set(a),set(b); return len(X&Y)/math.sqrt(len(X)*len(Y)) if X and Y else 0.0
def fit_all(g): return (sum(g)/3.0>=1.5) and (min(g)>=1)
def fit_max(g): return max(g)>=2

P=json.load(open(f'{SP}/h55_pools.json'))
rng=random.Random(20260916)
res={r:dict(all=0,mx=0,n=0,seeds=0,dom=collections.Counter()) for r in ('U','I','M','R')}
inter_sizes=[]; skipped=[]
for ss,blk in P.items():
    prof=blk['profile']
    A=[set(tags[(d,n)]) for d,n in prof]
    U=set().union(*A); I=set.intersection(*A)
    inter_sizes.append(len(I))
    seen=set(); items=[]
    for arm in ('D','C'):
        for dom in DOMS:
            for c in blk[arm][dom]:
                if (dom,c['n']) in seen: continue
                seen.add((dom,c['n']))
                g=[have[(rd,rn,dom,c['n'])] for rd,rn in prof]
                items.append(dict(dom=dom, tg=tags[(dom,c['n'])], all=fit_all(g), mx=fit_max(g)))
    scores={'U':[cos(it['tg'],U) for it in items],
            'M':[min(cos(it['tg'],a) for a in A) for it in items],
            'R':[rng.random() for _ in items]}
    if I: scores['I']=[cos(it['tg'],I) for it in items]
    else: skipped.append(ss)
    for r,sc in scores.items():
        top=sorted(range(len(items)), key=lambda i:-sc[i])[:5]
        res[r]['seeds']+=1; res[r]['n']+=len(top)
        res[r]['all'] += sum(items[i]['all'] for i in top)
        res[r]['mx']  += sum(items[i]['mx'] for i in top)
        for i in top: res[r]['dom'][items[i]['dom']]+=1

def p(r,k): return 100.0*res[r][k]/res[r]['n']
NAME={'U':'U 합집합(분업)','I':'I 교집합(합성)','M':'M 최소(합성)','R':'R 무작위'}
print("="*78); print("h56 판정 — fun_tag 랭커: 합집합(분업) vs 교집합·최소(합성) vs 무작위"); print("="*78)
print(f"  고정 풀: 시드당 18후보(h55 D+C) · top-5 · 신규 채점 0건")
print(f"  교집합 크기: {sorted(inter_sizes)} · 빈 시드 {len(skipped)}개 → I 랭커 분모 {res['I']['seeds']}시드\n")
print(f"  {'랭커':16s} {'적합_전체':>10s} {'적합_max':>10s}   {'도메인 구성 wn/st/tm'}")
for r in ('U','I','M','R'):
    d=res[r]['dom']
    print(f"  {NAME[r]:16s} {p(r,'all'):9.1f}% {p(r,'mx'):9.1f}%   {d['wn']}/{d['steam']}/{d['tmdb']}")

print("\n[등록 판정]")
a4 = p('R','all') <= 70
print(f"  A4 천장 가드: 무작위 적합_전체 {p('R','all'):.1f}% (기준 ≤70%)   {'통과' if a4 else '탈락 — 배치 무효'}")
if not a4: sys.exit()
a1 = (p('I','all')-p('U','all')) >= 10
a2 = (p('M','all')-p('U','all')) >= 10
a3 = (p('U','all')-p('R','all')) >= 10
print(f"  A1 핵심: I − U = {p('I','all')-p('U','all'):+.1f}%p (기준 ≥+10%p)   {'적중' if a1 else '빗나감'}")
print(f"  A2 변형: M − U = {p('M','all')-p('U','all'):+.1f}%p (기준 ≥+10%p)   {'적중' if a2 else '빗나감'}")
print(f"  A3 fun_tag 재검: U − R = {p('U','all')-p('R','all'):+.1f}%p (기준 ≥+10%p)   {'적중' if a3 else '빗나감'}")
print("\n[A5 · 기술 통계]")
print(f"  적합_max 로 보면: U {p('U','mx'):.1f}% · I {p('I','mx'):.1f}% · M {p('M','mx'):.1f}% · R {p('R','mx'):.1f}%")
print(f"  U − R (적합_max 기준) = {p('U','mx')-p('R','mx'):+.1f}%p")
