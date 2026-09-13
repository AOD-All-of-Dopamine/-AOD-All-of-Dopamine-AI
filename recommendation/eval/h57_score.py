"""h57 판정 — 등록 md5 70dffd0f357e7304530c539c477a53b3.
라벨 616d0f6c325822981d62fb9831bc47c4 · 채점 52f30f958f289849d4be64203001d2b3."""
import json, math, collections, random, sys
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L as L50
from h51_labels import L as L51
from h53_labels import L as L53
from h56_labels import L as L56
from h57_labels import L as L57
from h50_grades import G as G50
from h51_grades import G as G51
from h53_grades import G as G53
from h54_grades import G as G54
from h55_grades import G as G55
from h57_grades import G as G57
DOMS=('wn','steam','tmdb')

tags={}
for k in json.load(open(f'{SP}/h50_key.json')): tags[(k['dom'] or 'wn', k['name'])]=L50[k['id']]
for k in json.load(open(f'{SP}/h51_key.json')): tags[('wn',k['name'])]=L51[k['id']]
for k in json.load(open(f'{SP}/h53_key.json')): tags[(k['dom'],k['name'])]=L53[k['id']]
for k in json.load(open(f'{SP}/h56_key.json')): tags[(k['dom'],k['name'])]=L56[k['id']]
for k in json.load(open(f'{SP}/h57_key.json')): tags[(k['dom'],k['name'])]=L57[k['id']]

have={}
seedname={s:x['wn'] for s,x in enumerate(json.load(open(f'{SP}/h50_seeds20.json')))}
for k in json.load(open(f'{SP}/h50_grade_key.json')): have[('wn',seedname[k['seed']],k['dom'],k['cand_name'])]=G50[k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')): have[('wn',seedname[k['seed']],'wn',k['cand_name'])]=G51[k['pid']]
for k in json.load(open(f'{SP}/h53_grade_key.json')): have[('wn',seedname[k['seed']],k['dom'],k['cand_name'])]=G53[k['pid']]
for k in json.load(open(f'{SP}/h54_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G54[k['pid']]
for k in json.load(open(f'{SP}/h55_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G55[k['pid']]
for k in json.load(open(f'{SP}/h57_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G57[k['pid']]

def cos(a,b):
    X,Y=set(a),set(b); return len(X&Y)/math.sqrt(len(X)*len(Y)) if X and Y else 0.0
def fit_all(g): return (sum(g)/3.0>=1.5) and (min(g)>=1)
def fit_max(g): return max(g)>=2

P=json.load(open(f'{SP}/h57_pools.json'))
rng=random.Random(20260916)
res={r:dict(all=0,mx=0,n=0,seeds=0,dom=collections.Counter(),arm=collections.Counter()) for r in ('U','I','M','R')}
inter_sizes=[]; skipped=[]; ngrade=[]
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
                items.append(dict(dom=dom, arm=arm, tg=tags[(dom,c['n'])],
                                  all=fit_all(g), mx=fit_max(g), g=g))
    ngrade.append(len(items))
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
        for i in top:
            res[r]['dom'][items[i]['dom']]+=1
            res[r]['arm'][items[i]['arm']]+=1

def p(r,k): return 100.0*res[r][k]/res[r]['n']
NAME={'U':'U 합집합(분업)','I':'I 교집합(합성)','M':'M 최소(합성)','R':'R 무작위'}
print("="*78); print("h57 판정 — 태그 최소(M) 랭커 확증 · 신규 시드 10~19"); print("="*78)
print(f"  시드 {len(P)}개 · 후보 {ngrade} · top-5")
print(f"  교집합 크기: {sorted(inter_sizes)} · 빈 시드 {len(skipped)}개 → I 분모 {res['I']['seeds']}시드\n")
print(f"  {'랭커':16s} {'적합_전체':>10s} {'적합_max':>10s}   {'wn/st/tm':>10s}  {'D/C':>7s}")
for r in ('U','I','M','R'):
    d=res[r]['dom']; a=res[r]['arm']
    print(f"  {NAME[r]:16s} {p(r,'all'):9.1f}% {p(r,'mx'):9.1f}%   {d['wn']:>3}/{d['steam']:>2}/{d['tmdb']:>2}  {a['D']:>3}/{a['C']:<3}")

print("\n[등록 판정]")
b4 = p('R','all') <= 70
print(f"  B4 천장 가드: 무작위 적합_전체 {p('R','all'):.1f}% (기준 ≤70%)   {'통과' if b4 else '탈락 — 배치 무효'}")
if not b4: sys.exit()
b1 = (p('M','all')-p('R','all')) >= 10
b2 = (p('M','all')-p('U','all')) >= 10
b3 = (p('U','all')-p('R','all')) >= 10
print(f"  B1 1차: M − R = {p('M','all')-p('R','all'):+.1f}%p (기준 ≥+10%p)   {'적중' if b1 else '빗나감'}")
print(f"  B2 2차: M − U = {p('M','all')-p('U','all'):+.1f}%p (기준 ≥+10%p)   {'적중' if b2 else '빗나감'}")
print(f"  B3 재현: U − R = {p('U','all')-p('R','all'):+.1f}%p (기준 ≥+10%p)   {'적중' if b3 else '빗나감'}")
print(f"  (참고) I − U = {p('I','all')-p('U','all'):+.1f}%p")
print("\n[B5 기술 통계]")
print(f"  적합_max: U {p('U','mx'):.1f}% · I {p('I','mx'):.1f}% · M {p('M','mx'):.1f}% · R {p('R','mx'):.1f}%")
allg=[g for ss,blk in P.items() for k in ()]
