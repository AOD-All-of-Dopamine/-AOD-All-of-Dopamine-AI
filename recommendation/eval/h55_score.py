"""h55 판정 — 등록 md5 b93803c8fd1eb15b0e128bb86c16bfd5. 신규 등급 b28aa45a…"""
import json, collections, random, sys, statistics
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_grades import G as G50
from h51_grades import G as G51
from h53_grades import G as G53
from h54_grades import G as G54
from h55_grades import G as G55
DOMS=('wn','steam','tmdb')

have={}
seedname={s:x['wn'] for s,x in enumerate(json.load(open(f'{SP}/h50_seeds20.json')))}
for k in json.load(open(f'{SP}/h50_grade_key.json')): have[('wn',seedname[k['seed']],k['dom'],k['cand_name'])]=G50[k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')): have[('wn',seedname[k['seed']],'wn',k['cand_name'])]=G51[k['pid']]
for k in json.load(open(f'{SP}/h53_grade_key.json')): have[('wn',seedname[k['seed']],k['dom'],k['cand_name'])]=G53[k['pid']]
for k in json.load(open(f'{SP}/h54_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G54[k['pid']]
for k in json.load(open(f'{SP}/h55_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G55[k['pid']]

P=json.load(open(f'{SP}/h55_pools.json'))
def grades(prof, dom, name): return [have[(rd,rn,dom,name)] for rd,rn in prof]
def fit_all(g):  return (sum(g)/3.0 >= 1.5) and (min(g) >= 1)   # 1차 기준: 합성
def fit_max(g):  return max(g) >= 2                              # h54 기준

res={a:dict(all=0,mx=0,n=0, dom=collections.Counter(), domall=collections.Counter(),
            solo_in=0, solo_out=0, allsolo_in=0, allsolo_out=0, tri=0) for a in ('D','C')}
rndall=[]; rndmax=[]
rng=random.Random(20260914)
# 무작위 기준선: 두 갈래 후보 풀 전체에서 무작위 추출
allc=[]
for ss,blk in P.items():
    for arm in ('D','C'):
        for dom in DOMS:
            for c in blk[arm][dom]: allc.append((blk['profile'],dom,c['n']))
for _ in range(2000):
    prof,dom,n = rng.choice(allc); g=grades(prof,dom,n)
    rndall.append(fit_all(g)); rndmax.append(fit_max(g))

for ss,blk in P.items():
    prof=blk['profile']
    for arm in ('D','C'):
        doms_hit=set()
        for dom in DOMS:
            for c in blk[arm][dom]:
                g=grades(prof,dom,c['n']); fa=fit_all(g); fm=fit_max(g)
                r=res[arm]; r['n']+=1; r['all']+=fa; r['mx']+=fm
                r['dom'][dom]+=1; r['domall'][dom]+=fa
                if c['solo']: r['solo_in']+=1;  r['allsolo_in']+=fa
                else:         r['solo_out']+=1; r['allsolo_out']+=fa
                if fa: doms_hit.add(dom)
        res[arm]['tri'] += len(doms_hit)==3

D,C=res['D'],res['C']
def p(x,n): return 100.0*x/n if n else 0.0
print("="*78); print("h55 판정 — 분업(max 질의) vs 합성(min 질의)"); print("="*78)
print(f"  갈래별 후보 {D['n']} (10시드 × 3도메인 × 3)\n")
print(f"  {'':28s} {'분업 D':>10s} {'합성 C':>10s}   차이")
print(f"  {'적합_전체 (평균≥1.5 & 최소≥1)':28s} {p(D['all'],D['n']):9.1f}% {p(C['all'],C['n']):9.1f}%   {p(C['all'],C['n'])-p(D['all'],D['n']):+.1f}%p")
print(f"  {'적합_max (h54 기준)':28s} {p(D['mx'],D['n']):9.1f}% {p(C['mx'],C['n']):9.1f}%   {p(C['mx'],C['n'])-p(D['mx'],D['n']):+.1f}%p")
print(f"  {'세 도메인 모두 적합_전체 시드':28s} {D['tri']:>8d}/10 {C['tri']:>8d}/10   {C['tri']-D['tri']:+d}")
print(f"\n  무작위 기준선 (2000회 추출): 적합_전체 {100*sum(rndall)/len(rndall):.1f}% · 적합_max {100*sum(rndmax)/len(rndmax):.1f}%")
print("\n  [도메인별 적합_전체]")
for d in DOMS:
    print(f"    {d:6s} D {p(D['domall'][d],D['dom'][d]):5.1f}%  C {p(C['domall'][d],C['dom'][d]):5.1f}%   {p(C['domall'][d],C['dom'][d])-p(D['domall'][d],D['dom'][d]):+5.1f}%p")

print("\n[등록 판정]")
rb=100*sum(rndall)/len(rndall)
t2 = rb <= 70
print(f"  T2 천장 가드: 무작위 적합_전체 {rb:.1f}% (기준 ≤70%)   {'통과' if t2 else '탈락 — 배치 무효'}")
if not t2: sys.exit("  ⇒ 무효. T1·T3·T4 해석 금지.")
t3 = p(C['solo_out'], C['n']) >= 30
print(f"  T3 공허 방지: C 후보 중 단일항목 top-20 **밖** {p(C['solo_out'],C['n']):.1f}% (기준 ≥30%)   {'통과' if t3 else '탈락 — T1 해석 금지'}")
t1 = (p(C['all'],C['n']) - p(D['all'],D['n'])) >= 10
t4 = (p(C['mx'],C['n']) - p(D['mx'],D['n'])) >= -15
print(f"  T1 핵심: 적합_전체 C−D = {p(C['all'],C['n'])-p(D['all'],D['n']):+.1f}%p (기준 ≥+10%p)   {'적중' if t1 else '빗나감'}")
print(f"  T4 교환비: 적합_max C−D = {p(C['mx'],C['n'])-p(D['mx'],D['n']):+.1f}%p (기준 ≥−15%p)   {'적중' if t4 else '빗나감'}")

print("\n[T5 · 기술 통계]")
print(f"  C 갈래 · 단일항목 top-20 안 {C['solo_in']}건 적합_전체 {p(C['allsolo_in'],C['solo_in']):.1f}%"
      f"  |  밖 {C['solo_out']}건 {p(C['allsolo_out'],C['solo_out']):.1f}%")
print(f"  D 갈래 · 단일항목 top-20 안 {D['solo_in']}건 적합_전체 {p(D['allsolo_in'],D['solo_in']):.1f}%"
      f"  |  밖 {D['solo_out']}건 {p(D['allsolo_out'],D['solo_out']) if D['solo_out'] else float('nan'):.1f}%")
# 등급 벡터 모양
for arm in ('D','C'):
    mins=[]; means=[]
    for ss,blk in P.items():
        prof=blk['profile']
        for dom in DOMS:
            for c in blk[arm][dom]:
                g=grades(prof,dom,c['n']); mins.append(min(g)); means.append(sum(g)/3)
    print(f"  {arm}: 최솟값 평균 {statistics.mean(mins):.2f} · 3항목 평균의 평균 {statistics.mean(means):.2f} · 최솟값≥1 비율 {100*sum(1 for m in mins if m>=1)/len(mins):.1f}%")
