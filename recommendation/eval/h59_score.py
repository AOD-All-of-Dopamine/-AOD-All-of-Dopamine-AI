"""h59 판정 — 등록 6243c4fc23b7d01c910ce10a76808d24 · 채점 (아래 md5)"""
import json, random, sys, statistics as st, collections
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_grades import G as G50
from h51_grades import G as G51
from h53_grades import G as G53
from h54_grades import G as G54
from h55_grades import G as G55
from h57_grades import G as G57
from h58_grades import G as G58
from h59_grades import G as G59
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
def fit_all(g): return (sum(g)/3.0>=1.5) and (min(g)>=1)
def fit_max(g): return max(g)>=2

P=json.load(open(f'{SP}/h59_pools.json'))
rng=random.Random(20260823)
res={}; per_seed={'A':{}, 'B':{}}
for rule in ('A','B'):
    tot=0; cnt=0; mx=0; rnd=[]
    for ss,blk in P.items():
        prof=blk[rule]['profile']
        gs=[[have[(rd,rn,d,c['n'])] for rd,rn in prof]
            for d in DOMS for c in blk[rule]['D'][d]]
        f=[fit_all(g) for g in gs]
        tot+=sum(f); cnt+=len(f); mx+=sum(fit_max(g) for g in gs)
        per_seed[rule][blk['seed']]=100.0*sum(f)/len(f)
        vals=[]
        for _ in range(2000):
            idx=rng.sample(range(len(f)), min(6,len(f)))
            vals.append(sum(f[i] for i in idx)/len(idx))
        rnd.append(100.0*st.mean(vals))
    res[rule]=dict(fit=100.0*tot/cnt, mx=100.0*mx/cnt, n=cnt, rnd=st.mean(rnd))
print("="*78); print("h59 판정 — 프로필 앵커 규칙 A(라벨 제한) vs B(전체 코퍼스)"); print("="*78)
for rule,lab in (('A','규칙 A (라벨 제한)'),('B','규칙 B (전체 코퍼스)')):
    r=res[rule]
    print(f"  {lab:20s} 적합_전체 {r['fit']:5.1f}%  적합_max {r['mx']:5.1f}%  "
          f"비율 {r['fit']/r['mx']:.2f}  무작위 {r['rnd']:5.1f}%  (n={r['n']})")
print("\n[등록 판정]")
e4a = res['A']['rnd']<=70; e4b = res['B']['rnd']<=70
print(f"  E4 천장 가드: A {res['A']['rnd']:.1f}% · B {res['B']['rnd']:.1f}% (기준 ≤70%)   "
      f"{'통과' if e4a and e4b else '탈락'}")
e2 = res['B']['fit']/res['B']['mx'] <= 0.90
print(f"  E2 퇴화 가드: B 적합_전체÷적합_max = {res['B']['fit']/res['B']['mx']:.2f} (기준 ≤0.90)   "
      f"{'통과' if e2 else '탈락 — E1 채택 금지'}")
ov=[]
for ss,blk in P.items():
    for d in DOMS:
        a={c['n'] for c in blk['A']['D'][d]}; b={c['n'] for c in blk['B']['D'][d]}
        ov.append(len(a&b)/3)
e3 = 100*st.mean(ov) <= 70
print(f"  E3 공허 가드: 후보 겹침 {100*st.mean(ov):.1f}% (기준 ≤70%)   {'통과' if e3 else '탈락'}")
d = res['B']['fit']-res['A']['fit']
print(f"  E1 1차: B − A = {d:+.1f}%p (기준 ≥+10%p)   {'적중' if d>=10 else '빗나감'}")
print("\n[E5 기술 통계] 시드별 적합_전체 (짝지은 대조)")
print(f"  {'시드':22s} {'규칙A':>7s} {'규칙B':>7s} {'차이':>8s}")
diffs=[]
for ss,blk in P.items():
    a=per_seed['A'][blk['seed']]; b=per_seed['B'][blk['seed']]
    diffs.append(b-a)
    print(f"  {blk['seed'][:20]:22s} {a:6.0f}% {b:6.0f}% {b-a:+7.0f}%p")
print(f"  짝지은 차이: 평균 {st.mean(diffs):+.1f}%p · 중앙값 {st.median(diffs):+.1f}%p · "
      f"B 우세 {sum(x>0 for x in diffs)} · 동률 {sum(x==0 for x in diffs)} · A 우세 {sum(x<0 for x in diffs)}")
for rule in ('A','B'):
    cs=[c for blk in P.values() for c in blk[rule]['anchor_cos'][1:]]
    print(f"  규칙 {rule} 시드–앵커 cos 평균 {st.mean(cs):.3f}")
