"""h61 판정 — 두 기준이 지목하는 목록이 얼마나 다른가. 등록 352be0457d0e48a8543c87e6f58166b6.
신규 채점 0. 봉인된 h55·h57·h58·h59·h60 데이터만 사용."""
import json, sys, statistics as st, collections
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
def s_all(g): return (2 if ((sum(g)/3.0>=1.5) and (min(g)>=1)) else 0, sum(g))
def s_max(g): return (2 if max(g)>=2 else 0, sum(g))
def pass_all(g): return (sum(g)/3.0>=1.5) and (min(g)>=1)
def pass_max(g): return max(g)>=2

BATCH=[('h55','h55_pools.json','DC'),('h57','h57_pools.json','DC'),
       ('h58','h58_pools.json','DC'),('h59','h59_pools.json','A/B'),
       ('h60','h60_pools.json','D')]
rows=[]; allov=[]; onlyA=[]
for name,fn,kind in BATCH:
    P=json.load(open(f'{SP}/{fn}'))
    ovs=[]
    for ss,blk in P.items():
        blocks=[]
        if kind=='A/B':
            for r in ('A','B'): blocks.append((blk[r]['profile'], blk[r]['D']))
        elif kind=='D':
            blocks.append((blk['profile'], blk['D']))
        else:
            for arm in ('D','C'):
                if arm in blk: blocks.append((blk['profile'], blk[arm]))
        for prof,D in blocks:
            pr={(d,n) for d,n in prof}; seen=set(); items=[]
            for d in DOMS:
                for c in D[d]:
                    if (d,c['n']) in pr or (d,c['n']) in seen: continue
                    seen.add((d,c['n']))
                    g=[have[(rd,rn,d,c['n'])] for rd,rn in prof]
                    items.append((c['n'], g))
            if len(items)<5: continue
            A=sorted(range(len(items)), key=lambda i:(-s_all(items[i][1])[0], -s_all(items[i][1])[1], i))[:5]
            B=sorted(range(len(items)), key=lambda i:(-s_max(items[i][1])[0], -s_max(items[i][1])[1], i))[:5]
            sA={items[i][0] for i in A}; sB={items[i][0] for i in B}
            ovs.append(len(sA&sB)/5); allov.append(len(sA&sB)/5)
            for i in A:
                if items[i][0] not in sB: onlyA.append(items[i][1])
    rows.append((name, len(ovs), 100*st.mean(ovs), 100*st.median(ovs)))
print("="*78); print("h61 판정 — 두 기준이 지목하는 top-5 는 얼마나 겹치는가"); print("="*78)
print(f"  {'배치':6s} {'목록 수':>7s} {'평균 겹침':>10s} {'중앙값':>8s}")
for n,c,m,md in rows: print(f"  {n:6s} {c:7d} {m:9.1f}% {md:7.1f}%")
print(f"  {'전체':6s} {len(allov):7d} {100*st.mean(allov):9.1f}% {100*st.median(allov):7.1f}%")
print("\n[등록 판정]")
g1 = 100*st.mean(allov) >= 70
print(f"  G1 1차: 평균 겹침 {100*st.mean(allov):.1f}% (기준 ≥70%)   {'적중' if g1 else '빗나감'}")
print("\n[G2 기술 통계]")
d=collections.Counter(int(5*x) for x in allov)
print(f"  겹침 분포(5칸 중 몇 개): {dict(sorted(d.items()))}")
print(f"  A(전부 기준)에만 있는 항목 {len(onlyA)}개의 등급 성격:")
if onlyA:
    print(f"    평균 등급 {st.mean(sum(g)/3 for g in onlyA):.2f} · 최솟값 평균 {st.mean(min(g) for g in onlyA):.2f} · 최댓값 평균 {st.mean(max(g) for g in onlyA):.2f}")
    print(f"    적합_전체 통과 {100*st.mean(pass_all(g) for g in onlyA):.0f}% · 적합_max 통과 {100*st.mean(pass_max(g) for g in onlyA):.0f}%")
