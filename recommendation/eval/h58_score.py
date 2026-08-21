"""h58 판정 — 등록 d1806993ed147edd0b1d1bf1db363473 · 보정 cf34ce30c1fae992d7ccb50bf2f671cd
라벨 f441e39f12ef53ba2549b43012160e8b · 채점 5d64818befdfee57dbd5cd17c4b6ef88 (498쌍)"""
import json, math, random, sys, statistics as st, collections
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h58_grades import G
DOMS=('wn','steam','tmdb')
have={}
for k in json.load(open(f'{SP}/h58_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G[k['pid']]
def fit(g): return (sum(g)/3.0>=1.5) and (min(g)>=1)

P=json.load(open(f'{SP}/h58_pools.json'))
# ---- 후보 목록 구성 (보정 1: 자기 자신 후보 제외) ----
seeds=[]; dropped=0
for ss,blk in P.items():
    prof=blk['profile']; pr={(d,n) for d,n in prof}
    dom_items={d:[] for d in DOMS}; dom_cos={}
    for d in DOMS: dom_cos[d]=st.mean(c['sc'] for c in blk['D'][d])
    seen=set()
    for arm in ('D','C'):
        for d in DOMS:
            for c in blk[arm][d]:
                if (d,c['n']) in pr: dropped+=1; continue
                if (d,c['n']) in seen: continue
                seen.add((d,c['n']))
                dom_items[d].append(dict(n=c['n'], sc=c['sc'],
                                         fit=fit([have[(rd,rn,d,c['n'])] for rd,rn in prof])))
    seeds.append(dict(ss=ss, prof=prof, cos=dom_cos, items=dom_items))
print("="*78); print("h58 판정 — 도메인 게이팅"); print("="*78)
print(f"  시드 {len(seeds)} · 자기 자신 후보 제외 {dropped}슬롯(D-15)")

# ---- C1: 배치 내 도메인×시드 30점, 평균 cos 3분위 ----
pts=[(s['cos'][d], 100.0*sum(x['fit'] for x in s['items'][d])/len(s['items'][d]), d, s['ss'])
     for s in seeds for d in DOMS]
srt=sorted(pts); n=len(srt)//3
hi=st.mean(b for _,b,_,_ in srt[2*n:]); mid=st.mean(b for _,b,_,_ in srt[n:2*n]); lo=st.mean(b for _,b,_,_ in srt[:n])
print(f"\n[C1] 도메인×시드 {len(pts)}점 · 평균 검색 cos 3분위")
print(f"  하위 1/3  cos {st.mean(a for a,_,_,_ in srt[:n]):.3f} → 적합_전체 {lo:5.1f}%")
print(f"  중간 1/3  cos {st.mean(a for a,_,_,_ in srt[n:2*n]):.3f} → 적합_전체 {mid:5.1f}%")
print(f"  상위 1/3  cos {st.mean(a for a,_,_,_ in srt[2*n:]):.3f} → 적합_전체 {hi:5.1f}%")

# ---- 6칸 목록 4전략 ----
def take(items, k): return sorted(items, key=lambda x:-x['sc'])[:k]
def quota(s):  return [x for d in DOMS for x in take(s['items'][d],2)]
def gating(s):
    keep=sorted(DOMS, key=lambda d:-s['cos'][d])[:2]
    return [x for d in keep for x in take(s['items'][d],3)]
def fixed(s):  return [x for d in ('wn','steam') for x in take(s['items'][d],3)]
def rand_mean(s, R, rng):
    pool=[x for d in DOMS for x in s['items'][d]]; out=[]
    for _ in range(R):
        out.append(sum(x['fit'] for x in rng.sample(pool, min(6,len(pool))))/min(6,len(pool)))
    return 100.0*st.mean(out)
rng=random.Random(20260822)
res={}
for name,fn in (('쿼터',quota),('게이팅',gating),('고정제거(TMDB)',fixed)):
    tot=0; cnt=0; dom=collections.Counter()
    for s in seeds:
        L=fn(s); tot+=sum(x['fit'] for x in L); cnt+=len(L)
        for d in DOMS:
            dom[d]+=sum(1 for x in L if x in s['items'][d])
    res[name]=(100.0*tot/cnt, dom)
res['무작위']=(st.mean(rand_mean(s,2000,rng) for s in seeds), None)
print(f"\n[6칸 목록] 전략별 적합_전체")
for name in ('쿼터','게이팅','고정제거(TMDB)','무작위'):
    v,dom=res[name]
    extra = f"  wn/st/tm {dom['wn']}/{dom['steam']}/{dom['tmdb']}" if dom else "  (2000회 평균)"
    print(f"  {name:16s} {v:5.1f}%{extra}")

# ---- 등록 판정 ----
Q,GA,FX,R = res['쿼터'][0], res['게이팅'][0], res['고정제거(TMDB)'][0], res['무작위'][0]
print("\n[등록 판정]")
c4 = R <= 70
print(f"  C4 천장 가드: 무작위 {R:.1f}% (기준 ≤70%)   {'통과' if c4 else '탈락 — 배치 무효'}")
if not c4: sys.exit()
print(f"  C1 1차: 상위−하위 = {hi-lo:+.1f}%p (기준 ≥+10%p)   {'적중' if hi-lo>=10 else '빗나감'}")
print(f"  C2 2차: 게이팅−쿼터 = {GA-Q:+.1f}%p (기준 ≥+10%p)   {'적중' if GA-Q>=10 else '빗나감'}")
print(f"  C3 분해: 게이팅−고정제거 = {GA-FX:+.1f}%p (기준 ≥+10%p)   {'적중' if GA-FX>=10 else '빗나감'}")

# ---- C5 기술 통계 ----
print("\n[C5 기술 통계]")
low=collections.Counter(sorted(DOMS, key=lambda d:-s['cos'][d])[-1] for s in seeds)
print(f"  cos 최하위 도메인 분포: {dict(low)}")
poolfit=sorted((100.0*sum(x['fit'] for d in DOMS for x in s['items'][d])/sum(len(s['items'][d]) for d in DOMS), s['prof'][0][1]) for s in seeds)
print("  시드별 풀 적합률: " + " · ".join(f"{v:.0f}%({n[:9]})" for v,n in poolfit))
xs=[a for a,_,_,_ in pts]; ys=[b for _,b,_,_ in pts]
mx,my=st.mean(xs),st.mean(ys)
num=sum((a-mx)*(b-my) for a,b in zip(xs,ys)); den=math.sqrt(sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))
print(f"  배치 내 상관 r(cos, 적합_전체) = {num/den:+.3f}")
for d in DOMS:
    g=[p for p in pts if p[2]==d]
    print(f"    {d:5s} 평균 cos {st.mean(a for a,_,_,_ in g):.3f} · 평균 적합 {st.mean(b for _,b,_,_ in g):5.1f}%")
