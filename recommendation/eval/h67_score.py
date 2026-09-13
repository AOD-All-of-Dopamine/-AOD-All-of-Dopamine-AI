"""h67 집계 — 등록 96fd17ea84e4d04b635f0a9785af58a1 · 부록 7b774b90640992f563c0fd0003155b9b.
부록 조치 1: (tmdb, 하모니) 등급은 h62 것을 재사용하지 않는다."""
import json, collections, numpy as np
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
def loadG(p):
    ns={}; exec(open(f"{SP}/{p}").read(),ns); return ns['G']
# 기존 봉인 등급 (h67 신규가 우선; 하모니 키는 기존 것을 버린다)
OLD={}
for b in ["h54","h55","h57","h58","h59","h60","h62"]:
    K=json.load(open(f"{SP}/{b}_grade_key.json")); G=loadG(f"{b}_grades.py")
    for k in K:
        if 'rname' in k and k['pid'] in G:
            if k['rdom']=='tmdb' and k['rname']=='하모니': continue   # D-20 부록 조치 1
            OLD[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G[k['pid']]
NEW={}
K7=json.load(open(f"{SP}/h67_grade_key.json")); G7=loadG("h67_grades.py")
for k in K7: NEW[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G7[k['pid']]
PAIRS={**OLD,**NEW}
P=json.load(open(f"{SP}/h67_pools.json"))
ARMS=('R-max','R-min','R-cent')

def build(arm):
    out=[]; miss=0
    for ss,rec in P.items():
        prof=[(d,n) for d,n in rec['profile']]
        for dom,lst in rec[arm].items():
            for c in lst:
                n=c['n']; gs=[]
                for pd_,pn in prof:
                    if pn==n: continue
                    g=PAIRS.get((pd_,pn,dom,n))
                    if g is None: gs=None; break
                    gs.append(g)
                if gs is None or len(gs)<2: miss+=1; continue
                out.append(dict(seed=ss,dom=dom,rank=c['rank'],n=n,gs=gs,
                    mx=max(gs), allfit=(sum(gs)/len(gs)>=1.5 and min(gs)>=1),
                    allgood=(min(gs)>=2), sharp=(max(gs)==3 and min(gs)==0)))
    return out,miss
D={a:build(a) for a in ARMS}
for a in ARMS: print(f"[{a}] 후보 {len(D[a][0])}건 · 등급 누락 제외 {D[a][1]}건")

def top1(rows):   # 도메인별 1위 = 3칸
    sel=[]
    for ss in P:
        for dom in ('wn','steam','tmdb'):
            c=[r for r in rows if r['seed']==ss and r['dom']==dom]
            if c: sel.append(sorted(c,key=lambda r:r['rank'])[0])
    return sel
print(f"\n{'검색':<8} {'n':>4} {'적합_전체':>9} {'적합_max':>9} {'전체÷max':>9} {'모두좋음':>9} {'날카':>7} {'평균등급':>8}")
res={}
for a in ARMS:
    sel=top1(D[a][0])
    af=100*np.mean([r['allfit'] for r in sel]); mx=100*np.mean([r['mx']>=2 for r in sel])
    ag=100*np.mean([r['allgood'] for r in sel]); sh=100*np.mean([r['sharp'] for r in sel])
    av=np.mean([np.mean(r['gs']) for r in sel])
    res[a]=(af,mx,ag,sh,len(sel))
    print(f"{a:<8} {len(sel):>4} {af:>8.1f}% {mx:>8.1f}% {af/mx if mx else 0:>9.2f} {ag:>8.1f}% {sh:>6.1f}% {av:>8.2f}")

# 후보 수준(도메인별 2건 전부) — P2 용
print(f"\n[후보 수준 · 도메인별 상위 2건 전부]")
print(f"{'검색':<8} {'n':>4} {'모두좋음(≥2,≥2,≥2)':>18} {'전문가(3&0)':>12} {'모두무난(≤1)':>13}")
p2={}
for a in ARMS:
    rows=D[a][0]
    ag=100*np.mean([r['allgood'] for r in rows]); sh=100*np.mean([r['sharp'] for r in rows])
    bl=100*np.mean([max(r['gs'])<=1 for r in rows]); p2[a]=ag
    print(f"{a:<8} {len(rows):>4} {ag:>17.1f}% {sh:>11.1f}% {bl:>12.1f}%")

# 무작위 대조 2000회 (같은 선택 규칙: 도메인별 1건)
rng=np.random.default_rng(20260907)
pool=collections.defaultdict(list)
for a in ARMS:
    for r in D[a][0]: pool[(r['seed'],r['dom'])].append(r)
keys=sorted(pool)
draws_af=[]; draws_mx=[]
for _ in range(2000):
    sel=[pool[k][rng.integers(len(pool[k]))] for k in keys]
    draws_af.append(100*np.mean([r['allfit'] for r in sel]))
    draws_mx.append(100*np.mean([r['mx']>=2 for r in sel]))
s_af=float(np.std(draws_af,ddof=1)); s_mx=float(np.std(draws_mx,ddof=1))
print(f"\n[무작위 2000회 · 세 검색이 가져온 후보 전체에서 도메인별 1건]")
print(f"  적합_전체 {np.mean(draws_af):.1f}% (σ={s_af:.2f}) · 적합_max {np.mean(draws_mx):.1f}% (σ={s_mx:.2f})")

print("\n===== 판정 =====")
d1=res['R-min'][0]-res['R-max'][0]
print(f"P1  R-min 적합_전체 {res['R-min'][0]:.1f}% vs R-max {res['R-max'][0]:.1f}% = {d1:+.1f}%p "
      f"(기준 +{2*s_af:.1f}%p) → {'적중' if d1>=2*s_af else '빗나감'}")
print(f"P2  R-min '모두에게 좋음' {p2['R-min']:.1f}% (기준 20.0%, 기저선 h62 풀 10.0%) "
      f"→ {'적중' if p2['R-min']>=20.0 else '빗나감'}")
d3=res['R-min'][1]-res['R-max'][1]
print(f"P3  R-min 적합_max {res['R-min'][1]:.1f}% vs R-max {res['R-max'][1]:.1f}% = {d3:+.1f}%p (예상된 대가)")
print(f"P4  R-cent: 적합_전체 {res['R-cent'][0]:.1f}% · 적합_max {res['R-cent'][1]:.1f}% · 모두좋음 {p2['R-cent']:.1f}%")
ok1 = d1>=2*s_af; ok2 = p2['R-min']>=20.0
print(f"\n⇒ 결론: " + ("(가) 검색이 문제였다 — R-min 방향" if (ok1 and ok2) else
      ("기제는 작동, 표본 부족 — 재확인 필요" if ok2 else
       ("통과율만 올랐다 — 의심, 재현 필요" if ok1 else
        "(나) 코퍼스에 그런 후보가 드물다 — 목표를 적합_max 로 확정"))))
