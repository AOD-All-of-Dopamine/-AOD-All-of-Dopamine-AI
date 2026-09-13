"""h62 집계 — 채점 봉인(h62_grades.py d2c68784e01a64771d68e7acd4cfa54d) 이후에만 실행.
등급 조회는 h54~h62 전 배치의 (기준작품, 후보) 조합을 합쳐서 쓴다."""
import json, collections
SP="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
def load(p):
    ns={}; exec(open(f"{SP}/{p}").read(),ns); return ns['G']
GR={}
for b in ["h54","h55","h57","h58","h59","h60","h62"]:
    K=json.load(open(f"{SP}/{b}_grade_key.json")); G=load(f"{b}_grades.py")
    for k in K:
        if 'rname' in k and k['pid'] in G: GR[(k['rname'],k['cand'])]=G[k['pid']]

P=json.load(open(f"{SP}/h62_pools.json"))
def band(r): return "1-3" if r<=3 else ("4-10" if r<=10 else ("11-20" if r<=20 else "21-30"))
rows=[]; missing=0
for ss,blk in P.items():
    prof=[n for d,n in blk['profile']]
    cands=[(dom,c['rank'],c['n']) for dom in ('wn','steam','tmdb') for c in blk['deep'][dom][:3]]
    cands+=[(dom,c['rank'],c['n']) for dom in ('wn','steam','tmdb') for c in blk['sample'][dom]]
    for dom,r,n in cands:
        gs=[GR.get((pn,n)) for pn in prof if pn!=n]      # D-15 자기쌍 제외
        if any(g is None for g in gs) or len(gs)<2: missing+=1; continue
        rows.append(dict(seed=ss,dom=dom,rank=r,band=band(r),gs=gs,n=n,
                         fit_max=max(gs)>=2,
                         fit_all=(sum(gs)/len(gs)>=1.5 and min(gs)>=1)))
print(f"[집계] 후보 {len(rows)}건 · 제외 {missing}건")

def rate(sel,f):
    s=[r for r in rows if sel(r)]
    return (100*sum(f(r) for r in s)/len(s) if s else float('nan')), len(s)

print("\n=== H2 구간별 ===")
print(f"{'구간':>7} {'n':>4} {'적합_전체':>9} {'적합_max':>9} {'전체÷max':>9}")
res={}
for b in ["1-3","4-10","11-20","21-30"]:
    a,n=rate(lambda r,b=b:r['band']==b, lambda r:r['fit_all'])
    m,_=rate(lambda r,b=b:r['band']==b, lambda r:r['fit_max'])
    res[b]=(a,m,n); print(f"{b:>7} {n:>4} {a:>8.1f}% {m:>8.1f}% {a/m if m else 0:>8.2f}")
deep_all,dn=rate(lambda r:r['band']!="1-3", lambda r:r['fit_all'])
deep_max,_=rate(lambda r:r['band']!="1-3", lambda r:r['fit_max'])
sh_all,sh_n=res["1-3"][0],res["1-3"][2]; sh_max=res["1-3"][1]
print(f"\n랭크 4~30 합산: n={dn} · 적합_전체 {deep_all:.1f}% · 적합_max {deep_max:.1f}%")
ratio=deep_all/sh_all if sh_all else 0
print("\n=== H1 (1차) ===")
print(f"얕은(1-3) {sh_all:.1f}%  vs  깊은(4-30) {deep_all:.1f}%  → 유지율 {100*ratio:.1f}%")
print(f"판정: {'적중 (≥70%)' if ratio>=0.70 else '빗나감 (<70%)'}")
print(f"\n=== H3 천장 ===  랭크1~3 적합_전체 {sh_all:.1f}% → {'해석 가능(≤70%)' if sh_all<=70 else '보류'}")
print(f"=== H4 바닥 ===  랭크4~30 적합_전체 {deep_all:.1f}% → {'정상(≠0)' if deep_all>0 else '사멸'}")

print("\n=== H5 도메인별 ===")
for d in ('wn','steam','tmdb'):
    a1,n1=rate(lambda r,d=d:r['dom']==d and r['band']=="1-3", lambda r:r['fit_all'])
    a2,n2=rate(lambda r,d=d:r['dom']==d and r['band']!="1-3", lambda r:r['fit_all'])
    m2,_ =rate(lambda r,d=d:r['dom']==d and r['band']!="1-3", lambda r:r['fit_max'])
    print(f"  {d:>5} 1-3 {a1:>5.1f}%(n={n1})  4-30 {a2:>5.1f}%(n={n2}) · max {m2:>5.1f}%")

print("\n=== H5 시드별 (적합_전체) ===")
print(f"{'시드':>5} {'프로필 대표':<20} {'1-3':>7} {'4-30':>7} {'유지율':>7}")
seedrows=[]
for ss,blk in P.items():
    a1,_=rate(lambda r,s=ss:r['seed']==s and r['band']=="1-3", lambda r:r['fit_all'])
    a2,_=rate(lambda r,s=ss:r['seed']==s and r['band']!="1-3", lambda r:r['fit_all'])
    seedrows.append((ss,blk['profile'][0][1][:18],a1,a2))
    print(f"{ss:>5} {blk['profile'][0][1][:18]:<20} {a1:>6.1f}% {a2:>6.1f}% {(f'{100*a2/a1:.0f}%' if a1 else 'n/a'):>7}")
win=sum(1 for _,_,a1,a2 in seedrows if a2>a1); tie=sum(1 for _,_,a1,a2 in seedrows if a2==a1)
print(f"시드 단위: 깊은쪽 우세 {win} · 동률 {tie} · 얕은쪽 우세 {len(seedrows)-win-tie}")

print("\n=== 깊은 구간에서 나온 3점 후보 (얕은 풀에 없던 것) ===")
shallow={(r['seed'],r['n']) for r in rows if r['band']=="1-3"}
cnt=0
for r in sorted(rows,key=lambda r:(r['seed'],r['rank'])):
    if r['band']=="1-3" or max(r['gs'])<3: continue
    cnt+=1
    if cnt<=25: print(f"  시드 {r['seed']} · {r['dom']:>5} · 랭크 {r['rank']:>2} · {r['n'][:34]:<34} 등급 {r['gs']}")
print(f"  … 총 {cnt}건")

print("\n=== 추가 기술통계 (사후, 탐색적 — 사전 등록 아님) ===")
def r2(sel,f):
    s=[r for r in rows if sel(r)]
    return (100*sum(f(r) for r in s)/len(s) if s else float('nan')), len(s)
print(f"{'구간':>7} {'n':>4} {'최고=3':>8} {'날카(3&0)':>9} {'평균등급':>8}")
for b in ["1-3","4-10","11-20","21-30"]:
    t,n=r2(lambda r,b=b:r['band']==b, lambda r:max(r['gs'])==3)
    s,_=r2(lambda r,b=b:r['band']==b, lambda r:max(r['gs'])==3 and min(r['gs'])==0)
    m=[sum(r['gs'])/len(r['gs']) for r in rows if r['band']==b]
    print(f"{b:>7} {n:>4} {t:>7.1f}% {s:>8.1f}% {sum(m)/len(m):>7.2f}")
t1,_=r2(lambda r:r['band']=="1-3", lambda r:max(r['gs'])==3)
t2,_=r2(lambda r:r['band']!="1-3", lambda r:max(r['gs'])==3)
print(f"\n최고등급 3 보유 비율: 얕은 {t1:.1f}% vs 깊은 {t2:.1f}%")
u=len({r['n'] for r in rows if r['band']!="1-3"} - {r['n'] for r in rows if r['band']=="1-3"})
print(f"깊은 구간에만 등장한 서로 다른 작품: {u}건")
