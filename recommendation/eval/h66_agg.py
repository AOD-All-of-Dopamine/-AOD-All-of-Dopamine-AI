"""h66 — D-19 양방향 시험 + 쿼터 위에서 계측기 재측정. 등록 11b8323e6774398badf3bdd8c5c75ff9."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; NDRAW=2000; RNG=20260905
def frame(path,idcol):
    d=pd.read_parquet(f"{path}/dataset.parquet", columns=[idcol,"name"])
    i=pd.read_parquet(f"{path}/corpus_index.parquet").sort_values("embedding_row")
    return d.set_index(idcol).loc[i[idcol].to_numpy()].reset_index()
DF={'wn':frame(WN,"item_id"),'steam':frame(STEAM,"steam_appid"),'tmdb':frame(TMDB,"item_id")}
EMB={d:np.asarray(np.load(p+"/corpus_embeddings.npy",mmap_mode="r"),dtype=np.float32)
     for d,p in (('wn',WN),('steam',STEAM),('tmdb',TMDB))}
CEN={d:EMB[d].mean(axis=0) for d in EMB}
ROW={d:{} for d in DF}
for d in DF:
    for i,n in enumerate(DF[d]["name"]): ROW[d].setdefault(str(n), i)
def uv(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v
def loadG(p):
    ns={}; exec(open(f"{SP}/{p}").read(),ns); return ns['G']
PAIRS={}
for b in ["h54","h55","h57","h58","h59","h60","h62"]:
    K=json.load(open(f"{SP}/{b}_grade_key.json")); G=loadG(f"{b}_grades.py")
    for k in K:
        if 'rname' in k and k['pid'] in G and k['rname'] in ROW.get(k['rdom'],{}) and k['cand'] in ROW.get(k['dom'],{}):
            PAIRS[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G[k['pid']]
P=json.load(open(f"{SP}/h62_pools.json"))
AGGS=['max','top2_mean','mean','min']
def agg(v,how):
    v=sorted(v,reverse=True)
    return {'max':v[0],'top2_mean':float(np.mean(v[:2])),'mean':float(np.mean(v)),'min':v[-1]}[how]

pools={}
for ss in P:
    blk=P[ss]; prof=[(d,n) for d,n in blk['profile']]; out=[]
    for dom in ('wn','steam','tmdb'):
        for c in blk['deep'][dom][:3]+blk['sample'][dom]:
            n=c['n']; gs=[]; cs=[]; ok=True
            for pd_,pn in prof:
                if pn==n: continue
                g=PAIRS.get((pd_,pn,dom,n))
                if g is None: ok=False; break
                gs.append(g)
                cs.append(float(np.dot(uv(EMB[pd_][ROW[pd_][pn]]), uv(EMB[dom][ROW[dom][n]]-LAM*CEN[dom]))))
            if not ok or len(gs)<2: continue
            out.append(dict(dom=dom,gs=gs,cs=cs,mx=max(gs),
                            allfit=(sum(gs)/len(gs)>=1.5 and min(gs)>=1),
                            sharp=(max(gs)==3 and min(gs)==0),
                            sc={a:agg(cs,a) for a in AGGS}))
    pools[ss]=out
print("[풀] 시드 "+str(len(pools))+" · 후보 "+str(sum(len(v) for v in pools.values())))

def quota(pool, key, k=1):
    """도메인별 상위 k (확정 파이프라인). key(c) -> 점수"""
    out=[]
    for d in ('wn','steam','tmdb'):
        out += sorted([c for c in pool if c['dom']==d], key=lambda c:-key(c))[:k]
    return out
def rates(sel):
    picks=[c for ss in pools for c in sel(pools[ss])]
    return (100*np.mean([c['allfit'] for c in picks]),
            100*np.mean([c['mx']>=2 for c in picks]),
            100*np.mean([c['sharp'] for c in picks]), len(picks))

print("\n=== N1·N2·N3·N4 — 집계 규칙별 (도메인별 top-1, 3칸) ===")
print(f"{'집계':<11} {'n':>4} {'적합_전체':>9} {'적합_max':>9} {'전체÷max':>9} {'날카':>7}")
res={}
for a in AGGS:
    af,mx,sh,n=rates(lambda p,a=a: quota(p, lambda c: c['sc'][a]))
    res[a]=(af,mx,sh); print(f"{a:<11} {n:>4} {af:>8.1f}% {mx:>8.1f}% {af/mx if mx else 0:>9.2f} {sh:>6.1f}%")

rng=np.random.default_rng(RNG)
rand_af=[]; rand_mx=[]
for _ in range(NDRAW):
    picks=[]
    for ss in pools:
        p=pools[ss]
        for d in ('wn','steam','tmdb'):
            cand=[c for c in p if c['dom']==d]
            if cand: picks.append(cand[rng.integers(len(cand))])
    rand_af.append(100*np.mean([c['allfit'] for c in picks]))
    rand_mx.append(100*np.mean([c['mx']>=2 for c in picks]))
s_af=float(np.std(rand_af,ddof=1)); s_mx=float(np.std(rand_mx,ddof=1))
print(f"\n[무작위 {NDRAW}회 · 같은 쿼터 규칙] 적합_전체 {np.mean(rand_af):.1f}% (σ={s_af:.2f}) · "
      f"적합_max {np.mean(rand_mx):.1f}% (σ={s_mx:.2f})")

d1=res['min'][0]-res['max'][0]; d2=res['min'][1]-res['max'][1]
print("\n===== 판정 =====")
n1 = d1 >= 2*s_af
n2 = d2 <= -2*s_mx
print(f"N1 min > max on 적합_전체 : {d1:+.1f}%p (기준 +{2*s_af:.1f}%p) → {'적중' if n1 else '빗나감'}")
print(f"N2 min < max on 적합_max  : {d2:+.1f}%p (기준 −{2*s_mx:.1f}%p) → {'적중' if n2 else '빗나감'}")
print(f"⇒ D-19: {'확증' if (n1 and n2) else ('부분 적중' if (n1 or n2) else '기제로서 기각')}")
af_seq=[res[a][0] for a in AGGS]; mx_seq=[res[a][1] for a in AGGS]; sh_seq=[res[a][2] for a in AGGS]
mono=lambda v,up: all((v[i+1]>=v[i]) if up else (v[i+1]<=v[i]) for i in range(len(v)-1))
print(f"N3 단조성 (max→top2_mean→mean→min): 적합_전체 {'↑단조' if mono(af_seq,True) else '비단조'} {af_seq} · "
      f"적합_max {'↓단조' if mono(mx_seq,False) else '비단조'} {mx_seq}")
print(f"N4 날카 비율: {dict(zip(AGGS,[round(x,1) for x in sh_seq]))} → "
      f"{'max 최고·min 최저 (기제 부합)' if (sh_seq[0]==max(sh_seq) and sh_seq[-1]==min(sh_seq)) else '기제 불일치'}")

# ---- N5: 쿼터 위에서 계측기 재측정 ----
print("\n=== N5 · 쿼터(도메인별 top-2, 6칸) 위에서 계측기 재측정 ===")
def qorder(pool,key,k=2): return quota(pool,key,k)
def metrics(pool, key):
    sel=qorder(pool,key,2)                     # 6칸
    sel=sorted(sel,key=lambda c:-key(c))
    mxs=[c['mx'] for c in sel]
    disc=1/np.log2(np.arange(2,len(mxs[:5])+2))
    ideal=sorted([c['mx'] for c in pool],reverse=True)[:5]
    dd=float(np.sum(np.array(ideal)*(1/np.log2(np.arange(2,len(ideal)+2)))))
    scores=[key(c) for c in pool]; truth=[c['mx'] for c in pool]
    r=spearmanr(scores,truth).statistic
    return {'A':float(np.mean(mxs[:3])),'B':float(np.mean(mxs[:5])),'C':float(np.mean(mxs)),
            'D':(float(np.sum(np.array(mxs[:5])*disc))/dd if dd>0 else np.nan),
            'E':100*float(np.mean([c['allfit'] for c in sel[:3]])),
            'F':float(r) if np.isfinite(r) else 0.0}
MET=['A','B','C','D','E','F']
cosm={m:np.mean([metrics(pools[ss], lambda c: c['sc']['max'])[m] for ss in pools]) for m in MET}
draws={m:[] for m in MET}
for _ in range(NDRAW):
    vals={m:[] for m in MET}
    for ss in pools:
        rv={id(c):rng.random() for c in pools[ss]}
        r=metrics(pools[ss], lambda c: rv[id(c)])
        for m in MET: vals[m].append(r[m])
    for m in MET: draws[m].append(np.mean(vals[m]))
NAME={'A':'상위3 최고등급 평균','B':'상위5 최고등급 평균','C':'6칸 최고등급 평균',
      'D':'NDCG@5','E':'상위3 적합_전체 %','F':'스피어만(24건)'}
print(f"{'지표':>4} {'설명':<20} {'코사인':>9} {'무작위':>9} {'σ':>8} {'SNR':>7}")
snr={}
for m in MET:
    a=np.array(draws[m]); mu,sd=a.mean(),a.std(ddof=1)
    snr[m]=(cosm[m]-mu)/sd if sd>0 else np.nan
    print(f"{m:>4} {NAME[m]:<20} {cosm[m]:>9.3f} {mu:>9.3f} {sd:>8.4f} {snr[m]:>7.2f}")
best=max(MET,key=lambda m:snr[m])
print(f"\n쿼터 위 최고 SNR: {best} ({snr[best]:.2f}) · h65 채택은 F ({snr['F']:.2f})"
      f" → {'유지' if best=='F' else '**정정 필요**'}")
