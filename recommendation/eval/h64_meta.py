"""h64 — 임베딩 밖 메타데이터에 잔여 재랭킹 신호가 있는가. 등록 5709ce09507725b7bd060b4a230c4160.
신규 채점 0. h63 프로토콜(10겹 leave-one-seed-out) 그대로."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; SHUF=20260902
FEAT=['pop','pop_missing','rating','rating_missing','textlen','hub']

def frame(path,idcol,cols):
    d=pd.read_parquet(f"{path}/dataset.parquet", columns=[idcol]+cols)
    i=pd.read_parquet(f"{path}/corpus_index.parquet").sort_values("embedding_row")
    return d.set_index(idcol).loc[i[idcol].to_numpy()].reset_index()
wd=frame(WN,"item_id",["name","synopsis","interest_count","rating"])
sd=frame(STEAM,"steam_appid",["name","semantic_text","recommendations_total"])
td=frame(TMDB,"item_id",["name","overview","vote_count","vote_average"])
DF={'wn':wd,'steam':sd,'tmdb':td}
EMB={d:np.asarray(np.load(p+"/corpus_embeddings.npy",mmap_mode="r"),dtype=np.float32)
     for d,p in (('wn',WN),('steam',STEAM),('tmdb',TMDB))}
CEN={d:EMB[d].mean(axis=0) for d in EMB}
ROW={d:{} for d in DF}
for d in DF:
    for i,n in enumerate(DF[d]["name"]): ROW[d].setdefault(str(n), i)

def z(x):
    x=np.asarray(x,dtype=np.float64); m,s=np.nanmean(x),np.nanstd(x)
    return (x-m)/s if s>0 else x*0.0
# 도메인 내 표준화된 특징 테이블 (사전 등록 §3 그대로)
def unitv(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v
META={}
for d,(popcol,ratecol,txtcol) in (('wn',("interest_count","rating","synopsis")),
                                  ('steam',("recommendations_total",None,"semantic_text")),
                                  ('tmdb',("vote_count","vote_average","overview"))):
    df=DF[d]; n=len(df)
    pr=pd.to_numeric(df[popcol],errors="coerce").to_numpy(dtype=float)
    popmiss=(~np.isfinite(pr))|(pr<=0)
    pop=z(np.log1p(np.where(np.isfinite(pr),np.maximum(pr,0),0.0)))
    if ratecol is None:
        rate=np.zeros(n); ratemiss=np.ones(n)
    else:
        rr=pd.to_numeric(df[ratecol],errors="coerce").to_numpy(dtype=float)
        ratemiss=(~np.isfinite(rr)).astype(float)
        rate=z(np.where(np.isfinite(rr),rr,np.nanmean(rr)))
    tl=z(np.log1p(df[txtcol].fillna("").str.len().to_numpy(dtype=float)))
    E=EMB[d]; c=unitv(CEN[d])
    hub=z(np.array([float(np.dot(unitv(E[i]),c)) for i in range(n)]))
    META[d]=np.stack([pop,popmiss.astype(float),rate,ratemiss,tl,hub],axis=1)
    print(f"[특징] {d:>5} n={n} · pop 비결측 {100*(1-popmiss.mean()):.1f}% · rating 비결측 {100*(1-ratemiss.mean()):.1f}%")

def loadG(p):
    ns={}; exec(open(f"{SP}/{p}").read(),ns); return ns['G']
PAIRS={}
for b in ["h54","h55","h57","h58","h59","h60","h62"]:
    K=json.load(open(f"{SP}/{b}_grade_key.json")); G=loadG(f"{b}_grades.py")
    for k in K:
        if 'rname' not in k or k['pid'] not in G: continue
        if k['rname'] in ROW.get(k['rdom'],{}) and k['cand'] in ROW.get(k['dom'],{}):
            PAIRS[(k['rdom'],k['rname'],k['dom'],k['cand'])]=G[k['pid']]
keys=list(PAIRS); y=np.array([PAIRS[k] for k in keys],dtype=float)
refkey=np.array([f"{rd}|{rn}" for rd,rn,_,_ in keys])
def cospair(rd,rn,cd,cn):
    return float(np.dot(unitv(EMB[rd][ROW[rd][rn]]), unitv(EMB[cd][ROW[cd][cn]]-LAM*CEN[cd])))
COS=np.array([cospair(*k) for k in keys])
MET=np.stack([META[cd][ROW[cd][cn]] for _,_,cd,cn in keys])
X1=np.column_stack([COS,MET]); X2=MET
print(f"[데이터] 쌍 {len(keys)} · 기준작품 {len(set(refkey))} · y 평균 {y.mean():.3f}")

P=json.load(open(f"{SP}/h62_pools.json"))
def pool_of(ss):
    blk=P[ss]; prof=[(d,n) for d,n in blk['profile']]; out=[]
    for dom in ('wn','steam','tmdb'):
        for c in blk['deep'][dom][:3]+blk['sample'][dom]:
            n=c['n']; gs=[]; ok=True
            for pd_,pn in prof:
                if pn==n: continue
                g=PAIRS.get((pd_,pn,dom,n))
                if g is None: ok=False; break
                gs.append(g)
            if ok and len(gs)>=2: out.append((dom,n,prof,gs))
    return out
def top3(sc,tr):
    return float(np.mean([tr[i] for i in np.argsort(-np.asarray(sc))[:3]]))

def run(yv,tag):
    per=[]; coefs=[]; corr={'M0':[],'M1':[],'M2':[]}; partial=[]
    for ss in P:
        pk={f"{d}|{n}" for d,n in P[ss]['profile']}
        tr=~np.isin(refkey,list(pk)); te=~tr
        m1=LinearRegression().fit(X1[tr],yv[tr]); m2=LinearRegression().fit(X2[tr],yv[tr])
        coefs.append(m1.coef_.copy())
        pool=pool_of(ss); prof=P[ss]['profile']
        s0=[];s1=[];s2=[];truth=[]
        for dom,n,profl,gs in pool:
            cs=[];f1=[];f2=[]
            for pd_,pn in profl:
                if pn==n: continue
                c=cospair(pd_,pn,dom,n); mv=META[dom][ROW[dom][n]]
                cs.append(c); f1.append(float(m1.predict(np.r_[c,mv][None,:])[0]))
                f2.append(float(m2.predict(mv[None,:])[0]))
            s0.append(max(cs)); s1.append(max(f1)); s2.append(max(f2)); truth.append(max(gs))
        orc=float(np.mean(sorted(truth,reverse=True)[:3]))
        per.append(dict(seed=ss,n=len(pool),M0=top3(s0,truth),M1=top3(s1,truth),M2=top3(s2,truth),oracle=orc))
        if te.sum()>3:
            corr['M0'].append(np.corrcoef(COS[te],yv[te])[0,1])
            corr['M1'].append(np.corrcoef(m1.predict(X1[te]),yv[te])[0,1])
            corr['M2'].append(np.corrcoef(m2.predict(X2[te]),yv[te])[0,1])
            # K2: 코사인 잔차와 각 특징의 부분 상관
            b=LinearRegression().fit(COS[tr][:,None],yv[tr])
            res=yv[te]-b.predict(COS[te][:,None])
            partial.append([np.corrcoef(MET[te][:,j],res)[0,1] if np.std(MET[te][:,j])>0 else np.nan
                            for j in range(MET.shape[1])])
    return per,np.array(coefs),corr,np.array(partial)

def report(per,coefs,corr,partial,tag):
    A=lambda k: float(np.mean([p[k] for p in per]))
    M0,M1,M2,O=A('M0'),A('M1'),A('M2'),A('oracle')
    w1=sum(1 for p in per if p['M1']>p['M0']); t1=sum(1 for p in per if p['M1']==p['M0'])
    print(f"\n===== {tag} =====")
    print(f"{'시드':>5} {'n':>3} {'M0 코사인':>9} {'M1 +메타':>9} {'M2 메타만':>9} {'오라클':>7} {'M1−M0':>7}")
    for p in per:
        print(f"{p['seed']:>5} {p['n']:>3} {p['M0']:>9.2f} {p['M1']:>9.2f} {p['M2']:>9.2f} {p['oracle']:>7.2f} {p['M1']-p['M0']:>+7.2f}")
    print(f"{'평균':>5} {'':>3} {M0:>9.3f} {M1:>9.3f} {M2:>9.3f} {O:>7.3f} {M1-M0:>+7.3f}")
    print(f"M1 승 {w1} · 무 {t1} · 패 {len(per)-w1-t1}   |  M0/오라클 = {100*M0/O:.1f}%")
    print(f"보류 시드 등급 상관: M0 {np.nanmean(corr['M0']):.3f} · M1 {np.nanmean(corr['M1']):.3f} · M2 {np.nanmean(corr['M2']):.3f}")
    print(f"M1 계수 평균: cos {coefs[:,0].mean():+.3f} | " +
          " ".join(f"{f} {coefs[:,i+1].mean():+.3f}" for i,f in enumerate(FEAT)))
    print("K2 코사인 잔차와의 부분 상관: " + " ".join(f"{f} {np.nanmean(partial[:,i]):+.3f}" for i,f in enumerate(FEAT)))
    return M0,M1,O,w1

per,co,cr,pa=run(y,"실제 등급")
M0,M1,O,w1=report(per,co,cr,pa,"K1 · 실제 등급")
rng=np.random.default_rng(SHUF); ysh=y.copy(); rng.shuffle(ysh)
per2,co2,cr2,pa2=run(ysh,"뒤섞은 등급")
S0,S1,_,w2=report(per2,co2,cr2,pa2,"K3 · 음성 대조(등급 뒤섞음)")

print("\n===== 판정 =====")
d=M1-M0
print(f"K1: 차이 {d:+.3f} 등급 (기준 +0.30) · 승 {w1}/10 (기준 7) → {'적중' if (d>=0.30 and w1>=7) else '빗나감'}")
print(f"K4 천장: M0/오라클 {100*M0/O:.1f}% → {'해석 가능(<90%)' if 100*M0/O<90 else '여지 없음'}")
d2=S1-S0
print(f"K3 음성 대조: 차이 {d2:+.3f} · 승 {w2}/10 → {'절차 이상! K1 폐기' if (d2>=0.30 and w2>=7) else '정상(새지 않음)'}")
