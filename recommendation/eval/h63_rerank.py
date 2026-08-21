"""h63 — 대각 재가중 릿지가 코사인을 이기는가. 등록 ab4aa8b98fa3e9cf36a78418fa5d6ce6.
신규 채점 0. 봉인된 등급만 사용. 10겹 leave-one-seed-out 을 한 번에 돌린다."""
import json, collections
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; ALPHAS=[0.1,1,10,100,1000]; SEED_SHUF=20260901

def frame(path,idcol,cols):
    d=pd.read_parquet(f"{path}/dataset.parquet", columns=[idcol]+cols)
    i=pd.read_parquet(f"{path}/corpus_index.parquet").sort_values("embedding_row")
    return d.set_index(idcol).loc[i[idcol].to_numpy()].reset_index()
DF={'wn':frame(WN,"item_id",["name"]),'steam':frame(STEAM,"steam_appid",["name"]),
    'tmdb':frame(TMDB,"item_id",["name"])}
EMB={d:np.asarray(np.load(p+"/corpus_embeddings.npy",mmap_mode="r"),dtype=np.float32)
     for d,p in (('wn',WN),('steam',STEAM),('tmdb',TMDB))}
CEN={d:EMB[d].mean(axis=0) for d in EMB}
ROW={d:{} for d in DF}
for d in DF:
    for i,n in enumerate(DF[d]["name"]): ROW[d].setdefault(str(n), i)
def unit(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v
def vec(d,name):   return unit(EMB[d][ROW[d][name]])
def vec_hub(d,name):  # 허브니스 보정본(후보쪽에만 적용 — 검색과 동일)
    return unit(EMB[d][ROW[d][name]] - LAM*CEN[d])

# ---- 봉인된 등급 전부 ----
def loadG(p):
    ns={}; exec(open(f"{SP}/{p}").read(),ns); return ns['G']
PAIRS={}   # (rdom,rname,cdom,cand) -> grade
for b in ["h54","h55","h57","h58","h59","h60","h62"]:
    K=json.load(open(f"{SP}/{b}_grade_key.json")); G=loadG(f"{b}_grades.py")
    for k in K:
        if 'rname' not in k or k['pid'] not in G: continue
        key=(k['rdom'],k['rname'],k['dom'],k['cand'])
        if k['rname'] in ROW.get(k['rdom'],{}) and k['cand'] in ROW.get(k['dom'],{}):
            PAIRS[key]=G[k['pid']]
print(f"[데이터] 고유쌍 {len(PAIRS)} · 기준작품 {len({(a,b) for a,b,_,_ in PAIRS})} · 후보 {len({(c,d) for _,_,c,d in PAIRS})}")

# 특징 행렬(전체) — 캐시
keys=list(PAIRS)
X=np.stack([vec(rd,rn)*vec(cd,cn) for rd,rn,cd,cn in keys]).astype(np.float32)
y=np.array([PAIRS[k] for k in keys],dtype=np.float32)
refkey=np.array([f"{rd}|{rn}" for rd,rn,_,_ in keys])
print(f"[특징] X {X.shape} · y 평균 {y.mean():.3f}")

# ---- h62 시드 10개의 시험 풀 ----
P=json.load(open(f"{SP}/h62_pools.json"))
def eval_pool(ss):
    blk=P[ss]; prof=[(d,n) for d,n in blk['profile']]
    out=[]
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

def top3_mean(scores, truth):
    idx=np.argsort(-np.asarray(scores))[:3]
    return float(np.mean([truth[i] for i in idx]))

def run(y_used, tag):
    per=[]; overlap_tot=0; wstats=[]
    for ss in P:
        pool=eval_pool(ss)
        prof={(d,n) for d,n in P[ss]['profile']}
        profkeys={f"{d}|{n}" for d,n in prof}
        tr=~np.isin(refkey, list(profkeys))
        Xtr,ytr=X[tr],y_used[tr]
        # 훈련 폴드 내부 CV 로만 alpha 선택
        grp=refkey[tr]; best=(None,np.inf)
        gkf=GroupKFold(n_splits=5)
        for a in ALPHAS:
            errs=[]
            for ti,vi in gkf.split(Xtr,ytr,groups=grp):
                m=Ridge(alpha=a).fit(Xtr[ti],ytr[ti])
                errs.append(np.mean((m.predict(Xtr[vi])-ytr[vi])**2))
            e=float(np.mean(errs))
            if e<best[1]: best=(a,e)
        alpha=best[0]
        model=Ridge(alpha=alpha).fit(Xtr,ytr)
        w=model.coef_; wstats.append((np.abs(w).mean(), np.std(w)))
        cos_s=[]; lrn_s=[]; truth=[]
        for dom,n,profl,gs in pool:
            cv=vec_hub(dom,n)
            cs=max(float(np.dot(unit(EMB[pd_][ROW[pd_][pn]]), cv)) for pd_,pn in profl if pn!=n)
            ps=max(float(model.predict((vec(pd_,pn)*vec(dom,n))[None,:])[0]) for pd_,pn in profl if pn!=n)
            cos_s.append(cs); lrn_s.append(ps); truth.append(max(gs))
        orc=float(np.mean(sorted(truth,reverse=True)[:3]))
        c=top3_mean(cos_s,truth); l=top3_mean(lrn_s,truth)
        cand_names={n for _,n,_,_ in pool}
        ov=sum(1 for rd,rn,cd,cn in np.array(keys,dtype=object)[tr] if cn in cand_names)
        overlap_tot+=ov
        per.append(dict(seed=ss,n=len(pool),alpha=alpha,cos=c,learn=l,oracle=orc,overlap=ov))
    return per, wstats

def report(per,wstats,tag):
    C=np.mean([p['cos'] for p in per]); L=np.mean([p['learn'] for p in per])
    O=np.mean([p['oracle'] for p in per])
    win=sum(1 for p in per if p['learn']>p['cos']); tie=sum(1 for p in per if p['learn']==p['cos'])
    print(f"\n===== {tag} =====")
    print(f"{'시드':>5} {'n':>3} {'alpha':>6} {'코사인':>7} {'학습':>7} {'오라클':>7} {'차이':>7}")
    for p in per:
        print(f"{p['seed']:>5} {p['n']:>3} {p['alpha']:>6} {p['cos']:>7.2f} {p['learn']:>7.2f} {p['oracle']:>7.2f} {p['learn']-p['cos']:>+7.2f}")
    print(f"{'평균':>5} {'':>3} {'':>6} {C:>7.3f} {L:>7.3f} {O:>7.3f} {L-C:>+7.3f}")
    print(f"승 {win} · 무 {tie} · 패 {len(per)-win-tie}   |  코사인/오라클 = {100*C/O:.1f}%")
    print(f"학습된 w: |w| 평균 {np.mean([a for a,_ in wstats]):.4f} · 표준편차 {np.mean([b for _,b in wstats]):.4f}")
    print(f"후보 겹침(훈련에 등장한 시험 후보 쌍) 폴드 평균 {np.mean([p['overlap'] for p in per]):.0f}")
    return C,L,O,win

per,ws=run(y,"실제 등급")
C,L,O,win=report(per,ws,"J1 · 실제 등급")

rng=np.random.default_rng(SEED_SHUF); ysh=y.copy(); rng.shuffle(ysh)
per2,ws2=run(ysh,"뒤섞은 등급")
C2,L2,O2,win2=report(per2,ws2,"J3 · 음성 대조(등급 뒤섞음)")

print("\n===== 판정 =====")
d=L-C
print(f"J1: 차이 {d:+.3f} 등급 (기준 +0.30) · 승 {win}/10 (기준 7)")
print(f"  → {'적중' if (d>=0.30 and win>=7) else '빗나감'}")
print(f"J2 천장: 코사인/오라클 {100*C/O:.1f}% → {'해석 가능(<90%)' if 100*C/O<90 else '여지 없음(≥90%)'}")
d2=L2-C2
print(f"J3 음성 대조: 차이 {d2:+.3f} · 승 {win2}/10 → {'절차 이상! J1 폐기' if (d2>=0.30 and win2>=7) else '정상(새지 않음)'}")
