"""h65 계측기 연구 — 등록 a9a3ecb4bbfab0a6d4ff198e9cead37e. 신규 채점 0."""
import json, numpy as np, pandas as pd
from scipy.stats import spearmanr
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; NDRAW=2000; RNG=20260903
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

pools={}
for ss in P:
    blk=P[ss]; prof=[(d,n) for d,n in blk['profile']]; out=[]
    for dom in ('wn','steam','tmdb'):
        for c in blk['deep'][dom][:3]+blk['sample'][dom]:
            n=c['n']; gs=[]; ok=True
            for pd_,pn in prof:
                if pn==n: continue
                g=PAIRS.get((pd_,pn,dom,n))
                if g is None: ok=False; break
                gs.append(g)
            if not ok or len(gs)<2: continue
            cs=max(float(np.dot(uv(EMB[pd_][ROW[pd_][pn]]), uv(EMB[dom][ROW[dom][n]]-LAM*CEN[dom])))
                   for pd_,pn in prof if pn!=n)
            out.append(dict(gs=gs, mx=max(gs), allfit=(sum(gs)/len(gs)>=1.5 and min(gs)>=1), cos=cs))
    pools[ss]=out
print("[풀] " + " · ".join(f"{ss}:{len(v)}" for ss,v in pools.items()))

def topk_mean(order, pool, k): return float(np.mean([pool[i]['mx'] for i in order[:k]]))
def ndcg(order, pool, k=5):
    g=[pool[i]['mx'] for i in order[:k]]
    disc=1/np.log2(np.arange(2,len(g)+2))
    ideal=sorted([c['mx'] for c in pool],reverse=True)[:k]
    d=float(np.sum(np.array(ideal)*disc[:len(ideal)]))
    return float(np.sum(np.array(g)*disc))/d if d>0 else np.nan
def allfit3(order, pool): return 100*float(np.mean([pool[i]['allfit'] for i in order[:3]]))
def spear(scores, pool):
    r=spearmanr(scores,[c['mx'] for c in pool]).statistic
    return float(r) if np.isfinite(r) else 0.0

METRICS=['A','B','C','D','E','F']
def evaluate(scores, pool):
    o=list(np.argsort(-np.asarray(scores)))
    return {'A':topk_mean(o,pool,3),'B':topk_mean(o,pool,5),'C':topk_mean(o,pool,10),
            'D':ndcg(o,pool,5),'E':allfit3(o,pool),'F':spear(scores,pool)}

# 양성 대조 = 코사인
cosv={m:np.mean([evaluate([c['cos'] for c in pools[ss]], pools[ss])[m] for ss in pools]) for m in METRICS}
# 음성 대조 = 무작위 2,000회 (D-16)
rng=np.random.default_rng(RNG)
draws={m:[] for m in METRICS}
for _ in range(NDRAW):
    vals={m:[] for m in METRICS}
    for ss in pools:
        r=evaluate(rng.random(len(pools[ss])), pools[ss])
        for m in METRICS: vals[m].append(r[m])
    for m in METRICS: draws[m].append(np.mean(vals[m]))
print(f"\n{'지표':>4} {'설명':<24} {'코사인':>9} {'무작위평균':>10} {'σ(잡음)':>9} {'SNR':>7}")
NAME={'A':'상위3 최고등급 평균(현행)','B':'상위5 최고등급 평균','C':'상위10 최고등급 평균',
      'D':'NDCG@5','E':'상위3 적합_전체 통과율%','F':'스피어만 순위상관(24건)'}
snr={}
for m in METRICS:
    a=np.array(draws[m]); mu,sd=a.mean(),a.std(ddof=1)
    s=(cosv[m]-mu)/sd if sd>0 else np.nan
    snr[m]=s
    print(f"{m:>4} {NAME[m]:<24} {cosv[m]:>9.3f} {mu:>10.3f} {sd:>9.4f} {s:>7.2f}")
print("\nL3 건전성: " + " · ".join(f"{m} {'OK' if cosv[m]>np.mean(draws[m]) else '**실패**'}" for m in METRICS))
ORDER=['A','B','E','D','C','F']
best=max(METRICS,key=lambda m:snr[m])
tied=[m for m in METRICS if snr[best]-snr[m]<=0.2]
pick=min(tied,key=lambda m:ORDER.index(m))
print(f"\n최고 SNR: {best} ({snr[best]:.2f}) · 동률권(≤0.2): {tied} · **채택: {pick}** (단순성 우선순위 {ORDER})")
sd_pick=np.array(draws[pick]).std(ddof=1)
print(f"채택 지표 {pick} 의 새 효과 기준 = 2σ = {2*sd_pick:.3f}  ({NAME[pick]})")
sdA=np.array(draws['A']).std(ddof=1)
print(f"\nL2: h63·h64 가 쓴 +0.30 기준은 지표 A 의 {0.30/sdA:.2f}σ 였다 (σ={sdA:.4f})")
print(f"L1: 현행 A 가 최고 SNR 인가? {'예 → 빗나감' if best=='A' else '아니오 → 적중'}")
json.dump({'cos':cosv,'sigma':{m:float(np.array(draws[m]).std(ddof=1)) for m in METRICS},
           'snr':{m:float(snr[m]) for m in METRICS},'pick':pick},
          open(f"{SP}/h65_result.json","w"), ensure_ascii=False, indent=1)
