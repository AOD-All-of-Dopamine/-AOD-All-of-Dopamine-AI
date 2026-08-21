"""h57 검색(시드 10~19) — 분업(max) vs 합성(min) 질의. 등록 md5 70dffd0f357e7304530c539c477a53b3."""
import json, collections, sys
import numpy as np, pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; DOMS=('wn','steam','tmdb'); SEEDS=list(range(10,20)); K=3; TOP20=20
ADULT_GENRES={"신체 노출","선정적 콘텐츠"}; ADULT_DESC={3,4}

wd=pd.read_parquet(f"{WN}/dataset.parquet", columns=["item_id","name","synopsis"])
wi=pd.read_parquet(f"{WN}/corpus_index.parquet").sort_values("embedding_row")
wd=wd.set_index("item_id").loc[wi["item_id"].to_numpy()].reset_index()
sd=pd.read_parquet(f"{STEAM}/dataset.parquet", columns=["steam_appid","name","genres","content_descriptorids","semantic_text"])
si=pd.read_parquet(f"{STEAM}/corpus_index.parquet").sort_values("embedding_row")
sd=sd.set_index("steam_appid").loc[si["steam_appid"].to_numpy()].reset_index()
td=pd.read_parquet(f"{TMDB}/dataset.parquet", columns=["item_id","name","overview","overview_len","adult"])
ti=pd.read_parquet(f"{TMDB}/corpus_index.parquet").sort_values("embedding_row")
td=td.set_index("item_id").loc[ti["item_id"].to_numpy()].reset_index()
def is_adult(g,d):
    if g is not None and set(map(str,g)) & ADULT_GENRES: return True
    if d is not None and {int(x) for x in d} & ADULT_DESC: return True
    return False
KEEP={'wn': (wd["synopsis"].str.len()>=100).to_numpy(),
      'steam': ~np.array([is_adult(g,d) for g,d in zip(sd["genres"], sd["content_descriptorids"])]),
      'tmdb': (td["overview"].fillna("").str.contains(r"[가-힣]").to_numpy()
               & (td["overview_len"]>=100).to_numpy() & (~td["adult"].fillna(False).to_numpy()))}
DF={'wn':wd,'steam':sd,'tmdb':td}
EMB={d: np.asarray(np.load(p+"/corpus_embeddings.npy", mmap_mode="r"), dtype=np.float32)
     for d,p in (('wn',WN),('steam',STEAM),('tmdb',TMDB))}
CEN={d: EMB[d].mean(axis=0) for d in EMB}
row={d:{str(n):i for i,n in enumerate(DF[d]["name"])} for d in DF}
TXT={'wn':'synopsis','steam':'semantic_text','tmdb':'overview'}

profs=json.load(open(f'{SP}/h53_pools.json'))['profiles']
def unit(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v

out={}; stats=collections.Counter()
for s in SEEDS:
    ss=str(s); items=profs[ss]['M']            # [[dom,name],...]
    V=np.stack([unit(EMB[d][row[d][str(n)]]) for d,n in items])   # (3,dim)
    out[ss]={'profile': items, 'D':{}, 'C':{}, 'solo_top20':{}}
    for dom in DOMS:
        E=EMB[dom] - LAM*CEN[dom][None,:]
        S=V @ E.T                                                  # (3,N)
        bad=~KEEP[dom]
        for d,n in items:                                          # 프로필 항목 자신 제외
            r=row[dom].get(str(n))
            if r is not None: bad[r]=True
        S[:, bad]=-1e9
        solo={}
        for i,(d,n) in enumerate(items):
            solo[f"{d}:{n}"]=set(np.argsort(-S[i])[:TOP20].tolist())
        union20=set().union(*solo.values())
        out[ss]['solo_top20'][dom]=[int(x) for x in union20]
        for arm, agg in (('D', S.max(axis=0)), ('C', S.min(axis=0))):
            sc=agg.copy(); sc[bad]=-1e9
            top=np.argsort(-sc)[:K]
            out[ss][arm][dom]=[{"r":int(t), "n":str(DF[dom].loc[int(t),"name"]),
                                "d":str(DF[dom].loc[int(t),TXT[dom]]),
                                "sc":float(sc[t]), "solo":int(t) in union20} for t in top]
            stats[f"{arm}_{dom}"]+=K
            stats[f"{arm}_{dom}_solo"]+=sum(int(t) in union20 for t in top)
json.dump(out, open(f"{SP}/h57_pools.json","w"), ensure_ascii=False, indent=1)

tot=len(SEEDS)*len(DOMS)*K
for arm in ('D','C'):
    so=sum(stats[f"{arm}_{d}_solo"] for d in DOMS)
    print(f"  {arm} 갈래 후보 {tot} · 단일항목 top-20 안 {so} ({100*so/tot:.1f}%) · **밖 {100*(tot-so)/tot:.1f}%**")
ov=[]
for ss in out:
    for d in DOMS:
        A={c['n'] for c in out[ss]['D'][d]}; B={c['n'] for c in out[ss]['C'][d]}
        ov.append(len(A&B)/K)
print(f"  D/C 후보 겹침 평균 {100*sum(ov)/len(ov):.1f}%")
print("\n[표본] 시드 0 프로필:", " | ".join(f"{d}:{n}" for d,n in out['0']['profile']))
for arm in ('D','C'):
    print(f"  {arm}: " + " ‖ ".join(f"{d} " + " / ".join(c['n'] for c in out['0'][arm][d]) for d in DOMS))
