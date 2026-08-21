"""h62 깊은 검색 — h60 프로필 그대로, 도메인당 top-30. 등록 23cbef872ad79c2b46253e185d33c405."""
import json, collections, random
import numpy as np, pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; DOMS=('wn','steam','tmdb'); DEEP=30; RNG=20260829
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
NAMES={d:[str(x) for x in DF[d]["name"]] for d in DF}
rows_by_name={d: collections.defaultdict(list) for d in DF}
for d in DF:
    for i,n in enumerate(NAMES[d]): rows_by_name[d][n].append(i)
row={d:{n:v[0] for n,v in rows_by_name[d].items()} for d in DF}
TXT={'wn':'synopsis','steam':'semantic_text','tmdb':'overview'}
def unit(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v
P60=json.load(open(f'{SP}/h60_pools.json'))
rng=random.Random(RNG)
out={}; stat=collections.Counter()
for ss,blk in P60.items():
    items=blk['profile']
    V=np.stack([unit(EMB[d][row[d][str(n)]]) for d,n in items])
    rec={'seed':blk['seed'],'profile':items,'deep':{},'sample':{}}
    for dom in DOMS:
        E=EMB[dom] - LAM*CEN[dom][None,:]
        sc=(V @ E.T).max(axis=0)
        bad=~KEEP[dom].copy()
        for d,n in items:
            if d==dom:
                for i in rows_by_name[dom].get(str(n), []): bad[i]=True
        sc[bad]=-1e9
        top=np.argsort(-sc)[:DEEP]
        deep=[{"rank":r+1,"n":NAMES[dom][int(t)],"sc":float(sc[t])} for r,t in enumerate(top)]
        rec['deep'][dom]=deep
        # h60 상위 3개와 일치하는지 확인
        old=[c['n'] for c in blk['D'][dom]]
        stat['top3_match' if [c['n'] for c in deep[:3]]==old else 'top3_MISMATCH']+=1
        # 층화 추출: 4~10 에서 2 · 11~20 에서 2 · 21~30 에서 1
        picks=[]
        for lo,hi,k in ((4,10,2),(11,20,2),(21,30,1)):
            band=[c for c in deep if lo<=c['rank']<=hi]
            picks += rng.sample(band, min(k,len(band)))
        rec['sample'][dom]=[{"rank":c['rank'],"n":c['n'],"sc":c['sc'],
                             "d":str(DF[dom].loc[row[dom][c['n']],TXT[dom]])} for c in picks]
        stat['sampled']+=len(picks)
    out[ss]=rec
json.dump(out, open(f"{SP}/h62_pools.json","w"), ensure_ascii=False, indent=1)
print(f"[검증] h60 top-3 재현: {dict(stat)}")
uniq={(d,c['n']) for r in out.values() for d in DOMS for c in r['sample'][d]}
print(f"[표본] 시드 {len(out)} · 신규 후보 슬롯 {stat['sampled']} · 서로 다른 항목 {len(uniq)}")
bands=collections.Counter()
for r in out.values():
    for d in DOMS:
        for c in r['sample'][d]:
            bands['4-10' if c['rank']<=10 else ('11-20' if c['rank']<=20 else '21-30')]+=1
print(f"[구간별 표본] {dict(bands)}")
