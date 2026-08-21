"""h67 검색 — R-max / R-min / R-cent. 등록 96fd17ea84e4d04b635f0a9785af58a1."""
import json, collections
import numpy as np, pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; DOMS=('wn','steam','tmdb'); K=2
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
EMB={d:np.asarray(np.load(p+"/corpus_embeddings.npy",mmap_mode="r"),dtype=np.float32)
     for d,p in (('wn',WN),('steam',STEAM),('tmdb',TMDB))}
CEN={d:EMB[d].mean(axis=0) for d in EMB}
NAMES={d:[str(x) for x in DF[d]["name"]] for d in DF}
rows_by_name={d:collections.defaultdict(list) for d in DF}
for d in DF:
    for i,n in enumerate(NAMES[d]): rows_by_name[d][n].append(i)
row={d:{n:v[0] for n,v in rows_by_name[d].items()} for d in DF}
TXT={'wn':'synopsis','steam':'semantic_text','tmdb':'overview'}
def uv(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v
P=json.load(open(f"{SP}/h62_pools.json"))
out={}; stat=collections.Counter()
for ss,blk in P.items():
    items=[(d,n) for d,n in blk['profile']]
    V=np.stack([uv(EMB[d][row[d][str(n)]]) for d,n in items])          # 3 x 1024
    cent=uv(V.mean(axis=0))
    rec={'profile':blk['profile'],'R-max':{},'R-min':{},'R-cent':{}}
    for dom in DOMS:
        # h62 파이프라인 관례 그대로 — 보정 후 재정규화하지 않는다
        E = EMB[dom] - LAM*CEN[dom][None,:]
        S = V @ E.T                                                     # 3 x N
        bad=~KEEP[dom].copy()
        for d,n in items:
            if d==dom:
                for i in rows_by_name[dom].get(str(n), []): bad[i]=True  # D-15
        for tag,sc in (('R-max',S.max(axis=0)),('R-min',S.min(axis=0)),('R-cent',cent @ E.T)):
            v=sc.copy(); v[bad]=-1e9
            top=np.argsort(-v)[:K]
            rec[tag][dom]=[{"rank":r+1,"n":NAMES[dom][int(t)],"sc":float(v[t]),
                            "d":str(DF[dom][TXT[dom]].iloc[int(t)])[:1200]} for r,t in enumerate(top)]
    out[ss]=rec
    # h62 top-3 재현 확인 (R-max 의 상위 2 는 h62 deep 의 1~2 와 같아야 한다)
    for dom in DOMS:
        old=[c['n'] for c in blk['deep'][dom][:K]]
        stat['R-max 재현' if [c['n'] for c in rec['R-max'][dom]]==old else 'R-max 불일치']+=1
json.dump(out, open(f"{SP}/h67_pools.json","w"), ensure_ascii=False)
print("[검증]", dict(stat))
uniq={t:{(d,c['n']) for ss in out for d in DOMS for c in out[ss][t][d]} for t in ('R-max','R-min','R-cent')}
for t in uniq: print(f"  {t:>7}: 서로 다른 후보 {len(uniq[t])}")
print(f"  R-min ∩ R-max = {len(uniq['R-min'] & uniq['R-max'])} · R-cent ∩ R-max = {len(uniq['R-cent'] & uniq['R-max'])}"
      f" · R-min ∩ R-cent = {len(uniq['R-min'] & uniq['R-cent'])}")
