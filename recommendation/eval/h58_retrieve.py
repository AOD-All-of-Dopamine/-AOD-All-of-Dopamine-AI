"""h58 검색 — 신규 시드 10개 + 도메인 게이팅용 풀. 등록 md5 d1806993ed147edd0b1d1bf1db363473."""
import json, collections
import numpy as np, pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; DOMS=('wn','steam','tmdb'); K=3; NSEED=10; RNG=20260820
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
def unit(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v

# ---- 1. 신규 시드 10개: 난수 20260820, 기존 사용 시드 전부 제외 ----
used=set(json.load(open(f'{SP}/used_seeds.json'))) | set(json.load(open(f'{SP}/used_seeds_h50.json')))
used |= {x['wn'] for x in json.load(open(f'{SP}/h50_seeds20.json'))}
elig=np.array([i for i in range(len(wd)) if KEEP['wn'][i] and str(wd.loc[i,"name"]) not in used])
rng=np.random.default_rng(RNG)
seed_rows=sorted(rng.choice(elig, size=NSEED, replace=False).tolist())
print(f"[시드] 후보 {len(elig)}건에서 {NSEED}개 추출 (제외 {len(used)}건)")

# ---- 2. 프로필 M = 웹소설 시드 + 최근접 Steam 1 + 최근접 TMDB 1 (전체 코퍼스 대상) ----
prof={}
for sr in seed_rows:
    q=unit(EMB['wn'][sr]); items=[['wn', str(wd.loc[sr,"name"])]]
    for dom in ('steam','tmdb'):
        E=EMB[dom]/np.linalg.norm(EMB[dom],axis=1,keepdims=True)
        s=E @ q; s[~KEEP[dom]]=-1e9
        t=int(np.argmax(s)); items.append([dom, str(DF[dom].loc[t,"name"])])
    prof[str(sr)]=items

# ---- 3. 검색 D(max) / C(min) ----
out={}; stats=collections.Counter()
for ss,items in prof.items():
    V=np.stack([unit(EMB[d][row[d][str(n)]]) for d,n in items])
    out[ss]={'profile': items, 'D':{}, 'C':{}}
    for dom in DOMS:
        E=EMB[dom] - LAM*CEN[dom][None,:]
        S=V @ E.T
        bad=~KEEP[dom]
        for d,n in items:
            r=row[dom].get(str(n))
            if r is not None: bad[r]=True
        S[:, bad]=-1e9
        for arm, agg in (('D', S.max(axis=0)), ('C', S.min(axis=0))):
            sc=agg.copy(); sc[bad]=-1e9
            top=np.argsort(-sc)[:K]
            out[ss][arm][dom]=[{"r":int(t), "n":str(DF[dom].loc[int(t),"name"]),
                                "d":str(DF[dom].loc[int(t),TXT[dom]]),
                                "sc":float(sc[t])} for t in top]
            stats[f"{arm}_{dom}"]+=K
json.dump(out, open(f"{SP}/h58_pools.json","w"), ensure_ascii=False, indent=1)

uniq=set()
for ss,blk in out.items():
    for arm in ('D','C'):
        for d in DOMS:
            for c in blk[arm][d]: uniq.add((d,c['n']))
print(f"[풀] 시드 {len(out)} · 후보 슬롯 {sum(stats.values())} · 서로 다른 항목 {len(uniq)}")
print("\n[프로필]")
for ss,blk in out.items():
    print("  "+" | ".join(f"{d}:{n[:26]}" for d,n in blk['profile']))
