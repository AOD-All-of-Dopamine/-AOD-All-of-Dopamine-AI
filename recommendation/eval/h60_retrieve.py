"""h60 검색 — 신규 시드 10 · 앵커는 라벨 제한 규칙(A). 등록 f6d8d1aaee88fb364a8c7f3d75dc01bd."""
import json, collections
import numpy as np, pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; DOMS=('wn','steam','tmdb'); K=3; NSEED=10; RNG=20260825
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
UN={d: EMB[d]/np.linalg.norm(EMB[d],axis=1,keepdims=True) for d in EMB}
NAMES={d:[str(x) for x in DF[d]["name"]] for d in DF}
rows_by_name={d: collections.defaultdict(list) for d in DF}
for d in DF:
    for i,n in enumerate(NAMES[d]): rows_by_name[d][n].append(i)
row={d:{n:v[0] for n,v in rows_by_name[d].items()} for d in DF}
TXT={'wn':'synopsis','steam':'semantic_text','tmdb':'overview'}
def unit(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v
allow={d:set() for d in DOMS}
for k in json.load(open(f'{SP}/h50_key.json')): allow[k['dom'] or 'wn'].add(k['name'])
for k in json.load(open(f'{SP}/h51_key.json')): allow['wn'].add(k['name'])
used=set(json.load(open(f'{SP}/used_seeds.json'))) | set(json.load(open(f'{SP}/used_seeds_h50.json')))
used |= {x['wn'] for x in json.load(open(f'{SP}/h50_seeds20.json'))}
used |= {blk['profile'][0][1] for blk in json.load(open(f'{SP}/h58_pools.json')).values()}
elig=np.array([i for i in range(len(wd)) if KEEP['wn'][i] and NAMES['wn'][i] not in used])
rng=np.random.default_rng(RNG)
seed_rows=sorted(rng.choice(elig, size=NSEED, replace=False).tolist())
print(f"[시드] 후보 {len(elig)}건에서 {NSEED}개 추출 (제외 {len(used)}건)")
MASK={}
for dom in ('steam','tmdb'):
    m=np.zeros(len(NAMES[dom]), dtype=bool)
    for n in allow[dom]:
        for i in rows_by_name[dom].get(n, []): m[i]=True
    MASK[dom]=m & KEEP[dom]
out={}
for sr in seed_rows:
    q=UN['wn'][sr]; items=[['wn', NAMES['wn'][sr]]]
    for dom in ('steam','tmdb'):
        s=UN[dom] @ q; s[~MASK[dom]]=-1e9
        items.append([dom, NAMES[dom][int(np.argmax(s))]])
    V=np.stack([unit(EMB[d][row[d][n]]) for d,n in items])
    D={}
    for dom in DOMS:
        E=EMB[dom] - LAM*CEN[dom][None,:]
        sc=(V @ E.T).max(axis=0)
        bad=~KEEP[dom].copy()
        for d,n in items:
            if d==dom:
                for i in rows_by_name[dom].get(n, []): bad[i]=True
        sc[bad]=-1e9
        top=np.argsort(-sc)[:K]
        D[dom]=[{"r":int(t),"n":NAMES[dom][int(t)],
                 "d":str(DF[dom].loc[int(t),TXT[dom]]),"sc":float(sc[t])} for t in top]
    out[str(sr)]={'seed':NAMES['wn'][sr],'profile':items,'D':D,
                  'anchor_cos':[float(q @ UN[d][row[d][n]]) for d,n in items]}
json.dump(out, open(f"{SP}/h60_pools.json","w"), ensure_ascii=False, indent=1)
uniq={(d,c['n']) for blk in out.values() for d in DOMS for c in blk['D'][d]}
uniq |= {(d,n) for blk in out.values() for d,n in blk['profile']}
print(f"[풀] 시드 {len(out)} · 후보 슬롯 {len(out)*9} · 라벨 대상 서로 다른 항목 {len(uniq)}")
print("\n[프로필] (앵커 = 라벨 제한 규칙)")
for blk in out.values():
    print("  " + " | ".join(f"{d}:{n[:24]}" for d,n in blk['profile']))
