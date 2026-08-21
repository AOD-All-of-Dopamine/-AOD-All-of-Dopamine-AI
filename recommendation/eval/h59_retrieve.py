"""h59 검색 — 프로필 앵커 규칙 A(라벨 제한) vs B(전체 코퍼스). 등록 6243c4fc23b7d01c910ce10a76808d24."""
import json, collections
import numpy as np, pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35; DOMS=('wn','steam','tmdb'); K=3
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
UN={d: EMB[d]/np.linalg.norm(EMB[d],axis=1,keepdims=True) for d in EMB}

# 규칙 A 가 앵커를 고를 수 있는 집합 (h53 재현)
allow={d:set() for d in DOMS}
for k in json.load(open(f'{SP}/h50_key.json')): allow[k['dom'] or 'wn'].add(k['name'])
for k in json.load(open(f'{SP}/h51_key.json')): allow['wn'].add(k['name'])

B=json.load(open(f'{SP}/h58_pools.json'))
seeds=[(ss, blk['profile'][0][1], blk['profile']) for ss,blk in B.items()]

def anchors_A(sr):
    q=UN['wn'][sr]; out=[['wn', str(wd.loc[sr,"name"])]]
    for dom in ('steam','tmdb'):
        mask=np.zeros(len(NAMES[dom]), dtype=bool)
        for n in allow[dom]:
            for i in rows_by_name[dom].get(n, []): mask[i]=True
        mask &= KEEP[dom]
        s=UN[dom] @ q; s[~mask]=-1e9
        out.append([dom, NAMES[dom][int(np.argmax(s))]])
    return out

def retrieve(items):
    V=np.stack([unit(EMB[d][row[d][str(n)]]) for d,n in items])
    res={}
    for dom in DOMS:
        E=EMB[dom] - LAM*CEN[dom][None,:]
        S=V @ E.T
        bad=~KEEP[dom].copy()
        for d,n in items:                      # D-15 수정: 이름 기준 제외
            if d==dom:
                for i in rows_by_name[dom].get(str(n), []): bad[i]=True
        sc=S.max(axis=0); sc[bad]=-1e9
        top=np.argsort(-sc)[:K]
        res[dom]=[{"r":int(t),"n":NAMES[dom][int(t)],
                   "d":str(DF[dom].loc[int(t),TXT[dom]]),"sc":float(sc[t])} for t in top]
    return res

out={}
for ss, seedname, profB in seeds:
    sr=row['wn'][seedname]
    profA=anchors_A(sr)
    out[ss]={'seed':seedname,
             'A':{'profile':profA,'D':retrieve(profA),
                  'anchor_cos':[float(UN['wn'][sr] @ UN[d][row[d][n]]) for d,n in profA]},
             'B':{'profile':[list(x) for x in profB],'D':retrieve([list(x) for x in profB]),
                  'anchor_cos':[float(UN['wn'][sr] @ UN[d][row[d][n]]) for d,n in profB]}}
json.dump(out, open(f"{SP}/h59_pools.json","w"), ensure_ascii=False, indent=1)

print("[프로필 대조]  A = 라벨 제한 · B = 전체 코퍼스")
for ss,blk in out.items():
    print(f"  시드 {blk['seed'][:16]:18s}")
    for r in ('A','B'):
        p=blk[r]['profile']; c=blk[r]['anchor_cos']
        print(f"    {r}: " + " | ".join(f"{d}:{n[:22]}({cc:.2f})" for (d,n),cc in zip(p,c)))
ov=[]
for ss,blk in out.items():
    for d in DOMS:
        a={c['n'] for c in blk['A']['D'][d]}; b={c['n'] for c in blk['B']['D'][d]}
        ov.append(len(a&b)/K)
print(f"\n[E3 공허 가드] 두 규칙 후보 겹침 평균 {100*sum(ov)/len(ov):.1f}% (기준 ≤70%)")
import statistics as st
for r in ('A','B'):
    cs=[c for blk in out.values() for c in blk[r]['anchor_cos'][1:]]
    print(f"[E5] 규칙 {r} 시드–앵커 cos 평균 {st.mean(cs):.3f} (최소 {min(cs):.3f} · 최대 {max(cs):.3f})")
