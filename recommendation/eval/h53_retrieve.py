"""h53 재검색 — 혼합/단일 프로필 top2_mean 질의. 등록 md5 2feffc400bd886febe1fdd15ef7ba3d3."""
import json, math, collections, sys
import numpy as np, pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L as L50
from h51_labels import L as L51
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
LAM=0.35
ADULT_GENRES={"신체 노출","선정적 콘텐츠"}; ADULT_DESC={3,4}

# ---------- 코퍼스 ----------
wd=pd.read_parquet(f"{WN}/dataset.parquet", columns=["item_id","name","genres","synopsis"])
wi=pd.read_parquet(f"{WN}/corpus_index.parquet").sort_values("embedding_row")
wd=wd.set_index("item_id").loc[wi["item_id"].to_numpy()].reset_index()
sd=pd.read_parquet(f"{STEAM}/dataset.parquet", columns=["steam_appid","name","genres","content_descriptorids","semantic_text"])
si=pd.read_parquet(f"{STEAM}/corpus_index.parquet").sort_values("embedding_row")
sd=sd.set_index("steam_appid").loc[si["steam_appid"].to_numpy()].reset_index()
td=pd.read_parquet(f"{TMDB}/dataset.parquet", columns=["item_id","name","media","overview","overview_len","adult"])
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
for d in KEEP: print(f"  {d:6s} 코퍼스 {len(DF[d]):,} → 필터 후 {int(KEEP[d].sum()):,}")

# ---------- 기존 라벨 풀 + 시드 ----------
k50=json.load(open(f'{SP}/h50_key.json')); k51=json.load(open(f'{SP}/h51_key.json'))
d50=json.load(open(f'{SP}/h50_seeds20.json'))
seed_name={s:x['wn'] for s,x in enumerate(d50)}
seed_tag={k['seed']:L50[k['id']] for k in k50 if k['kind']=='seed'}
labeled={}; owner=collections.defaultdict(set)     # (dom,name)->tags ; (dom,name)->{seeds}
for k in k50:
    if k['kind']=='cand': labeled[(k['dom'],k['name'])]=L50[k['id']]; owner[(k['dom'],k['name'])].add(k['seed'])
for k in k51:
    labeled[('wn',k['name'])]=L51[k['id']]; owner[('wn',k['name'])].add(k['seed'])
for s,n in seed_name.items(): labeled[('wn',n)]=seed_tag[s]

def unit(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v
def pick(seed_r, dom, s_ex, k):
    q=unit(EMB['wn'][seed_r]); out=[]; seen=set()
    cands=[(nm,r) for (dm,nm),tg in labeled.items() if dm==dom and s_ex not in owner[(dm,nm)]
           and (r:=row[dom].get(str(nm))) is not None]
    for nm,r in sorted(cands, key=lambda t:-float(q @ unit(EMB[dom][t[1]]))):
        if nm in seen: continue
        seen.add(nm); out.append((dom,nm,r))
        if len(out)==k: break
    return out

srow={s: row['wn'][seed_name[s]] for s in seed_name}
prof={}
for s in sorted(seed_tag):
    x=pick(srow[s],'steam',s,1)[0]; y=pick(srow[s],'tmdb',s,1)[0]; ab=pick(srow[s],'wn',s,2)
    prof[s]={'M':[('wn',seed_name[s],srow[s]), x, y],
             'S':[('wn',seed_name[s],srow[s])]+ab}

# ---------- top2_mean 질의로 재검색 ----------
def top2_mean_query(items, dom):
    """확정 파이프라인: 코퍼스 각 항목에 대해 프로필 항목별 유사도 상위 2개의 평균."""
    V=np.stack([unit(EMB[d][r]) for d,_,r in items])          # (3, dim)
    E=EMB[dom] - LAM*CEN[dom][None,:]
    S=V @ E.T                                                  # (3, N)
    S.sort(axis=0)
    return S[-2:,:].mean(axis=0)                               # 상위 2개 평균

out={}
for s in sorted(seed_tag):
    out[s]={}
    for arm in ('M','S'):
        out[s][arm]={}
        pn={n for _,n,_ in prof[s][arm]}
        for dom in ('wn','steam','tmdb'):
            sc=top2_mean_query(prof[s][arm], dom).copy()
            sc[~KEEP[dom]]=-1e9
            for _,n,_ in prof[s][arm]:                          # 프로필 항목 자신 제외
                if (r:=row[dom].get(str(n))) is not None: sc[r]=-1e9
            top=np.argsort(-sc)[:10]
            out[s][arm][dom]=[{"r":int(t), "n":str(DF[dom].loc[int(t),"name"]),
                               "d": str(DF[dom].loc[int(t),"semantic_text"]) if dom=='steam'
                                    else str(DF[dom].loc[int(t),"overview"] if dom=='tmdb'
                                             else DF[dom].loc[int(t),"synopsis"]),
                               "sim": float(sc[t])} for t in top]
json.dump({'profiles':{str(s):{a:[[d,n] for d,n,_ in prof[s][a]] for a in ('M','S')} for s in prof},
           'pools':{str(s):{a:{d:[{k:v for k,v in c.items()} for c in out[s][a][d]]
                               for d in ('wn','steam','tmdb')} for a in ('M','S')} for s in out}},
          open(f"{SP}/h53_pools.json","w"), ensure_ascii=False, indent=1)

# ---------- 신규 라벨 필요 건수 ----------
need=set(); tot=0; known=0
for s in out:
    for a in ('M','S'):
        for d in ('wn','steam','tmdb'):
            for c in out[s][a][d]:
                tot+=1
                if (d,c['n']) in labeled: known+=1
                else: need.add((d,c['n']))
print(f"\n총 후보 슬롯 {tot} (20시드 × 2갈래 × 3도메인 × 10)")
print(f"  기존 라벨 재사용 {known} ({100*known/tot:.1f}%) · 신규 라벨 필요 고유 항목 {len(need)}건")
ov=[]
for s in out:
    for d in ('wn','steam','tmdb'):
        A={c['n'] for c in out[s]['M'][d]}; B={c['n'] for c in out[s]['S'][d]}
        ov.append(len(A&B)/10)
print(f"  M/S 풀 겹침 평균 {100*sum(ov)/len(ov):.1f}%")
print("\n[프로필 표본]")
for s in (0,5,11,17):
    print(f"  시드{s:2d} M: " + " | ".join(f"{d}:{n}" for d,n,_ in prof[s]['M']))
    print(f"       M 검색: wn " + " / ".join(c['n'] for c in out[s]['M']['wn'][:2])
          + " ‖ st " + " / ".join(c['n'] for c in out[s]['M']['steam'][:2])
          + " ‖ tm " + " / ".join(c['n'] for c in out[s]['M']['tmdb'][:2]))
