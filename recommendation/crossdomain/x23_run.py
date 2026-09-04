"""X-23 실행 — 각색 쌍 양방향 known-item 검색. 사전등록 b1dd90e8… 대로만 잰다."""
import pandas as pd, numpy as np, json

D={"steam":"steam/artifacts/tags_full","tmdb":"tmdb/artifacts/tmdb_v1",
   "wn":"/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"}
POP={"steam":("dataset.parquet","steam_appid","recommendations_total"),
     "tmdb":("dataset.parquet","item_id","vote_count"),
     "wn":("dataset.parquet","item_id","interest_count")}
E,IDX,ROW,POPRANK={},{},{},{}
for p,d in D.items():
    e=np.load(f"{d}/corpus_embeddings.npy").astype(np.float32)
    e/= (np.linalg.norm(e,axis=1,keepdims=True)+1e-9)          # L2 정규화
    E[p]=e
    ci=pd.read_parquet(f"{d}/corpus_index.parquet")
    key="steam_appid" if p=="steam" else "item_id"
    ids=ci.sort_values("embedding_row")[key].tolist()
    IDX[p]=ids; ROW[p]={v:i for i,v in enumerate(ids)}
    f,k,c=POP[p]; ds=pd.read_parquet(f"{d}/{f}",columns=[k,c])
    top=ds.sort_values(c,ascending=False).head(50)[k].tolist()
    POPRANK[p]=set(top)
    print(f"{p}: {e.shape} · 인기 top-50 준비")
CENT={p: (E[p].mean(axis=0)/ (np.linalg.norm(E[p].mean(axis=0))+1e-9)) for p in E}

pairs=json.load(open("crossdomain/x23_pairs.json"))
Q=[]
for p in pairs:
    src="steam" if p["kind"]=="game" else "wn"
    a=p["steam"] if p["kind"]=="game" else p["wn"]
    Q.append(dict(pair=p["name"],kind=p["kind"],dir="원작→각색",sp=src,si=a,tp="tmdb",ti=p["tmdb"]))
    Q.append(dict(pair=p["name"],kind=p["kind"],dir="각색→원작",sp="tmdb",si=p["tmdb"],tp=src,ti=a))

def rank_of(sp,si,tp,ti,hub):
    v=E[sp][ROW[sp][si]].copy()
    if hub: v=v-0.35*CENT[sp]; v/= (np.linalg.norm(v)+1e-9)
    s=E[tp]@v
    if sp==tp: s[ROW[tp][si]]=-1e9
    tgt=ROW[tp][ti]
    return int((s>s[tgt]).sum())+1                              # 1-기반 순위

out=[]
for q in Q:
    r=dict(q)
    r["rank_pure"]=rank_of(q["sp"],q["si"],q["tp"],q["ti"],False)
    r["rank_hub"] =rank_of(q["sp"],q["si"],q["tp"],q["ti"],True)
    r["pop50"]=q["ti"] in POPRANK[q["tp"]]
    r["N"]=len(IDX[q["tp"]])
    out.append(r)
json.dump(out,open("crossdomain/x23_ranks.json","w"),ensure_ascii=False,indent=1)
print(f"\n질의 {len(out)}건 완료 → crossdomain/x23_ranks.json")
