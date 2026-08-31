"""R-1 2단계: 리랭커 전용 프로세스. 랭커·임베딩을 절대 로드하지 않는다.

  python crossdomain/r1_stage2.py <split> <variant> <n_profiles> [platforms]

필요한 id 의 semantic_text 만 parquet 에서 뽑아 쓰고(전체 데이터셋을 메모리에 두지 않는다),
모델은 그 뒤에 올린다. 결과는 r1_reranked_{split}_{variant}.json 에 이어 쓴다(중단·재개 가능).
"""
import json, os, sys, time, gc
os.environ.setdefault("OMP_NUM_THREADS","12"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
import numpy as np, pandas as pd

split=sys.argv[1]; variant=sys.argv[2]; nprof=int(sys.argv[3])
plats=(sys.argv[4].split(",") if len(sys.argv)>4 else ["steam","tmdb"])
TOP=json.load(open("crossdomain/r1_top100.json"))
keys=[k for k in TOP if k.split("|")[0]==split and k.split("|")[1] in plats]
keys.sort(); pids=sorted({k.split("|")[2] for k in keys})[:nprof]
keys=[k for k in keys if k.split("|")[2] in pids]
outf=f"crossdomain/r1_reranked_{split}_{variant}.json"
done=json.load(open(outf)) if os.path.exists(outf) else {}
keys=[k for k in keys if k not in done]
print(f"{split}/{variant}: 프로필 {len(pids)} · 남은 작업 {len(keys)}", flush=True)

need={"steam":set(),"tmdb":set()}
for k in keys:
    pl=k.split("|")[1]; e=TOP[k]; need[pl]|={int(x) for x in e["seeds"]}|{int(x) for x in e["top100"]}
TXT={}
if need["steam"]:
    d=pd.read_parquet("steam/artifacts/tags_full/dataset.parquet", columns=["steam_appid","semantic_text"])
    d=d[d["steam_appid"].astype(int).isin(need["steam"])]
    TXT["steam"]={int(a):str(t)[:1000] for a,t in zip(d["steam_appid"],d["semantic_text"])}; del d
if need["tmdb"]:
    idx=pd.read_parquet("tmdb/artifacts/tmdb_v1/corpus_index.parquet", columns=["item_id","embedding_row"]).sort_values("embedding_row")
    ds=pd.read_parquet("tmdb/artifacts/tmdb_v1/dataset.parquet", columns=["item_id","semantic_text"]).set_index("item_id")
    ds=ds.loc[idx["item_id"].to_numpy()].reset_index(drop=True)
    TXT["tmdb"]={r:str(ds["semantic_text"].iloc[r])[:1000] for r in need["tmdb"] if r<len(ds)}; del ds, idx
gc.collect()
print(f"텍스트 {sum(len(v) for v in TXT.values())}건 적재", flush=True)

from sentence_transformers import CrossEncoder
ce=CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
import resource
print(f"모델 로드 후 RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB", flush=True)

t0=time.time(); npair=0
for n,k in enumerate(keys,1):
    pl=k.split("|")[1]; e=TOP[k]; T=TXT[pl]
    cands=[int(c) for c in e["top100"]]; docs=[T.get(c,"") for c in cands]
    seeds=[int(s) for s in e["seeds"]]
    if variant=="a":
        q=" / ".join(T.get(s,"")[:300] for s in seeds)
        sc=np.array(ce.predict([(q,d) for d in docs], batch_size=32, show_progress_bar=False)); npair+=len(docs)
    else:
        S=np.full((len(seeds),len(cands)),-1e9,dtype=np.float32)
        for i,s in enumerate(seeds):
            S[i]=ce.predict([(T.get(s,""),d) for d in docs], batch_size=32, show_progress_bar=False); npair+=len(docs)
        sc=S.max(axis=0)
    order=np.argsort(-sc)
    done[k]=[cands[i] for i in order]
    json.dump(done,open(outf,"w"))
    el=time.time()-t0
    print(f"  {n}/{len(keys)} {k} · {el/n:.1f}s/작업 · {npair/el:.2f}쌍/s · 남은 {(len(keys)-n)*el/n/60:.0f}분", flush=True)
print("완료", outf)
