"""R-1: bge-reranker-v2-m3 로 Steam·TMDB top-100 재정렬.
질의 = 시드 semantic_text, 문서 = 후보 semantic_text[:1000]. 점수는 리랭커 단독.
변형 (a) 시드 텍스트 이어붙여 질의 1개 / (b) 시드별 질의 → 후보 점수 = 시드별 최대.
후처리(시리즈 상한 등)는 이미 top-100 생성 시 적용돼 있으므로 재정렬만 한다.
"""
import numpy as np, pandas as pd
from pathlib import Path
AOD=Path("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
_CE=None
def ce():
    global _CE
    if _CE is None:
        from sentence_transformers import CrossEncoder
        _CE=CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cpu")
    return _CE

_TEXT={}
def texts(plat):
    if plat not in _TEXT:
        if plat=="steam":
            d=pd.read_parquet(AOD/"steam/artifacts/tags_full/dataset.parquet")
            _TEXT[plat]=dict(zip(d["steam_appid"].astype(int), d["semantic_text"]))
        elif plat=="tmdb":
            idx=pd.read_parquet(AOD/"tmdb/artifacts/tmdb_v1/corpus_index.parquet").sort_values("embedding_row")
            d=pd.read_parquet(AOD/"tmdb/artifacts/tmdb_v1/dataset.parquet").set_index("item_id")
            d=d.loc[idx["item_id"].to_numpy()].reset_index()
            _TEXT[plat]=dict(zip(range(len(d)), d["semantic_text"]))   # row 인덱스 키
    return _TEXT[plat]

def rerank(plat, seed_ids, cand_ids, variant="b", batch_size=16):
    T=texts(plat)
    docs=[str(T.get(int(c),""))[:1000] for c in cand_ids]
    if variant=="a":
        q=" / ".join(str(T.get(int(s),""))[:300] for s in seed_ids)
        scores=np.array(ce().predict([(q,d) for d in docs], batch_size=batch_size, show_progress_bar=False))
    else:
        S=np.full((len(seed_ids),len(cand_ids)),-1e9,dtype=np.float32)
        for i,s in enumerate(seed_ids):
            q=str(T.get(int(s),""))[:1000]
            S[i]=ce().predict([(q,d) for d in docs], batch_size=batch_size, show_progress_bar=False)
        scores=S.max(axis=0)
    order=np.argsort(-scores)
    return [int(cand_ids[i]) for i in order], scores[order]
