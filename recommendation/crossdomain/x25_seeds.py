"""X-25 시드 절제 — 같은 페르소나 v2 안에서 **시드만** 바꾼 4개 팔을 만든다.

X-24 는 근거 페르소나(0.662)와 v2(0.909)를 비교했는데 구성 방식까지 달라 교란이 있었다.
여기서는 페르소나·랭커·후처리를 전부 고정하고 시드 집합만 움직인다.

팔 (전부 기계적으로 결정, 관측 전 확정):
  k5      : 현행 페르소나 v2 그대로 (플랫폼당 5개)
  k1      : 플랫폼마다 1개 — 그 플랫폼 시드들의 중심에 **가장 가까운** 것 (대표 1개)
  k3_div  : 5개 중 3개 조합(C(5,3)=10) 가운데 내부 평균 유사도가 **최소** 인 조합
  k3_red  : 같은 조합 중 내부 평균 유사도가 **최대** 인 조합
시드가 3개인 플랫폼은 k3_div = k3_red (조합이 하나뿐). 시드가 없는 플랫폼은 그대로 없다.
"""
import json, itertools, numpy as np, pandas as pd
D={"steam":"steam/artifacts/tags_full","tmdb":"tmdb/artifacts/tmdb_v1",
   "wn":"/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"}
E={};ROW={}
for p,d in D.items():
    e=np.load(f"{d}/corpus_embeddings.npy").astype(np.float32); e/=(np.linalg.norm(e,axis=1,keepdims=True)+1e-9); E[p]=e
    ci=pd.read_parquet(f"{d}/corpus_index.parquet").sort_values("embedding_row")
    k="steam_appid" if p=="steam" else "item_id"
    ROW[p]={v:i for i,v in enumerate(ci[k].tolist())}
TID=pd.read_parquet("tmdb/artifacts/tmdb_v1/dataset.parquet",columns=["item_id"])["item_id"].tolist()
def vec(plat,x):
    if plat=="tmdb": return E["tmdb"][ROW["tmdb"][TID[int(x)]]]
    return E[plat][ROW[plat][int(x)]]
def ils(vs):
    M=np.array(vs); S=M@M.T; iu=np.triu_indices(len(vs),1); return float(S[iu].mean())

out=[]
for e in json.load(open("crossdomain/personas_v2.json")):
    arms={"k5":{}, "k1":{}, "k3_div":{}, "k3_red":{}}
    diag={}
    for plat,ids in e["seeds"].items():
        if not ids: continue
        ids=list(ids); V=[vec(plat,x) for x in ids]
        arms["k5"][plat]=ids
        c=np.mean(V,axis=0); c/= (np.linalg.norm(c)+1e-9)
        arms["k1"][plat]=[ids[int(np.argmax([float(v@c) for v in V]))]]
        if len(ids)>=3:
            combos=[(ils([V[i] for i in c3]),c3) for c3 in itertools.combinations(range(len(ids)),3)]
            lo=min(combos)[1]; hi=max(combos)[1]
            arms["k3_div"][plat]=[ids[i] for i in lo]; arms["k3_red"][plat]=[ids[i] for i in hi]
            diag[plat]=dict(div=round(min(combos)[0],3), red=round(max(combos)[0],3), all5=round(ils(V),3))
        else:
            arms["k3_div"][plat]=ids; arms["k3_red"][plat]=ids
    out.append(dict(pid=e["pid"], desc=e["desc"], arms=arms, ils=diag, names=e["names"]))
json.dump(out,open("crossdomain/x25_seeds.json","w"),ensure_ascii=False,indent=1)
dv=[d["div"] for p in out for d in p["ils"].values()]; rd=[d["red"] for p in out for d in p["ils"].values()]
a5=[d["all5"] for p in out for d in p["ils"].values()]
print(f"페르소나 {len(out)} · 플랫폼-프로필 {len(dv)}건")
print(f"시드 내부 평균 유사도:  k3_div {np.mean(dv):.3f} · k5(전체5) {np.mean(a5):.3f} · k3_red {np.mean(rd):.3f}")
print(f"  다양−중복 간격 평균 {np.mean(rd)-np.mean(dv):.3f}")
