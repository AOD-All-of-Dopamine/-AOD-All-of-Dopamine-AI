"""h51 — h50 의 20시드에 웹소설 후보 top-10 추가. 등록 md5 acc1f8f6640f0e6ae83877aa0be0ee64."""
import json
import numpy as np, pandas as pd
SP="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
LAM=0.35
d=json.load(open(f"{SP}/h50_seeds20.json"))
names=[x["wn"] for x in d]

wd=pd.read_parquet(f"{WN}/dataset.parquet", columns=["item_id","name","genres","synopsis"])
wi=pd.read_parquet(f"{WN}/corpus_index.parquet").sort_values("embedding_row")
wd=wd.set_index("item_id").loc[wi["item_id"].to_numpy()].reset_index()
row={str(n):i for i,n in enumerate(wd["name"])}
missing=[n for n in names if n not in row]
if missing: raise SystemExit(f"시드 미발견: {missing}")

E=np.asarray(np.load(f"{WN}/corpus_embeddings.npy", mmap_mode="r"), dtype=np.float32)
cen=E.mean(axis=0)
Q=E[[row[n] for n in names]] - LAM*cen[None,:]
S=Q @ E.T
short=(wd["synopsis"].str.len()<100).to_numpy()
S[:, short]=-1e9
def pref(x): return str(x)[:6]
for r,n in enumerate(names):                       # 시드 자신 + 동일 작품(제목 접두) 제외
    same=(wd["name"].map(pref)==pref(n)).to_numpy()
    S[r, same]=-1e9
print(f"웹소설 {len(wd):,} → 줄거리 100자 미만 {int(short.sum()):,} 제외")

out=[]
for r,n in enumerate(names):
    top=np.argsort(-S[r])[:10]
    out.append({"g": d[r]["g"], "wn": n, "syn": d[r]["syn"],
        "c":[{"r":k+1, "n": wd.loc[int(t),"name"], "d": wd.loc[int(t),"synopsis"],
              "sim": float(S[r,t])} for k,t in enumerate(top)]})
json.dump(out, open(f"{SP}/h51_wn20.json","w"), ensure_ascii=False, indent=1)
D=[len(c["d"]) for x in out for c in x["c"]]
print(f"저장 h51_wn20.json · 후보 텍스트 중앙 {sorted(D)[len(D)//2]} 최대 {max(D)} "
      f"{'절단 없음' if sorted(D)[len(D)//2]!=max(D) else '** 절단 의심 **'}")
for i in (0,5,11,17):
    print(f"  [{out[i]['g']}] {out[i]['wn']} → " + " / ".join(c['n'] for c in out[i]['c'][:4]))
