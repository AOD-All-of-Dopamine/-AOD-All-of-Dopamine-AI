"""h63 J4 — 사전 등록된 기술 통계(쌍 단위 순위 정확도 · 예측–실제 상관)."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
exec(open('/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad/h63_rerank.py').read().split("def run(")[0])
ALPHAS=[0.1,1,10,100,1000]
P=json.load(open(f"{SP}/h62_pools.json"))
rows=[]
for ss in P:
    profkeys={f"{d}|{n}" for d,n in P[ss]['profile']}
    tr=~np.isin(refkey,list(profkeys)); te=~tr
    if te.sum()==0: continue
    m=Ridge(alpha=0.1).fit(X[tr],y[tr])
    pred=m.predict(X[te]); act=y[te]
    cos=np.array([float(np.dot(unit(EMB[rd][ROW[rd][rn]]), unit(EMB[cd][ROW[cd][cn]]-LAM*CEN[cd])))
                  for rd,rn,cd,cn in np.array(keys,dtype=object)[te]])
    rows.append((ss,te.sum(),np.corrcoef(pred,act)[0,1],np.corrcoef(cos,act)[0,1]))
print(f"{'시드':>5} {'시험쌍':>6} {'학습 상관':>9} {'코사인 상관':>11}")
for ss,n,a,b in rows: print(f"{ss:>5} {n:>6} {a:>9.3f} {b:>11.3f}")
print(f"{'평균':>5} {np.mean([r[1] for r in rows]):>6.0f} {np.nanmean([r[2] for r in rows]):>9.3f} {np.nanmean([r[3] for r in rows]):>11.3f}")
# 쌍 단위 순위 정확도: 같은 기준작품 안에서 등급이 다른 두 후보의 순서를 맞히는가
def pair_acc(score):
    ok=t=0
    by={}
    for i,(rd,rn,cd,cn) in enumerate(np.array(keys,dtype=object)):
        by.setdefault((rd,rn),[]).append(i)
    for k,idx in by.items():
        for a in range(len(idx)):
            for b in range(a+1,len(idx)):
                i,j=idx[a],idx[b]
                if y[i]==y[j]: continue
                t+=1
                if (score[i]-score[j])*(y[i]-y[j])>0: ok+=1
    return 100*ok/t, t
cos_all=np.array([float(np.dot(unit(EMB[rd][ROW[rd][rn]]), unit(EMB[cd][ROW[cd][cn]]-LAM*CEN[cd])))
                  for rd,rn,cd,cn in np.array(keys,dtype=object)])
a,t=pair_acc(cos_all)
print(f"\n코사인 쌍 단위 순위 정확도(전체, 같은 기준작품 내): {a:.1f}%  (비교쌍 {t})")
