"""X-18 dev: p1 52프로필에서 kw_w 격자 {0, 0.2, 0.4, 0.6} 를 은행으로 상대 비교."""
import os, sys, numpy as np, pandas as pd, collections
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb"); sys.path.insert(0,".")
os.environ.setdefault("AOD_ARTIFACTS","artifacts/tmdb_v1")
from src.personalized_retrieve import build_components, recommend
from src.config import PRODUCTION
ns={}; exec(open("artifacts/p1/grades.py",encoding="utf-8").read(), ns); G=ns["G"]
prof=pd.read_parquet("artifacts/p1/profiles.parquet")
K=50
base={k:v for k,v in PRODUCTION.items() if k!="strategy"}
res={}
for kw in (0.0,0.2,0.4,0.6):
    comps=build_components(**base, kw_w=kw)
    fits=[]; ungraded=0; tot=0; nkw=[]
    for r in prof.itertuples():
        df=recommend([int(x) for x in r.seed_rows], components=comps, top_n=K,
                     strategy=PRODUCTION["strategy"], postprocess_on=True)
        rows=[int(x) for x in df["row"].head(K)]
        g=[G.get(f"{r.profile_id}\t{x}") for x in rows]
        graded=[x for x in g if x is not None]
        ungraded+=sum(1 for x in g if x is None); tot+=len(g)
        if graded: fits.append(np.mean([x>=2 for x in graded]))
        nkw+= [comps[3].dataset["n_keywords"].iloc[x] for x in rows]
    res[kw]=(np.mean(fits), ungraded, tot, np.median(nkw))
    print(f"kw_w={kw}: 적합률 {np.mean(fits):.4f} · 미채점 {ungraded}/{tot} ({ungraded/tot:.1%}) · n_keywords 중앙 {np.median(nkw):.0f}", flush=True)
    del comps
best=max((v[0],k) for k,v in res.items() if k>0)
print(f"\n기준선(0) {res[0.0][0]:.4f} · 최적 kw_w={best[1]} {best[0]:.4f} (Δ {best[0]-res[0.0][0]:+.4f})")
