"""X-22: Steam quality_w ∈ {0, 0.25, 0.50} 을 저리뷰 9 + 대작 5 프로필에서 뽑는다."""
import os, sys, json, random, collections, numpy as np, pandas as pd
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
S="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
ROOT=os.getcwd(); sys.path.insert(0,os.path.join(ROOT,"steam")); os.chdir("steam")
os.environ.setdefault("AOD_ARTIFACTS","artifacts/tags_full")
from src.personalized_retrieve import build_components, run_multi
from src.postprocess import postprocess as pp
from src.config import PRODUCTION
os.chdir(ROOT)
ds=pd.read_parquet("steam/artifacts/tags_full/dataset.parquet",columns=["steam_appid","name","tags","recommendations_total"])
rc=pd.to_numeric(ds["recommendations_total"],errors="coerce").fillna(0); by=dict(zip(ds["steam_appid"].astype(int),rc))
prof=pd.read_parquet("steam/artifacts/p1/profiles.parquet")
rows_p=[(np.median([by.get(int(x),0) for x in r.liked_appids]), r.profile_id, [int(x) for x in r.liked_appids]) for r in prof.itertuples()]
rows_p.sort()
LOW=[(p,s) for m,p,s in rows_p if m<1500]
HIGH=[(p,s) for m,p,s in rows_p[-5:]]
print(f"저리뷰 {len(LOW)}개: {[p for p,_ in LOW]}")
print(f"대작 대조군 {len(HIGH)}개: {[p for p,_ in HIGH]}")
lists={}
for qw in (0.0,0.25,0.50):
    comps=build_components(quality_w=qw)
    for grp,profs in (("low",LOW),("high",HIGH)):
        for pid,seeds in profs:
            r=run_multi(seeds, strategies=["top2_mean"], top_n=50, components=comps, postprocess=True)["top2_mean"]
            col=next(c for c in ("steam_appid","appid","candidate_appid") if c in r.columns)
            lists[f"{qw}|{grp}|{pid}"]=[int(x) for x in r[col].head(50)]
    print(f"qw={qw} 완료", flush=True)
    del comps
json.dump(lists,open("crossdomain/x22_lists.json","w"))
for qw in (0.0,0.25,0.50):
    for grp in ("low","high"):
        L=[v for k,v in lists.items() if k.startswith(f"{qw}|{grp}|")]
        med=np.median([np.median([by.get(a,0) for a in x]) for x in L])
        print(f"qw={qw} {grp}: 리뷰수 중앙 {med:,.0f} · 고유 {len(set(sum(L,[])))}")
