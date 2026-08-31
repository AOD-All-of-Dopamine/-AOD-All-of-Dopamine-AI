"""X-18 dev 미채점 합집합을 뽑아 눈가림 배치로 만든다."""
import os, sys, json, collections, numpy as np, pandas as pd
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb"); sys.path.insert(0,".")
os.environ.setdefault("AOD_ARTIFACTS","artifacts/tmdb_v1")
from src.personalized_retrieve import build_components, recommend
from src.config import PRODUCTION
ns={}; exec(open("artifacts/p1/grades.py",encoding="utf-8").read(), ns); G=ns["G"]
prof=pd.read_parquet("artifacts/p1/profiles.parquet"); K=50
base={k:v for k,v in PRODUCTION.items() if k!="strategy"}
lists={}; need=collections.OrderedDict()
for kw in (0.0,0.2,0.4,0.6):
    comps=build_components(**base, kw_w=kw); ds=comps[3].dataset
    for r in prof.itertuples():
        rows=[int(x) for x in recommend([int(x) for x in r.seed_rows], components=comps, top_n=K,
              strategy=PRODUCTION["strategy"], postprocess_on=True)["row"].head(K)]
        lists[f"{kw}|{r.profile_id}"]=rows
        for x in rows:
            if f"{r.profile_id}\t{x}" not in G: need.setdefault((r.profile_id,x), None)
    del comps
    print(f"kw_w={kw} 완료", flush=True)
json.dump({k:v for k,v in lists.items()}, open("artifacts/p1/x18_lists.json","w"))
comps=build_components(**base, kw_w=0.4); ds=comps[3].dataset
NAME=dict(zip(prof.profile_id, prof.seed_rows))
def seedstr(pid):
    return "T: "+" / ".join(str(ds.iloc[int(s)]["name"])[:22] for s in NAME[pid][:6])
todo=[]
for i,(pid,row) in enumerate(need):
    r=ds.iloc[int(row)]; g=r["genres"]; g=list(g) if g is not None and len(g) else []
    media="영화" if r["media"]=="movie" else "TV"
    todo.append(dict(id=f"k{i:04d}", pid=pid, item=str(row), seed=seedstr(pid), cand=str(r["name"]),
                     meta=f"영상 · {media} · {', '.join(map(str,g)) or '-'} | 평점 {float(r['vote_average'] or 0):.1f} · 투표 {int(r['vote_count'] or 0):,}"))
json.dump(todo, open("artifacts/p1/x18_todo.json","w"), ensure_ascii=False)
S="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
nb=(len(todo)+59)//60
for b in range(nb):
    open(f"{S}/x18_b{b}.json","w").write(json.dumps([{k:t[k] for k in ("id","seed","cand","meta")} for t in todo[b*60:(b+1)*60]],ensure_ascii=False))
print(f"미채점 합집합 {len(todo)} · 배치 {nb}")
