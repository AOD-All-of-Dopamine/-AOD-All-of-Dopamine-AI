"""h67 채점 쌍 생성 — 기존 봉인 등급은 재사용, 신규만 낸다. 등록 96fd17ea84e4d04b635f0a9785af58a1."""
import json, random, collections
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
RNG=20260906
def loadG(p):
    ns={}; exec(open(f"{SP}/{p}").read(),ns); return ns['G']
HAVE=set()
for b in ["h54","h55","h57","h58","h59","h60","h62"]:
    K=json.load(open(f"{SP}/{b}_grade_key.json")); G=loadG(f"{b}_grades.py")
    for k in K:
        if 'rname' in k and k['pid'] in G: HAVE.add((k['rdom'],k['rname'],k['dom'],k['cand']))
P=json.load(open(f"{SP}/h67_pools.json"))
# 기준 작품 본문 (프로필 항목) — h62 pools 에 없으므로 코퍼스에서 가져온다
import pandas as pd, numpy as np
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
def frame(path,idcol,txt):
    d=pd.read_parquet(f"{path}/dataset.parquet", columns=[idcol,"name",txt])
    i=pd.read_parquet(f"{path}/corpus_index.parquet").sort_values("embedding_row")
    d=d.set_index(idcol).loc[i[idcol].to_numpy()].reset_index()
    m={}
    for n,t in zip(d["name"],d[txt]): m.setdefault(str(n), str(t))
    return m
TXTM={'wn':frame(WN,"item_id","synopsis"),'steam':frame(STEAM,"steam_appid","semantic_text"),
      'tmdb':frame(TMDB,"item_id","overview")}
need=[]; reuse=0; seen=set()
for ss,rec in P.items():
    prof=[(d,n) for d,n in rec['profile']]
    for tag in ('R-min','R-cent'):
        for dom,lst in rec[tag].items():
            for c in lst:
                for pd_,pn in prof:
                    if pn==c['n']: continue
                    key=(pd_,pn,dom,c['n'])
                    if key in HAVE: reuse+=1; continue
                    if key in seen: continue
                    seen.add(key)
                    need.append(dict(rdom=pd_,rname=pn,dom=dom,cand=c['n'],
                                     rtext=TXTM[pd_].get(pn,"")[:900], ctext=c['d'][:900]))
rng=random.Random(RNG); rng.shuffle(need)
for i,r in enumerate(need): r['pid']=f"W{i:03d}"
json.dump([{k:r[k] for k in ('pid','rdom','rname','dom','cand')} for r in need],
          open(f"{SP}/h67_grade_key.json","w"), ensure_ascii=False)
CH=30
for i in range(0,len(need),CH):
    with open(f"{SP}/wg{i//CH+1}.txt","w",encoding="utf-8") as f:
        for r in need[i:i+CH]:
            f.write(f"## {r['pid']}\n**기준 작품** ({r['rdom']}) {r['rname']}\n{r['rtext']}\n\n"
                    f"**후보** ({r['dom']}) {r['cand']}\n{r['ctext']}\n\n")
print(f"필요 조합 {len(need)+reuse} · 기존 재사용 {reuse} · 신규 고유 {len(need)}")
print(f"파일 wg1~wg{(len(need)+CH-1)//CH}.txt")
ln=[len(r['ctext']) for r in need]
print(f"본문 중앙 {int(np.median(ln))} · 최대 {max(ln)} · 절단 {'있음' if max(ln)>=900 else '없음'}")
