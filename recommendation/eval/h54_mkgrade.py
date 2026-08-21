"""h54 채점표 — 등록 md5 884cfa6e6b1c0fbfd200919400acd139. 시드 0~9, h53 등급 ≤1 인 후보만."""
import json, random, sys, collections
import pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_grades import G as G50
from h51_grades import G as G51
from h53_grades import G as G53
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
DOMS=('wn','steam','tmdb')
SEEDS=list(range(10))

grade={}
for k in json.load(open(f'{SP}/h50_grade_key.json')): grade[(k['seed'],k['dom'],k['cand_name'])]=G50[k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')): grade[(k['seed'],'wn',k['cand_name'])]=G51[k['pid']]
for k in json.load(open(f'{SP}/h53_grade_key.json')): grade[(k['seed'],k['dom'],k['cand_name'])]=G53[k['pid']]

D=json.load(open(f'{SP}/h53_pools.json')); pools=D['pools']; profs=D['profiles']

# 본문 지도 (dom,name) -> 전문
def texts(base, idcol, col):
    df=pd.read_parquet(f"{base}/dataset.parquet", columns=[idcol,"name",col])
    return {str(n): ("" if pd.isna(d) else str(d)) for n,d in zip(df["name"], df[col])}
TX={'wn':texts(WN,"item_id","synopsis"),'steam':texts(STEAM,"steam_appid","semantic_text"),
    'tmdb':texts(TMDB,"item_id","overview")}

need=[]; seen=set()
for s in SEEDS:
    ss=str(s)
    ref=profs[ss]  # {'M':[[dom,name],...], 'S':[...]}
    for arm in ('M','S'):
        for d in DOMS:
            for c in pools[ss][arm][d]:
                if grade[(s,d,c['n'])] >= 2: continue          # max 가 바뀔 수 없다
                for rdom, rname in ref[arm][1:]:                # 항목2·항목3
                    key=(s,arm,rdom,rname,d,c['n'])
                    kk=(rdom,rname,d,c['n'])
                    if kk in seen: continue
                    seen.add(kk)
                    need.append(dict(seed=s, arm=arm, rdom=rdom, rname=rname,
                                     dom=d, cand=c['n'], text=TX[d][c['n']],
                                     rtext=TX[rdom][rname]))
print(f"신규 채점 {len(need)}쌍")
print("  참조 도메인별", dict(collections.Counter(p['rdom'] for p in need)))
print("  후보 도메인별", dict(collections.Counter(p['dom'] for p in need)))
L=sorted(len(p['text']) for p in need)
if L[len(L)//2]==L[-1]: raise SystemExit("[절단 감지]")
print(f"  본문 중앙 {L[len(L)//2]} · 최대 {L[-1]} · 절단 없음")

rng=random.Random(20260911)
pm=list(range(len(need))); rng.shuffle(pm)
for pos,i in enumerate(pm): need[i]['pid']=f"Z{pos:03d}"
json.dump([{k:p[k] for k in ('pid','seed','arm','rdom','rname','dom','cand')} for p in need],
          open(f"{SP}/h54_grade_key.json","w"), ensure_ascii=False, indent=1)
order=list(range(len(need))); rng.shuffle(order)
bl=[f"## {need[i]['pid']}\n**기준 작품** ({need[i]['rdom']}) {need[i]['rname']}\n{need[i]['rtext']}\n\n"
    f"**후보** ({need[i]['dom']}) {need[i]['cand']}\n{need[i]['text']}\n" for i in order]
for k in range(0, len(bl), 30):
    body="\n".join(bl[k:k+30])
    open(f"{SP}/zg{k//30+1}.txt","w").write(body)
    print(f"  zg{k//30+1}.txt {len(bl[k:k+30])}쌍 {len(body):,}자")
