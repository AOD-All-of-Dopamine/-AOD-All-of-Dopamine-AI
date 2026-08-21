"""h62 채점표 — 깊은 구간 표본 × 프로필 3항목. 기존 등급 재사용."""
import json, random, sys, collections
import pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
mods={}
for h in ('h50','h51','h53','h54','h55','h57','h58','h59','h60'):
    mods[h]=__import__(f"{h}_grades").G
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
DOMS=('wn','steam','tmdb')
have={}
sn={s:x['wn'] for s,x in enumerate(json.load(open(f'{SP}/h50_seeds20.json')))}
for k in json.load(open(f'{SP}/h50_grade_key.json')): have[('wn',sn[k['seed']],k['dom'],k['cand_name'])]=mods['h50'][k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')): have[('wn',sn[k['seed']],'wn',k['cand_name'])]=mods['h51'][k['pid']]
for k in json.load(open(f'{SP}/h53_grade_key.json')): have[('wn',sn[k['seed']],k['dom'],k['cand_name'])]=mods['h53'][k['pid']]
for h in ('h54','h55','h57','h58','h59','h60'):
    for k in json.load(open(f'{SP}/{h}_grade_key.json')): have[(k['rdom'],k['rname'],k['dom'],k['cand'])]=mods[h][k['pid']]
print(f"기존 등급 지도 {len(have)}건")
def texts(base, idcol, col):
    df=pd.read_parquet(f"{base}/dataset.parquet", columns=[idcol,"name",col])
    return {str(n): ("" if pd.isna(d) else str(d)) for n,d in zip(df["name"], df[col])}
TX={'wn':texts(WN,"item_id","synopsis"),'steam':texts(STEAM,"steam_appid","semantic_text"),
    'tmdb':texts(TMDB,"item_id","overview")}
P=json.load(open(f'{SP}/h62_pools.json'))
need=[]; seen=set(); reuse=0; tot=0
for blk in P.values():
    prof=blk['profile']
    for dom in DOMS:
        for c in blk['sample'][dom]:
            for rdom,rname in prof:
                key=(rdom,rname,dom,c['n']); tot+=1
                if key in have: reuse+=1; continue
                if key in seen: continue
                seen.add(key)
                need.append(dict(rdom=rdom,rname=rname,dom=dom,cand=c['n'],
                                 rtext=TX[rdom][rname], text=TX[dom][c['n']]))
print(f"필요 조합 {tot} · 기존 재사용 {reuse} · 신규 고유 {len(need)}")
L=sorted(len(p['text']) for p in need)
if L[len(L)//2]==L[-1]: raise SystemExit("[절단 감지]")
print(f"  본문 중앙 {L[len(L)//2]} · 최대 {L[-1]} · 절단 없음")
rng=random.Random(20260830)
pm=list(range(len(need))); rng.shuffle(pm)
for pos,i in enumerate(pm): need[i]['pid']=f"V{pos:03d}"
json.dump([{k:p[k] for k in ('pid','rdom','rname','dom','cand')} for p in need],
          open(f"{SP}/h62_grade_key.json","w"), ensure_ascii=False, indent=1)
order=list(range(len(need))); rng.shuffle(order)
bl=[f"## {need[i]['pid']}\n**기준 작품** ({need[i]['rdom']}) {need[i]['rname']}\n{need[i]['rtext']}\n\n"
    f"**후보** ({need[i]['dom']}) {need[i]['cand']}\n{need[i]['text']}\n" for i in order]
for k in range(0, len(bl), 30):
    body="\n".join(bl[k:k+30])
    open(f"{SP}/vg{k//30+1}.txt","w").write(body)
    print(f"  vg{k//30+1}.txt {len(bl[k:k+30])}쌍 {len(body):,}자")
