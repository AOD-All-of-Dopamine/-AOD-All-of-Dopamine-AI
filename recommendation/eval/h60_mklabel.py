"""h60 라벨 입력 — 항목마다 추상 14축 3개 + 구체 14축 3개를 한 패스에서. 등록 f6d8d1aaee88fb364a8c7f3d75dc01bd."""
import json, random, collections
import pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
DOMS=('wn','steam','tmdb')
def texts(base, idcol, col):
    df=pd.read_parquet(f"{base}/dataset.parquet", columns=[idcol,"name",col])
    return {str(n): ("" if pd.isna(d) else str(d)) for n,d in zip(df["name"], df[col])}
TX={'wn':texts(WN,"item_id","synopsis"),'steam':texts(STEAM,"steam_appid","semantic_text"),
    'tmdb':texts(TMDB,"item_id","overview")}
P=json.load(open(f'{SP}/h60_pools.json'))
need={}
for blk in P.values():
    for d,n in blk['profile']: need[(d,n)]=TX[d][n]
    for d in DOMS:
        for c in blk['D'][d]: need[(d,c['n'])]=TX[d][c['n']]
items=[dict(dom=d, name=n, text=t) for (d,n),t in need.items()]
print(f"라벨 대상 {len(items)}건 · 도메인별 {dict(collections.Counter(i['dom'] for i in items))}")
L=sorted(len(i['text']) for i in items)
if L[len(L)//2]==L[-1]: raise SystemExit("[절단 감지]")
print(f"  본문 중앙 {L[len(L)//2]} · 최대 {L[-1]} · 절단 없음")
rng=random.Random(20260827)
pm=list(range(len(items))); rng.shuffle(pm)
for pos,i in enumerate(pm): items[i]['id']=f"U{pos:03d}"
json.dump([{k:i[k] for k in ('id','dom','name')} for i in items],
          open(f"{SP}/h60_key.json","w"), ensure_ascii=False, indent=1)
order=list(range(len(items))); rng.shuffle(order)
bl=[f"## {items[i]['id']}  ({items[i]['dom']}) {items[i]['name']}\n{items[i]['text']}\n" for i in order]
for k in range(0, len(bl), 40):
    body="\n".join(bl[k:k+40])
    open(f"{SP}/ul{k//40+1}.txt","w").write(body)
    print(f"  ul{k//40+1}.txt {len(bl[k:k+40])}건 {len(body):,}자")
