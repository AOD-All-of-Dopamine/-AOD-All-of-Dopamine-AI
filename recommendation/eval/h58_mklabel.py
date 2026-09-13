"""h58 fun_tag 라벨 입력. 등록 md5 d1806993ed147edd0b1d1bf1db363473."""
import json, random, sys, collections
import pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L as L50
from h51_labels import L as L51
from h53_labels import L as L53
from h56_labels import L as L56
from h57_labels import L as L57
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
DOMS=('wn','steam','tmdb')

tags={}
for k in json.load(open(f'{SP}/h50_key.json')):
    tags[(k['dom'] or 'wn', k['name'])]=L50[k['id']]
for k in json.load(open(f'{SP}/h51_key.json')): tags[('wn',k['name'])]=L51[k['id']]
for k in json.load(open(f'{SP}/h53_key.json')): tags[(k['dom'],k['name'])]=L53[k['id']]
for k in json.load(open(f'{SP}/h56_key.json')): tags[(k['dom'],k['name'])]=L56[k['id']]
for k in json.load(open(f'{SP}/h57_key.json')): tags[(k['dom'],k['name'])]=L57[k['id']]
print(f"기존 라벨 {len(tags)}건")

def texts(base, idcol, col):
    df=pd.read_parquet(f"{base}/dataset.parquet", columns=[idcol,"name",col])
    return {str(n): ("" if pd.isna(d) else str(d)) for n,d in zip(df["name"], df[col])}
TX={'wn':texts(WN,"item_id","synopsis"),'steam':texts(STEAM,"steam_appid","semantic_text"),
    'tmdb':texts(TMDB,"item_id","overview")}

P=json.load(open(f'{SP}/h58_pools.json'))
need={}; slots=0; prof_missing=[]
for ss,blk in P.items():
    for rd,rn in blk['profile']:
        if (rd,rn) not in tags: need[(rd,rn)]=TX[rd][rn]
    for arm in ('D','C'):
        for dom in DOMS:
            for c in blk[arm][dom]:
                slots+=1
                if (dom,c['n']) not in tags: need[(dom,c['n'])]=TX[dom][c['n']]
print(f"풀 슬롯 {slots} · 신규 라벨 필요 {len(need)}건 (프로필 항목 포함)")
print("  신규 도메인별", dict(collections.Counter(d for d,_ in need)))
items=[dict(dom=d, name=n, text=t) for (d,n),t in need.items()]
L=sorted(len(i['text']) for i in items)
if L[len(L)//2]==L[-1]: raise SystemExit("[절단 감지]")
print(f"  본문 중앙 {L[len(L)//2]} · 최대 {L[-1]} · 절단 없음")

rng=random.Random(20260820)
pm=list(range(len(items))); rng.shuffle(pm)
for pos,i in enumerate(pm): items[i]['id']=f"T{pos:03d}"
json.dump([{k:i[k] for k in ('id','dom','name')} for i in items],
          open(f"{SP}/h58_key.json","w"), ensure_ascii=False, indent=1)
order=list(range(len(items))); rng.shuffle(order)
bl=[f"## {items[i]['id']}  ({items[i]['dom']}) {items[i]['name']}\n{items[i]['text']}\n" for i in order]
for k in range(0, len(bl), 50):
    body="\n".join(bl[k:k+50])
    open(f"{SP}/yl{k//50+1}.txt","w").write(body)
    print(f"  yl{k//50+1}.txt {len(bl[k:k+50])}건 {len(body):,}자")
