"""h52 판정 — 등록 md5 b48a6ea5231f2bf736afdaef7c1a05b0. 신규 채점 0건."""
import json, math, collections, random, sys
import numpy as np, pandas as pd
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L as L50
from h50_grades import G as G50
from h51_labels import L as L51
from h51_grades import G as G51
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
STEAM="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB="/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"

# ---------- 봉인된 라벨·등급 조립 ----------
k50=json.load(open(f'{SP}/h50_key.json')); g50=json.load(open(f'{SP}/h50_grade_key.json'))
k51=json.load(open(f'{SP}/h51_key.json')); g51=json.load(open(f'{SP}/h51_grade_key.json'))
seed_tag={k['seed']:L50[k['id']] for k in k50 if k['kind']=='seed'}
seed_name={}
d50=json.load(open(f'{SP}/h50_seeds20.json')); d51=json.load(open(f'{SP}/h51_wn20.json'))
for s,x in enumerate(d50): seed_name[s]=x['wn']
cand={}; grade={}; cname={}
for k in k50:
    if k['kind']=='cand':
        cand[(k['seed'],k['dom'],k['cand'])]=L50[k['id']]; cname[(k['seed'],k['dom'],k['cand'])]=k['name']
for k in g50: grade[(k['seed'],k['dom'],k['cand'])]=G50[k['pid']]
for k in k51:
    cand[(k['seed'],'wn',k['cand'])]=L51[k['id']]; cname[(k['seed'],'wn',k['cand'])]=k['name']
for k in g51: grade[(k['seed'],'wn',k['cand'])]=G51[k['pid']]
DOMS=('wn','steam','tmdb')

# ---------- 라벨된 항목 풀 (620건) 의 임베딩 ----------
def load(base, idcol, namecol, dscol=None):
    df=pd.read_parquet(f"{base}/dataset.parquet")
    idx=pd.read_parquet(f"{base}/corpus_index.parquet").sort_values("embedding_row")
    df=df.set_index(idcol).loc[idx[idcol].to_numpy()].reset_index()
    E=np.load(f"{base}/corpus_embeddings.npy", mmap_mode="r")
    return df, E
wdf,WE = load(WN,"item_id","name")
sdf,SE = load(STEAM,"steam_appid","name")
tdf,TE = load(TMDB,"item_id","name")
rowmap={'wn':{str(n):i for i,n in enumerate(wdf["name"])},
        'steam':{str(n):i for i,n in enumerate(sdf["name"])},
        'tmdb':{str(n):i for i,n in enumerate(tdf["name"])}}
EMB={'wn':WE,'steam':SE,'tmdb':TE}

# 라벨된 항목 풀: (dom, name, tags, embedding row) — 시드 자신 제외는 아래에서
pool=collections.defaultdict(list)          # dom -> [(name, tags, row, owner_seed)]
for (s,dom,c),tags in cand.items():
    nm=cname[(s,dom,c)]
    r=rowmap[dom].get(str(nm))
    if r is not None: pool[dom].append((nm, tags, r, s))
for dom in DOMS: print(f"  라벨된 {dom:6s} 항목 {len(pool[dom])}건 (임베딩 매칭 성공)")

def unit(v):
    v=np.asarray(v,dtype=np.float32); n=np.linalg.norm(v); return v/n if n else v

def pick_similar(seed_row, dom, s_exclude, k):
    """s 자신의 후보를 제외한 라벨 풀에서 시드와 임베딩 유사도 상위 k개."""
    q=unit(WE[seed_row])
    cands=[(nm,tg,r) for (nm,tg,r,ow) in pool[dom] if ow!=s_exclude]
    seen=set(); out=[]
    for nm,tg,r in sorted(cands, key=lambda t:-float(q @ unit(EMB[dom][t[2]]))):
        if nm in seen: continue
        seen.add(nm); out.append((nm,tg,r))
        if len(out)==k: break
    return out

seed_row={s: rowmap['wn'][seed_name[s]] for s in seed_name}
profiles={}
for s in sorted(seed_tag):
    x=pick_similar(seed_row[s],'steam',s,1)[0]
    y=pick_similar(seed_row[s],'tmdb',s,1)[0]
    ab=pick_similar(seed_row[s],'wn',s,2)
    profiles[s]={'M':[('wn',seed_name[s],seed_tag[s],seed_row[s]),
                      ('steam',x[0],x[1],x[2]), ('tmdb',y[0],y[1],y[2])],
                 'S':[('wn',seed_name[s],seed_tag[s],seed_row[s]),
                      ('wn',ab[0][0],ab[0][1],ab[0][2]), ('wn',ab[1][0],ab[1][1],ab[1][2])]}

print("\n[프로필 구성 표본]")
for s in (0,5,11,17):
    for arm in ('M','S'):
        print(f"  시드{s:2d} {arm}: " + " | ".join(f"{d}:{n}" for d,n,_,_ in profiles[s][arm]))

# ---------- 점수 ----------
def cos(a,b):
    X,Y=set(a),set(b); return len(X&Y)/math.sqrt(len(X)*len(Y)) if X and Y else 0.0
def pct(v):
    n=len(v); o=sorted(range(n),key=lambda i:v[i]); r=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and v[o[j+1]]==v[o[i]]: j+=1
        for k in range(i,j+1): r[o[k]]=((i+j)/2)/(n-1)
        i=j+1
    return r

rng=random.Random(20260907)
res={arm:dict(tri=0, wn=0, fit=0, rnd=0) for arm in ('M','S')}
for s in sorted(seed_tag):
    items=[dict(dom=dom, rank=c, gr=grade[(s,dom,c)], tg=cand[(s,dom,c)])
           for dom in DOMS for c in range(10)]
    rsel=rng.sample(range(30),5)
    for arm in ('M','S'):
        ptags=set().union(*[set(t) for _,_,t,_ in profiles[s][arm]])   # 합집합
        ft=[cos(ptags, it['tg']) for it in items]
        sim=[-it['rank'] for it in items]
        E=[0.643*a+0.357*b for a,b in zip(pct(ft),pct(sim))]
        top=sorted(range(30), key=lambda i:-ft[i])[:5]
        res[arm]['tri'] += len({items[i]['dom'] for i in top})==3
        res[arm]['wn']  += sum(1 for i in top if items[i]['dom']=='wn')
        res[arm]['fit'] += sum(1 for i in top if items[i]['gr']>=2)
        res[arm]['rnd'] += sum(1 for i in rsel if items[i]['gr']>=2)
        res[arm].setdefault('dom',collections.Counter())
        for i in top: res[arm]['dom'][items[i]['dom']]+=1
        res[arm].setdefault('Etri',0); res[arm].setdefault('Efit',0); res[arm].setdefault('Ewn',0)
        te=sorted(range(30), key=lambda i:-E[i])[:5]
        res[arm]['Etri'] += len({items[i]['dom'] for i in te})==3
        res[arm]['Efit'] += sum(1 for i in te if items[i]['gr']>=2)
        res[arm]['Ewn']  += sum(1 for i in te if items[i]['dom']=='wn')

print("\n"+"="*78); print("h52 판정 — 혼합 프로필 M vs 단일 도메인 프로필 S (크기 3, 집계 동일)"); print("="*78)
print(f"  {'':22s} {'혼합 M':>10s} {'단일 S':>10s}   차이")
def row(lab, a, b, fmt="{:>10d}"):
    print(f"  {lab:22s} {fmt.format(a):>10s} {fmt.format(b):>10s}   {a-b:+d}")
print(f"  {'ft top-5 세 도메인':22s} {res['M']['tri']:>7d}/20 {res['S']['tri']:>7d}/20   {res['M']['tri']-res['S']['tri']:+d}")
print(f"  {'ft top-5 웹소설 칸':22s} {res['M']['wn']:>6d}/100 {res['S']['wn']:>6d}/100   {res['M']['wn']-res['S']['wn']:+d}")
print(f"  {'ft top-5 적합률':22s} {res['M']['fit']:>6d}/100 {res['S']['fit']:>6d}/100   {res['M']['fit']-res['S']['fit']:+d}%p")
print(f"  {'무작위 top-5 적합률':22s} {res['M']['rnd']:>6d}/100 {res['S']['rnd']:>6d}/100")
for arm in ('M','S'):
    d=res[arm]['dom']; print(f"  {arm} 도메인 구성: wn {d['wn']} / steam {d['steam']} / tmdb {d['tmdb']}")

print("\n[등록 판정]")
w4 = all((res[a]['fit']-res[a]['rnd'])>=10 for a in ('M','S'))
print(f"  W4 건전성: ft − 무작위  M {res['M']['fit']-res['M']['rnd']:+d}%p · S {res['S']['fit']-res['S']['rnd']:+d}%p"
      f" (기준 +10%p 둘 다)  {'통과' if w4 else '탈락'}")
if not w4: print("  ⇒ 무효. W1~W3 해석 금지."); sys.exit()
w1=(res['M']['tri']-res['S']['tri'])>=6
w2=(res['M']['wn']-res['S']['wn'])<=-15
w3=(res['M']['fit']-res['S']['fit'])>=-5
print(f"  W1 세 도메인 시드 M−S = {res['M']['tri']-res['S']['tri']:+d} (기준 ≥+6)   {'적중' if w1 else '빗나감'}")
print(f"  W2 웹소설 칸 M−S = {res['M']['wn']-res['S']['wn']:+d} (기준 ≤−15)       {'적중' if w2 else '빗나감'}")
print(f"  W3 적합률 M−S = {res['M']['fit']-res['S']['fit']:+d}%p (기준 ≥−5%p)     {'적중' if w3 else '빗나감'}")
print(f"  W1·W2 방향 일치: {'예' if (w1==w2) else '아니오 — 해석 보류'}")

print("\n[참고 · 기각 조건 없음] E(기획서 원안)로 세웠을 때")
print(f"  세 도메인 시드  M {res['M']['Etri']}/20 · S {res['S']['Etri']}/20")
print(f"  웹소설 칸       M {res['M']['Ewn']}/100 · S {res['S']['Ewn']}/100")
print(f"  적합률          M {res['M']['Efit']}/100 · S {res['S']['Efit']}/100")

# ---- W5 기술 통계 (기각 조건 없음) ----
print("\n[W5 · 기술 통계] 프로필 태그 합집합 크기")
for arm in ('M','S'):
    sizes=[len(set().union(*[set(t) for _,_,t,_ in profiles[s][arm]])) for s in sorted(seed_tag)]
    print(f"  {arm}: 중앙 {sorted(sizes)[len(sizes)//2]} · 최소 {min(sizes)} · 최대 {max(sizes)}")

print("\n[W5 · 기술 통계] top2_mean 혼합 질의로 재검색 시 풀이 얼마나 바뀌는가")
def top2_mean(rows_by_dom):
    """확정 파이프라인: 프로필 항목별 유사도 상위 2개의 평균 (여기선 질의 벡터를 항목 평균으로)"""
    V=[unit(EMB[d][r]) for d,_,_,r in rows_by_dom]
    return unit(np.mean(V,axis=0))
LAM=0.35
cen={d: unit(np.asarray(EMB[d][::37],dtype=np.float32).mean(axis=0)) for d in DOMS}
chg=collections.Counter(); tot=collections.Counter()
for s in sorted(seed_tag):
    q_single = unit(WE[seed_row[s]])
    q_mix    = top2_mean(profiles[s]['M'])
    for d in DOMS:
        E_=np.asarray(EMB[d],dtype=np.float32)
        base=set(np.argsort(-( (q_single-LAM*cen[d]) @ E_.T ))[:10])
        mix =set(np.argsort(-( (q_mix   -LAM*cen[d]) @ E_.T ))[:10])
        chg[d]+=len(mix-base); tot[d]+=10
for d in DOMS:
    print(f"  {d:6s} top-10 중 바뀐 후보 {chg[d]}/{tot[d]} = {100*chg[d]/tot[d]:.1f}%")
