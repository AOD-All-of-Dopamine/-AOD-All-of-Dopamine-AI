"""h51 판정 — 등록 md5 acc1f8f6640f0e6ae83877aa0be0ee64 · 라벨 md5 1155992ba39f71177f3c0f1c5a408e83."""
import json, math, collections, random, sys
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L as L50
from h50_grades import G as G50
from h51_labels import L as L51
from h51_grades import G as G51

k50=json.load(open(f'{SP}/h50_key.json')); g50=json.load(open(f'{SP}/h50_grade_key.json'))
k51=json.load(open(f'{SP}/h51_key.json')); g51=json.load(open(f'{SP}/h51_grade_key.json'))
seed_tag={k['seed']:L50[k['id']] for k in k50 if k['kind']=='seed'}
cand={}; grade={}
for k in k50:
    if k['kind']=='cand': cand[(k['seed'],k['dom'],k['cand'])]=L50[k['id']]
for k in g50: grade[(k['seed'],k['dom'],k['cand'])]=G50[k['pid']]
for k in k51: cand[(k['seed'],'wn',k['cand'])]=L51[k['id']]
for k in g51: grade[(k['seed'],'wn',k['cand'])]=G51[k['pid']]

def jac(a,b):
    X,Y=set(a),set(b); return len(X&Y)/len(X|Y) if X|Y else 0.0
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
def auc(s,g):
    pos=[i for i,x in enumerate(g) if x>=2]; neg=[i for i,x in enumerate(g) if x<2]
    if not pos or not neg: return None
    return sum(1.0 if s[p]>s[n] else (0.5 if s[p]==s[n] else 0.0) for p in pos for n in neg)/(len(pos)*len(neg))
def med(v):
    s=sorted(v); n=len(s); return s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
def wilcox(d):
    d=[x for x in d if x!=0]; n=len(d)
    if n<6: return 1.0
    o=sorted(range(n),key=lambda i:abs(d[i])); rk=[0]*n; i=0
    while i<n:
        j=i
        while j+1<n and abs(d[o[j+1]])==abs(d[o[i]]): j+=1
        for k in range(i,j+1): rk[o[k]]=(i+j)/2+1
        i=j+1
    W=sum(rk[i] for i in range(n) if d[i]>0)
    mu=n*(n+1)/4; sd=(n*(n+1)*(2*n+1)/24)**.5
    return math.erfc(abs((W-mu)/sd)/2**.5)

DOMS=('wn','steam','tmdb')
rows=[]
for s in sorted(seed_tag):
    st=seed_tag[s]; items=[]
    for dom in DOMS:
        for ci in range(10):
            items.append(dict(dom=dom, rank=ci, gr=grade[(s,dom,ci)],
                              ft=cos(st,cand[(s,dom,ci)]), J=jac(st,cand[(s,dom,ci)])))
    sim=[-x['rank'] for x in items]; ft=[x['ft'] for x in items]; J=[x['J'] for x in items]
    E=[0.643*a+0.357*b for a,b in zip(pct(ft), pct(sim))]
    gr=[x['gr'] for x in items]
    rows.append(dict(seed=s, items=items, gr=gr, sim=sim, ft=ft, E=E, J=J,
        a_sim=auc(sim,gr), a_ft=auc(ft,gr), a_E=auc(E,gr), a_J=auc(J,gr)))

print("="*78); print("h51 판정 — 웹소설 20시드 × 30후보(웹소설10 + Steam10 + TMDB10) = 600쌍"); print("="*78)
for dom in DOMS:
    v=[grade[(s,dom,c)] for s in seed_tag for c in range(10)]
    print(f"  {dom:6s} 적합 {sum(x>=2 for x in v):3d}/200 = {sum(x>=2 for x in v)/2:5.1f}%")

print("\n[Z5 가드]")
u=[r for r in rows if r['a_sim'] is not None]
z5a=len(u)>=12
print(f"  Z5a 대비: AUC 계산가능 {len(u)}/20 (기준 ≥12)  {'통과' if z5a else '탈락'}")
z5b=False
for lab,k in (('sim','a_sim'),('ft','a_ft'),('E','a_E'),('J','a_J')):
    v=[r[k] for r in u]; p=wilcox([x-0.5 for x in v]); hit=med(v)>0.5 and p<0.0125
    z5b|=hit; print(f"    {lab:4s} 중앙값 {med(v):.3f}  p={p:.4f}  {'통과' if hit else '—'}")
print(f"  Z5b 신호: {'통과' if z5b else '탈락'}  ⇒ 통합 조건 {'유효' if (z5a and z5b) else '무효'}")
valid = z5a and z5b

# ---- Z1 혼합 규칙 ----
rng=random.Random(20260906)
cnt=collections.defaultdict(int); dm=collections.defaultdict(collections.Counter); tri=0
for r in rows:
    n=len(r['items'])
    ranks={'sim':sorted(range(n), key=lambda i:-r['sim'][i])[:5],
           'ft': sorted(range(n), key=lambda i:-r['ft'][i])[:5],
           'E':  sorted(range(n), key=lambda i:-r['E'][i])[:5],
           'J':  sorted(range(n), key=lambda i:-r['J'][i])[:5],
           '무작위': rng.sample(range(n),5)}
    for nm,o in ranks.items():
        cnt[nm]+=sum(1 for i in o if r['items'][i]['gr']>=2)
        for i in o: dm[nm][r['items'][i]['dom']]+=1
    if len({r['items'][i]['dom'] for i in ranks['ft']})==3: tri+=1
print("\n[Z1 혼합 규칙] 30후보 통합 top-5 (분모 100 = 20시드 × 5칸)")
print(f"  {'갈래':7s} {'적합률':>8s}   도메인 wn / steam / tmdb")
for nm in ('무작위','sim','J','E','ft'):
    print(f"  {nm:7s} {cnt[nm]:3d}/100 = {cnt[nm]:5.1f}%   "
          f"{dm[nm]['wn']:3d} / {dm[nm]['steam']:5d} / {dm[nm]['tmdb']:4d}")
d_rand=(cnt['ft']-cnt['무작위'])/100; d_sim=(cnt['ft']-cnt['sim'])/100
z1 = d_rand>=0.10 and d_sim>=0.10
print(f"  ft − 무작위 {100*d_rand:+.1f}%p (기준 +10%p) · ft − sim {100*d_sim:+.1f}%p (기준 +10%p)  ⇒ Z1 {'적중' if z1 else '빗나감'}")

# ---- Z2 가중치 축 ----
print("\n[Z2 가중치 축] 통합 30후보 중앙값(AUC_ft − AUC_E) ≥ +0.03")
if valid:
    d=[r['a_ft']-r['a_E'] for r in u]; p=wilcox(d); z2=med(d)>=0.03
    print(f"  AUC 중앙값 sim {med([r['a_sim'] for r in u]):.3f} · ft {med([r['a_ft'] for r in u]):.3f} · "
          f"E {med([r['a_E'] for r in u]):.3f} · J {med([r['a_J'] for r in u]):.3f}")
    print(f"  ft − E 중앙값 {med(d):+.3f}  (참고 p={p:.4f})  ⇒ Z2 {'적중' if z2 else '빗나감'}")
else: print("  통합 조건 무효 — 해석 금지")

# ---- Z3 존립 ----
wn4=sum(1 for r in rows if dm and sum(1 for i in sorted(range(30), key=lambda j:-r['ft'][j])[:5]
                                       if r['items'][i]['dom']=='wn')>=4)
print(f"\n[Z3 존립] ft top-5 에서 웹소설이 4칸 이상인 시드 {wn4}/20")
if wn4>=15: z3="확정 — 품질만으로 세우면 크로스 플랫폼은 일어나지 않는다. 명시적 도메인 쿼터 필수"
elif wn4<=5: z3="fun_tag 이 도메인을 자연스럽게 섞는다 — 쿼터 불필요"
else: z3=f"판정 보류(6~14 구간). 수치만 기록: {wn4}/20"
print(f"  ⇒ {z3}")
print(f"  참고: ft top-5 에 세 도메인이 모두 들어간 시드 {tri}/20")

# ---- Z4 내부 우위 ----
wn=[grade[(s,'wn',c)] for s in seed_tag for c in range(10)]
cross=[grade[(s,d,c)] for s in seed_tag for d in ('steam','tmdb') for c in range(10)]
rw, rc = sum(x>=2 for x in wn)/len(wn), sum(x>=2 for x in cross)/len(cross)
z4=(rw-rc)>=0.20
print(f"\n[Z4 내부 우위] 웹소설 {100*rw:.1f}% − 교차평균 {100*rc:.1f}% = {100*(rw-rc):+.1f}%p (기준 +20%p)  ⇒ {'적중' if z4 else '빗나감'}")

# ---- Z6 기술 통계 ----
print("\n[Z6 기술 통계] 도메인별 top-1 적합률 (분모 20시드)")
for dom in DOMS:
    k=sum(1 for s in seed_tag if grade[(s,dom,0)]>=2)
    print(f"  {dom:6s} 유사도 1위 {k}/20 = {100*k/20:5.1f}%")

# ---- Z6 추가 (기각 조건 없음) : 도메인 다양성 · 사후 기술용 쿼터 갈래 ----
print("\n[Z6 추가] 도메인 다양성 (분모 20시드)")
for nm,key in (('sim','sim'),('E','E'),('ft','ft'),('J','J')):
    t3=t2=0
    for r in rows:
        o=sorted(range(30), key=lambda j:-r[key][j])[:5]
        d={r['items'][i]['dom'] for i in o}
        t3 += len(d)==3; t2 += len(d)>=2
    print(f"  {nm:4s} 세 도메인 모두 {t3:2d}/20 · 두 도메인 이상 {t2:2d}/20")

print("\n[사후 기술 — 등록 안 됨, h52 에서 사전 등록 대상] 도메인 쿼터 갈래")
print("  규칙: 각 도메인에서 ft 상위 1개씩(3칸) + 나머지 2칸은 전체 ft 순")
q=0; qd=collections.Counter()
for r in rows:
    picked=[]
    for dom in DOMS:
        idx=[i for i in range(30) if r['items'][i]['dom']==dom]
        picked.append(max(idx, key=lambda i:r['ft'][i]))
    rest=sorted([i for i in range(30) if i not in picked], key=lambda i:-r['ft'][i])[:2]
    for i in picked+rest:
        q += r['items'][i]['gr']>=2; qd[r['items'][i]['dom']]+=1
print(f"  쿼터 적합 {q}/100 = {q:.1f}%   도메인 wn {qd['wn']} / steam {qd['steam']} / tmdb {qd['tmdb']}")
print(f"  (비교: ft 92.0% wn65/st18/tm17 · E 92.0% wn51/st22/tm27 · 무작위 71.0%)")
