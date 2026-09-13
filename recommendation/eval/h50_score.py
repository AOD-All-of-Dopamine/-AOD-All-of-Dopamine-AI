"""h50 판정 — 사전 등록 md5 3a72b9be3a1378e356b9a1927abd52f2. 라벨 md5 ad966719d46abcf162abed9a579632e4."""
import json, math, collections, random, sys
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_labels import L
from h50_grades import G

key = json.load(open(f'{SP}/h50_key.json'))
gk  = json.load(open(f'{SP}/h50_grade_key.json'))
seed_tag = {k['seed']: L[k['id']] for k in key if k['kind']=='seed'}
cand_tag = {(k['seed'], k['dom'], k['cand']): L[k['id']] for k in key if k['kind']=='cand'}
grade    = {(k['seed'], k['dom'], k['cand']): G[k['pid']] for k in gk}

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

def build(dom):
    rows=[]
    for s in sorted(seed_tag):
        idx=sorted(c for (sd,dm,c) in grade if sd==s and dm==dom)
        st=seed_tag[s]; ct=[cand_tag[(s,dom,i)] for i in idx]; gr=[grade[(s,dom,i)] for i in idx]
        sim=[-i for i in idx]; ft=[cos(st,c) for c in ct]; J=[jac(st,c) for c in ct]
        E=[0.643*a+0.357*b for a,b in zip(pct(ft),pct(sim))]
        rows.append(dict(seed=s, idx=idx, gr=gr, st=st, ct=ct, ft=ft, E=E, sim=sim, J=J,
            a_sim=auc(sim,gr), a_ft=auc(ft,gr), a_E=auc(E,gr), a_J=auc(J,gr),
            sim_top=idx[sim.index(max(sim))], ft_top=idx[ft.index(max(ft))],
            E_top=idx[E.index(max(E))], J_top=(idx[J.index(max(J))] if max(J)>=0.34 else None),
            gate=max(J)>=0.34, best=max(gr)))
    return rows
S, T = build('steam'), build('tmdb')

print("="*76); print("h50 판정 — 웹소설 신규 20시드 → Steam · TMDB, 400쌍 눈가림 채점"); print("="*76)
print(f"등급 분포 {dict(sorted(collections.Counter(G.values()).items()))}")
for nm,rs in (('Steam',S),('TMDB',T)):
    fit=sum(1 for r in rs for g in r['gr'] if g>=2)
    print(f"{nm}: 적합 {fit}/200 = {fit/2:.1f}% · AUC 계산가능 {sum(1 for r in rs if r['a_sim'] is not None)}/20")

print("\n[Y4 가드]")
valid={}
for nm,rs in (('Steam',S),('TMDB',T)):
    u=[r for r in rs if r['a_sim'] is not None]
    y4a = len(u)>=12
    print(f"  {nm} Y4a 대비: AUC 계산가능 {len(u)}/20 (기준 ≥12)  {'통과' if y4a else '탈락'}")
    y4b=False
    for lab,k in (('sim','a_sim'),('ft','a_ft'),('E','a_E'),('J','a_J')):
        v=[r[k] for r in u]; p=wilcox([x-0.5 for x in v]); hit=med(v)>0.5 and p<0.0125
        y4b |= hit
        print(f"    {lab:4s} 중앙값 {med(v):.3f}  p={p:.4f}  {'통과' if hit else '—'}")
    print(f"  {nm} Y4b 신호: {'통과' if y4b else '탈락'}  ⇒ 조건 {'유효' if (y4a and y4b) else '무효'}")
    valid[nm]=y4a and y4b

print("\n[Y1 확증 · Steam] / [Y2 일반화 · TMDB]  중앙값(AUC_E − AUC_sim) ≥ +0.05, p<0.05")
for nm,rs,tag in (('Steam',S,'Y1'),('TMDB',T,'Y2')):
    if not valid[nm]: print(f"  {tag} {nm}: 조건 무효 — 해석 금지"); continue
    u=[r for r in rs if r['a_sim'] is not None]
    d=[r['a_E']-r['a_sim'] for r in u]; p=wilcox(d); hit=med(d)>=0.05 and p<0.05
    print(f"  {tag} {nm}: 중앙값 {med(d):+.3f}  p={p:.4f}  n={len(u)}  {'적중' if hit else '빗나감'}"
          f"   (AUC_sim {med([r['a_sim'] for r in u]):.3f} · AUC_ft {med([r['a_ft'] for r in u]):.3f} · AUC_E {med([r['a_E'] for r in u]):.3f})")

print("\n[Y3 3도메인 혼합] Steam10 + TMDB10 통합 재순위 top-5")
def merged(r_s, r_t):
    items=[]
    for r,dom in ((r_s,'steam'),(r_t,'tmdb')):
        for pos,i in enumerate(r['idx']):
            items.append(dict(dom=dom, gr=r['gr'][pos], ft=r['ft'][pos], rank=i))
    sim_all=[-x['rank'] for x in items]      # 도메인 내 순위를 그대로 (동순위는 동점)
    ft_all=[x['ft'] for x in items]
    E_all=[0.643*a+0.357*b for a,b in zip(pct(ft_all), pct(sim_all))]
    order_sim=sorted(range(len(items)), key=lambda i:-sim_all[i])[:5]
    order_E  =sorted(range(len(items)), key=lambda i:-E_all[i])[:5]
    return items, order_sim, order_E
fs=fe=chg=0; dom_e=collections.Counter(); dom_s=collections.Counter()
for rs_, rt_ in zip(S, T):
    items, os_, oe_ = merged(rs_, rt_)
    fs += sum(1 for i in os_ if items[i]['gr']>=2)
    fe += sum(1 for i in oe_ if items[i]['gr']>=2)
    if set(os_)!=set(oe_): chg+=1
    for i in oe_: dom_e[items[i]['dom']]+=1
    for i in os_: dom_s[items[i]['dom']]+=1
rs_rate, re_rate = fs/100, fe/100
y3 = (re_rate-rs_rate)>=0.10 and chg>=8
print(f"  sim 재순위 top-5 적합률 {100*rs_rate:.1f}%  ·  E 재순위 {100*re_rate:.1f}%  차이 {100*(re_rate-rs_rate):+.1f}%p (기준 +10%p)")
print(f"  top-5 구성이 바뀐 시드 {chg}/20 (기준 ≥8)   ⇒ Y3 {'적중' if y3 else '빗나감'}")

print("\n[Y5 기술 통계 — 기각 조건 없음]")
print(f"  혼합 top-5 도메인 비율:  sim  Steam {dom_s['steam']} / TMDB {dom_s['tmdb']}"
      f"   ·  E  Steam {dom_e['steam']} / TMDB {dom_e['tmdb']}")
rng=random.Random(20260901)
print(f"  {'갈래':14s} {'Steam':>12s} {'TMDB':>12s}")
for nm,f in (('A 현행 게이트',lambda r:r['J_top']),('B 유사도 1위',lambda r:r['sim_top']),
             ('C 무작위',lambda r:rng.choice(r['idx'])),('E 기획서 원안',lambda r:r['E_top']),
             ('ft 단독',lambda r:r['ft_top'])):
    out=[]
    for rs in (S,T):
        k=sum(1 for r in rs if (i:=f(r)) is not None and r['gr'][r['idx'].index(i)]>=2)
        out.append(f"{k}/20 {100*k/20:5.1f}%")
    print(f"  {nm:14s} {out[0]:>14s} {out[1]:>14s}")
for nm,rs in (('Steam',S),('TMDB',T)):
    print(f"  {nm} 게이트 통과 {sum(1 for r in rs if r['gate'])}/20 · 적합후보 보유 {sum(1 for r in rs if r['best']>=2)}/20")

# ---- 기술 통계 추가 (기각 조건 없음): 혼합 재순위에서 ft 단독 · 무작위 ----
print("\n[Y5 추가] 혼합 top-5 갈래 비교 (분모 100 = 20시드 x 5칸)")
rng2=random.Random(20260901); rows=collections.defaultdict(int); dm=collections.defaultdict(collections.Counter)
for rs_, rt_ in zip(S, T):
    items, os_, oe_ = merged(rs_, rt_)
    ft_all=[x['ft'] for x in items]
    of_=sorted(range(len(items)), key=lambda i:-ft_all[i])[:5]
    oc_=rng2.sample(range(len(items)), 5)
    for nm,o in (('sim',os_),('ft',of_),('E',oe_),('무작위',oc_)):
        rows[nm]+=sum(1 for i in o if items[i]['gr']>=2)
        for i in o: dm[nm][items[i]['dom']]+=1
for nm in ('무작위','sim','ft','E'):
    print(f"  {nm:6s} 적합 {rows[nm]:3d}/100 = {rows[nm]:5.1f}%   도메인 Steam {dm[nm]['steam']:3d} / TMDB {dm[nm]['tmdb']:3d}")
