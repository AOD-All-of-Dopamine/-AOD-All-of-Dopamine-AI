"""h49-B 판정 — 사전 등록 md5 2451fc6f32c8be0702952f902dec6b42.
교차(웹소설→Steam, h48 신규 20시드) vs 내부(웹소설→웹소설, 동일 20시드)."""
import json, math, collections, random, sys
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h48_grades import G as G_X
from h48f_labels import L as L_FR
from h49w_labels import L as L_WN
from h49w_grades import G as G_W

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
    t=sum(1.0 if s[p]>s[n] else (0.5 if s[p]==s[n] else 0.0) for p in pos for n in neg)
    return t/(len(pos)*len(neg))
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

def build(seeds_lab, cands, grades):
    """seeds_lab: {seed:tags}, cands: {(seed,ci):tags}, grades: {(seed,ci):grade}"""
    rows=[]
    for s in sorted(seeds_lab):
        idx=sorted(ci for (sd,ci) in grades if sd==s)
        if not idx: continue
        st=seeds_lab[s]; ct=[cands[(s,i)] for i in idx]; gr=[grades[(s,i)] for i in idx]
        sim=[-i for i in idx]; ft=[cos(st,c) for c in ct]; J=[jac(st,c) for c in ct]
        E=[0.643*a+0.357*b for a,b in zip(pct(ft),pct(sim))]
        rows.append(dict(seed=s, idx=idx, gr=gr, st=st, ct=ct,
            a_sim=auc(sim,gr), a_ft=auc(ft,gr), a_E=auc(E,gr), a_J=auc(J,gr),
            sim_top=idx[sim.index(max(sim))], ft_top=idx[ft.index(max(ft))],
            E_top=idx[E.index(max(E))], J_top=(idx[J.index(max(J))] if max(J)>=0.34 else None),
            gate=max(J)>=0.34, best=max(gr)))
    return rows

# ---- 교차 (h48 신규 20시드) ----
fk=json.load(open(f'{SP}/h48f_key.json')); gk=json.load(open(f'{SP}/h48_grade_key.json'))
lab_fr={(k['seed'],k['cand']):L_FR[k['id']] for k in fk}
Xs={s:lab_fr[(s,None)] for (s,c) in lab_fr if c is None}
Xc={(s,c):t for (s,c),t in lab_fr.items() if c is not None}
Xg={(int(k['seed']),k['cand']):G_X[k['pid']] for k in gk if k['origin']!='g6'}
cross=build(Xs,Xc,Xg)

# ---- 내부 (동일 20시드, 웹소설→웹소설) ----
wk=json.load(open(f'{SP}/h49w_key.json')); wgk=json.load(open(f'{SP}/h49w_grade_key.json'))
Wc={(k['seed'],k['cand']):L_WN[k['id']] for k in wk}
Wg={(k['seed'],k['cand']):G_W[k['pid']] for k in wgk}
within=build(Xs,Wc,Wg)   # 시드 태그는 동일 시드이므로 교차와 같은 라벨을 쓴다

print("="*76); print("h49-B 판정 — 교차(웹소설→Steam) vs 내부(웹소설→웹소설), 동일 20시드"); print("="*76)
for nm,rs in (('교차',cross),('내부',within)):
    u=[r for r in rs if r['a_sim'] is not None]
    print(f"{nm}: 시드 {len(rs)} · AUC 계산가능 {len(u)} · 적합률 "
          f"{sum(1 for r in rs for g in r['gr'] if g>=2)}/{sum(len(r['gr']) for r in rs)}")

# ---- X3 건전성 (S1 가드, 본페로니 α=0.0125) ----
print("\n[X3 건전성] S1 가드 — 네 점수 중 최소 하나의 AUC 중앙값이 유의하게 > 0.5 (α=0.0125)")
def s1(rs, nm):
    u=[r for r in rs if r['a_sim'] is not None]; ok=False
    for lab,k in (('sim','a_sim'),('ft','a_ft'),('E','a_E'),('J','a_J')):
        v=[r[k] for r in u]; p=wilcox([x-0.5 for x in v]); hit=med(v)>0.5 and p<0.0125
        ok |= hit
        print(f"   {nm} {lab:4s} 중앙값 {med(v):.3f}  p={p:.4f}  {'통과' if hit else '—'}")
    return ok
s1c=s1(cross,'교차'); s1w=s1(within,'내부')
print(f"   → 교차 {'통과' if s1c else '탈락'} · 내부 {'통과' if s1w else '탈락'}  ⇒ X3 {'적중' if s1w else '빗나감'}")

# ---- X1 핵심 ----
pair=[(c,w) for c in cross for w in within if c['seed']==w['seed']
      and c['a_sim'] is not None and w['a_sim'] is not None]
dX=[(c['a_ft']-c['a_sim'])-(w['a_ft']-w['a_sim']) for c,w in pair]
mc=med([c['a_ft']-c['a_sim'] for c,w in pair]); mw=med([w['a_ft']-w['a_sim'] for c,w in pair])
p1=wilcox(dX); x1=med(dX)>=0.10 and p1<0.05
print(f"\n[X1 핵심] 짝지은 시드 {len(pair)}개")
print(f"   Δ_cross  = AUC_ft − AUC_sim (교차) 중앙값 {mc:+.3f}")
print(f"   Δ_within = AUC_ft − AUC_sim (내부) 중앙값 {mw:+.3f}")
print(f"   중앙값(Δ_cross − Δ_within) {med(dX):+.3f} (기준 +0.10)  p={p1:.4f}  {'적중' if x1 else '빗나감'}")

# ---- X2 임베딩 열화 ----
msw=med([w['a_sim'] for c,w in pair]); msc=med([c['a_sim'] for c,w in pair])
x2=(msw-msc)>=0.10
print(f"\n[X2 임베딩 열화] AUC_sim 내부 {msw:.3f} − 교차 {msc:.3f} = {msw-msc:+.3f} (기준 +0.10)  {'적중' if x2 else '빗나감'}")
p2=wilcox([w['a_sim']-c['a_sim'] for c,w in pair]); print(f"   (참고 부호순위 p={p2:.4f})")

# ---- X4 기술 통계 ----
print("\n[X4 기술 통계 — 기각 조건 없음] top-1 적합률 (분모 = 20시드)")
rng=random.Random(20260819)
print(f"   {'갈래':14s} {'교차':>10s} {'내부':>10s}")
for nm,f in (('A 현행 게이트',lambda r:r['J_top']),('B 유사도 1위',lambda r:r['sim_top']),
             ('C 무작위',lambda r:rng.choice(r['idx'])),('E 기획서 원안',lambda r:r['E_top']),
             ('ft 단독',lambda r:r['ft_top'])):
    out=[]
    for rs in (cross,within):
        k=sum(1 for r in rs if (i:=f(r)) is not None and r['gr'][r['idx'].index(i)]>=2)
        out.append(f"{k}/{len(rs)} {100*k/len(rs):5.1f}%")
    print(f"   {nm:14s} {out[0]:>12s} {out[1]:>12s}")
for nm,rs in (('교차',cross),('내부',within)):
    gp=[r for r in rs if r['gate']]
    print(f"   {nm} 게이트 통과 {len(gp)}/{len(rs)} · 적합후보 보유 시드 {sum(1 for r in rs if r['best']>=2)}/{len(rs)}")
