"""h48 판정 — 사전 등록 md5 ec7e59236ae528fd115d51d6e9dc7f5f (보정 2)."""
import json, math, collections, random, sys
sys.path.insert(0,'/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad')
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
from h48_grades import G
from g6_relabel import L as L_G6
from h48f_labels import L as L_FR

# ---- 시드별 (라벨, 등급) 조립 ----
gk = json.load(open(f'{SP}/h48_grade_key.json'))
g6k = json.load(open(f'{SP}/g6_key.json')); fk = json.load(open(f'{SP}/h48f_key.json'))
lab_g6 = {(k['batch'], k['seed'], k['cand']): L_G6[k['id']] for k in g6k}
lab_fr = {(k['seed'], k['cand']): L_FR[k['id']] for k in fk}

seeds = collections.defaultdict(dict)      # key -> {cand: (tags, grade)}, [None] = 시드 태그
for k in gk:
    org, sk, ci = k['origin'], k['seed'], k['cand']
    if org == 'g6':
        b, s = sk.split('#'); s = int(s)
        st, ct = lab_g6[(b, s, None)], lab_g6[(b, s, ci)]
    else:
        s = int(sk); st, ct = lab_fr[(s, None)], lab_fr[(s, ci)]
    key = (org, sk)
    seeds[key][None] = st
    seeds[key][ci] = (ct, G[k['pid']])

def jac(a,b):
    X,Y=set(a),set(b); return len(X&Y)/len(X|Y) if X|Y else 0.0
def cos(a,b):
    X,Y=set(a),set(b); return len(X&Y)/math.sqrt(len(X)*len(Y)) if X and Y else 0.0
def pct(v):
    n=len(v); o=sorted(range(n),key=lambda i:v[i]); r=[0.0]*n; i=0
    while i<n:
        j=i
        while j+1<n and v[o[j+1]]==v[o[i]]: j+=1
        for k2 in range(i,j+1): r[o[k2]]=((i+j)/2)/(n-1)
        i=j+1
    return r
def auc(scores, grades):
    """적합(>=2) 후보가 부적합보다 위에 오는 비율. 동점 0.5. 대비 없으면 None."""
    pos=[i for i,g in enumerate(grades) if g>=2]; neg=[i for i,g in enumerate(grades) if g<2]
    if not pos or not neg: return None
    t=0.0
    for p in pos:
        for n in neg:
            t += 1.0 if scores[p]>scores[n] else (0.5 if scores[p]==scores[n] else 0.0)
    return t/(len(pos)*len(neg))

rows=[]
for key, d in sorted(seeds.items()):
    st = d[None]; idx = sorted(i for i in d if i is not None)
    ct = [d[i][0] for i in idx]; gr = [d[i][1] for i in idx]
    sim  = [-i for i in idx]                       # 인덱스 작을수록 유사도 높음
    ft   = [cos(st,c) for c in ct]
    J    = [jac(st,c) for c in ct]
    E    = [0.643*a+0.357*b for a,b in zip(pct(ft), pct(sim))]
    rows.append(dict(key=key, org=key[0], idx=idx, gr=gr, st=st, ct=ct, J=J,
                     a_sim=auc(sim,gr), a_ft=auc(ft,gr), a_E=auc(E,gr), a_J=auc(J,gr),
                     sim_top=idx[sim.index(max(sim))], ft_top=idx[ft.index(max(ft))],
                     E_top=idx[E.index(max(E))],
                     J_top=(idx[J.index(max(J))] if max(J)>=0.34 else None),
                     gate=max(J)>=0.34, best=max(gr)))
usable=[r for r in rows if r['a_sim'] is not None]
print("="*74); print("h48 판정 — 40시드 x 10후보 = 400쌍 눈가림 채점"); print("="*74)
print(f"시드 {len(rows)} · AUC 계산 가능 {len(usable)} (전부 적합/전부 부적합 {len(rows)-len(usable)}개 제외)")

def med(v): 
    s=sorted(v); n=len(s); return (s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2)
def wilcox(d):
    d=[x for x in d if x!=0]; n=len(d)
    if n<6: return 1.0
    o=sorted(range(n),key=lambda i:abs(d[i])); rk=[0]*n; i=0
    while i<n:
        j=i
        while j+1<n and abs(d[o[j+1]])==abs(d[o[i]]): j+=1
        for k2 in range(i,j+1): rk[o[k2]]=(i+j)/2+1
        i=j+1
    W=sum(rk[i] for i in range(n) if d[i]>0)
    mu=n*(n+1)/4; sd=(n*(n+1)*(2*n+1)/24)**.5
    from math import erfc
    return erfc(abs((W-mu)/sd)/2**.5)

print("\n[AUC] 시드별 중앙값")
for nm,k in (('sim (유사도)','a_sim'),('ft (fun_tag 코사인)','a_ft'),('E (기획서 혼합)','a_E'),('J (자카드)','a_J')):
    v=[r[k] for r in usable]; print(f"  {nm:20s} {med(v):.3f}   평균 {sum(v)/len(v):.3f}")

dE=[r['a_E']-r['a_sim'] for r in usable]
dF=[r['a_ft']-0.5 for r in usable]
dJ=[r['a_ft']-r['a_J'] for r in usable]
dS=[r['a_sim']-0.5 for r in usable]
print("\n" + "="*74); print("사전 등록 판정"); print("="*74)
def rep(tag, txt, val, thr, p, ok):
    print(f"  {tag} {txt:44s} {val:+.3f} (기준 {thr:+.2f})  p={p:.3f}  {'적중' if ok else '빗나감'}")
h1 = med(dE)>=0.05 and wilcox(dE)<0.05
rep('H1','E − sim  중앙값', med(dE), 0.05, wilcox(dE), h1)
h2 = med([r['a_ft'] for r in usable])>=0.55 and wilcox(dF)<0.05
rep('H2','ft − 0.5  중앙값', med(dF), 0.05, wilcox(dF), h2)
h4 = med(dJ)>=0.03
rep('H4','ft − J   중앙값', med(dJ), 0.03, wilcox(dJ), h4)
h5 = med([r['a_sim'] for r in usable])>0.5 and wilcox(dS)<0.05
rep('H5','sim − 0.5 중앙값 (건전성)', med(dS), 0.00, wilcox(dS), h5)

gp=[r for r in rows if r['gate']]; gf=[r for r in rows if not r['gate']]
def anyfit(rs): return sum(1 for r in rs if r['best']>=2)/len(rs) if rs else 0
h3 = (anyfit(gp)-anyfit(gf))>=0.20
print(f"  H3 게이트 통과 시드의 '적합 후보 보유율' {100*anyfit(gp):.1f}% ({len(gp)}시드)"
      f" vs 탈락 {100*anyfit(gf):.1f}% ({len(gf)}시드)  차이 {100*(anyfit(gp)-anyfit(gf)):+.1f}%p  {'적중' if h3 else '빗나감'}")

fr=[r for r in usable if r['org']=='fresh']; g6=[r for r in usable if r['org']=='g6']
mf, mg = med([r['a_E']-r['a_sim'] for r in fr]), med([r['a_E']-r['a_sim'] for r in g6])
h6 = (mf>0)==(mg>0)
print(f"  H6 오염 점검 — H1 부호 일치: 신규 {mf:+.3f} · 재사용 {mg:+.3f}  {'적중' if h6 else '빗나감'}")

pairs=[(r['st'],c) for r in rows for c in r['ct']]
b=collections.defaultdict(lambda:[0,0])
for s,c in pairs:
    one = min(len(set(s)),len(set(c)))==1
    b[one][0]+=1; b[one][1]+= jac(s,c)>=0.34
r1 = b[True][1]/b[True][0] if b[True][0] else 0
r2 = b[False][1]/b[False][0] if b[False][0] else 0
print(f"  G5 태그1개 포함 쌍 통과율 {100*r1:.1f}% ({b[True][0]}쌍) vs 양쪽2개+ {100*r2:.1f}% ({b[False][0]}쌍)"
      f"  차이 {100*(r1-r2):+.1f}%p  {'적중' if (r1-r2)>=0.20 else '빗나감'}")

print("\n[기술 통계 — 기각 조건 없음] top-1 적합률 (분모 = 40시드)")
rng=random.Random(20260819)
for nm,f in (('A 현행 게이트', lambda r: r['J_top']),
             ('B 유사도 1위',  lambda r: r['sim_top']),
             ('C 무작위',      lambda r: rng.choice(r['idx'])),
             ('E 기획서 원안', lambda r: r['E_top']),
             ('ft 단독',       lambda r: r['ft_top'])):
    k=0
    for r in rows:
        i=f(r)
        if i is not None and r['gr'][r['idx'].index(i)]>=2: k+=1
    print(f"  {nm:12s} {k}/40 = {100*k/40:5.1f}%")
print(f"  게이트 통과 시드 {len(gp)}/40 = {100*len(gp)/40:.0f}%")
