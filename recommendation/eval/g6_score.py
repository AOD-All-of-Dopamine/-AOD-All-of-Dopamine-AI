"""G6 판정 — 전문 재라벨링 vs 40자 절단 라벨.

사전 등록 h48 보정1 (md5 74f4330a280c56af9f78eeda281f67f4):
  G6. fun_tag 이 바뀌는 아이템이 25% 이상이면 적중
      → 기존 배치의 라벨 기반 결론을 '열화 조건 측정'으로 재분류
      → 빗나가면 절단의 영향이 제한적이었다는 뜻이므로 기존 결론 유지
"""
import json, sys, collections, math
sys.path.insert(0,'/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad')
from g6_relabel import L as NEW
import h41_funtags as A, h43_funtags as B, h45_funtags as C

def norm(S, mg):
    o=[]
    for r in S:
        if mg: g,n,st,sk,m,cs = r
        else:  g,n,st,sk,cs = r
        o.append((list(st), [list(x[0]) for x in cs]))
    return o
OLD = {'15차': norm(A.S,0), '16차': norm(B.S,1), '18차': norm(C.S,1)}

key = json.load(open(f'{__import__("os").path.dirname(__file__) or "."}/g6_key.json')) \
      if False else json.load(open('/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad/g6_key.json'))

def jac(a,b):
    X,Y=set(a),set(b); return len(X&Y)/len(X|Y) if X|Y else 1.0

rows=[]
for k in key:
    st, cs = OLD[k['batch']][k['seed']]
    old = st if k['cand'] is None else cs[k['cand']]
    new = NEW[k['id']]
    rows.append((k, old, new))

same = sum(1 for _,o,n in rows if set(o)==set(n))
chg  = len(rows)-same
J    = [jac(o,n) for _,o,n in rows]
print("="*70)
print("G6 판정 — 전문 재라벨링 vs 40자 절단 라벨")
print("="*70)
print(f"\n항목 {len(rows)}건")
print(f"  라벨 완전 동일   {same:3d}건 ({100*same/len(rows):.1f}%)")
print(f"  라벨 변경        {chg:3d}건 ({100*chg/len(rows):.1f}%)")
print(f"  옛↔새 자카드 평균 {sum(J)/len(J):.3f}  ·  중앙 {sorted(J)[len(J)//2]:.3f}")

seeds = [(k,o,n) for k,o,n in rows if k['cand'] is None]
cands = [(k,o,n) for k,o,n in rows if k['cand'] is not None]
for nm, grp in (('시드(웹소설)', seeds), ('후보(Steam)', cands)):
    s = sum(1 for _,o,n in grp if set(o)==set(n))
    print(f"  {nm:14s} 변경 {len(grp)-s:3d}/{len(grp):3d} = {100*(len(grp)-s)/len(grp):5.1f}%")

# 축별로 얼마나 새로 붙고/떨어졌나
add = collections.Counter(); rm = collections.Counter()
for _,o,n in rows:
    for t in set(n)-set(o): add[t]+=1
    for t in set(o)-set(n): rm[t]+=1
print(f"\n축별 증감 (전문으로 보니 새로 붙은 건수 / 떨어진 건수)")
AX = '성장역전 사이다응징 피폐고통 관계로맨스 전략운영 탐험발견 생존긴장 수집육성 유머경쾌 미스터리추리 세계관로어 잔혹공포 힐링평온 성적노출'.split()
for t in sorted(AX, key=lambda x:-(add[x]+rm[x])):
    print(f"  {t:6s} +{add[t]:3d}  -{rm[t]:3d}   순 {add[t]-rm[t]:+4d}")

# ---- 하류 효과: 게이트 판정이 얼마나 바뀌는가 ----
print(f"\n하류 효과 — 같은 시드/후보 쌍에서 게이트(J>=0.34) 판정 변화")
byseed = collections.defaultdict(dict)
for k,o,n in rows:
    byseed[(k['batch'],k['seed'])][k['cand']] = (o,n)
p2f=f2p=0; both=0; sel_chg=0
for kk, d in byseed.items():
    so,sn = d[None]
    for ci in range(10):
        if ci not in d: continue
        co,cn = d[ci]
        a = jac(so,co)>=0.34; b = jac(sn,cn)>=0.34
        both+=1
        if a and not b: p2f+=1
        elif b and not a: f2p+=1
    # 선택(최대 겹침 후보)이 바뀌었나
    idx = sorted(i for i in d if i is not None)
    Jo=[jac(so,d[i][0]) for i in idx]; Jn=[jac(sn,d[i][1]) for i in idx]
    if idx and (idx[Jo.index(max(Jo))] != idx[Jn.index(max(Jn))]): sel_chg+=1
print(f"  쌍 {both}건 중 통과→탈락 {p2f}건 · 탈락→통과 {f2p}건  (총 {p2f+f2p}건 = {100*(p2f+f2p)/both:.1f}%)")
print(f"  최대겹침 후보 선택이 바뀐 시드 {sel_chg}/20")

r = chg/len(rows)
print("\n" + "="*70)
print(f"  G6  라벨 변경 {100*r:.1f}%  (기준 25%)   {'적중' if r>=0.25 else '빗나감'}")
if r>=0.25:
    print("  ** 기존 배치의 라벨 기반 결론을 '열화 조건에서의 측정'으로 재분류한다.")
    print("     기각된 8건 중 라벨 의존적인 것들을 재검토 대상으로 올린다.")
else:
    print("  절단의 영향이 제한적 — 기존 결론을 유지한다.")
