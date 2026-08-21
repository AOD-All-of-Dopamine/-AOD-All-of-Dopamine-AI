"""H29 — 8차 채점 + 사전 등록(h28, md5 67bfcbed…) 판정. 게이트 선택만 채점(선택은 fun_tag 로 확정 후 결정)."""
import sys, math
sys.path.insert(0,'.')
from h29_funtags import S
# 후보 인덱스(0-base) → 등급. 게이트가 고른 것만.
GR={0:{2:2, 6:3}, 1:{0:1, 8:2}, 2:{2:1, 13:1}, 3:{2:2, 12:1}, 4:{0:0, 4:1, 13:2}, 5:{1:1}}
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
def run(K):
    sel=[];ranks=[]
    for i,(nm,st,cts) in enumerate(S):
        J=[jac(st,c) for c in cts[:K]]
        b=max(range(K),key=lambda j:J[j])
        if J[b]>=0.34: sel.append(GR[i][b]); ranks.append(b+1)
    return sel,ranks
res={}
for K in (3,10,30):
    sel,ranks=run(K); k,n=sum(1 for g in sel if g>=2),len(sel); lo,hi=wilson(k,n)
    res[K]=(k,n,ranks)
    print(f"top-{K:2d}  통과 {n}/6 = {100*n/6:3.0f}%   적합 {k}/{n} = {100*k/n if n else 0:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]   순위 {sorted(ranks)}")
k10,n10,_=res[10]; k30,n30,ranks30=res[30]
r10=k10/n10 if n10 else 0; r30=k30/n30 if n30 else 0
deep=sum(1 for r in ranks30 if r>=11)/len(ranks30) if ranks30 else 0
u1=n30>=n10; u2=r30>=0.70; u3=r30>=r10-0.05; u4=deep>=0.5
print("\n=== 사전 등록 판정 ===")
print(f"  U1 top-30 통과율 >= top-10    {n30}/6 vs {n10}/6        {'적중' if u1 else '빗나감'}")
print(f"  U2 top-30 적합률 >= 70%       {100*r30:.1f}%            {'적중' if u2 else '빗나감'}")
print(f"  U3 top-30 >= top-10 - 5%p     {100*r30:.1f}% vs {100*r10:.1f}%   {'적중' if u3 else '빗나감'}")
print(f"  U4 11위 이하 선택 >= 50%      {100*deep:.0f}%              {'적중' if u4 else '빗나감'}")
print(f"\n  기각 조건: " + ("발동 — '넓힐수록 좋다'는 3~10 구간에만 성립. 운영 지점을 top-10 으로 고정한다."
      if (not u2 or not u3) else "미발동"))
