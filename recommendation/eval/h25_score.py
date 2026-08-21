"""H25 — 6차 채점 + 사전 등록(h24, md5 f6f54477…) 판정. 후보 풀 top-10."""
import sys, json, math, statistics
sys.path.insert(0,'.')
from h25_funtags import S
G=[[1,1,2,3,1,3,2,0,2,2],[0,0,2,1,1,1,1,1,1,1],[1,2,1,2,2,1,2,2,0,2],
   [0,1,1,2,0,1,1,2,0,0],[2,2,0,1,2,2,0,1,2,0],[1,1,2,2,2,2,1,1,1,1],
   [1,1,0,2,0,0,1,2,0,0],[1,1,0,0,2,0,0,1,0,1],[1,3,0,0,1,1,1,0,0,0],
   [1,1,0,0,1,1,2,0,1,0],[1,2,1,3,0,0,0,1,0,1],[1,0,2,2,1,1,0,0,0,1]]
assert len(G)==12 and all(len(g)==10 for g in G)
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
def run(K):
    sel=[]; ranks=[]
    for (nm,stg,cts),gs in zip(S,G):
        J=[jac(stg,c) for c in cts[:K]]
        bi=max(range(K),key=lambda i:J[i])
        if J[bi]>=0.34: sel.append(gs[bi]); ranks.append(bi+1)
    return sel, ranks
print(f"{'시드':26s} {'top-3 게이트':>12s} {'top-10 게이트':>14s} {'선택 순위':>8s}")
for (nm,stg,cts),gs in zip(S,G):
    J3=[jac(stg,c) for c in cts[:3]]; J10=[jac(stg,c) for c in cts]
    b3=max(range(3),key=lambda i:J3[i]); b10=max(range(10),key=lambda i:J10[i])
    t3="통과 "+("O" if gs[b3]>=2 else "X") if J3[b3]>=0.34 else "차단"
    t10="통과 "+("O" if gs[b10]>=2 else "X") if J10[b10]>=0.34 else "차단"
    print(f"{nm[:24]:26s} {t3:>12s} {t10:>14s} {b10+1:6d}위  J={J10[b10]:.2f}")
for K in (3,10):
    sel,ranks=run(K); k,n=sum(1 for g in sel if g>=2),len(sel); lo,hi=wilson(k,n)
    print(f"\n=== 후보 풀 top-{K} ===")
    print(f"  게이트 통과 {n}/12 = {100*n/12:.0f}%   적합률 {k}/{n} = {100*k/n if n else 0:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
    if ranks: print(f"  선택된 후보의 유사도 순위: {sorted(ranks)}  중앙값 {statistics.median(ranks)}")
sel3,_=run(3); sel10,r10=run(10)
print("\n=== 사전 등록 판정 ===")
r1=len(sel10)/12>=0.60; r2=(sum(1 for g in sel10 if g>=2)/len(sel10) if sel10 else 0)>=0.70
r3=statistics.median(r10)>3 if r10 else False
print(f"  R1 top-10 통과율 >=60%       관측 {100*len(sel10)/12:.0f}%        {'적중' if r1 else '빗나감'}")
print(f"  R2 통과 후 적합률 >=70%       관측 {100*sum(1 for g in sel10 if g>=2)/len(sel10) if sel10 else 0:.1f}%      {'적중' if r2 else '빗나감'}")
print(f"  R3 선택 순위 중앙값 > 3위     관측 {statistics.median(r10) if r10 else 0}위       {'적중' if r3 else '빗나감'}")
print(f"\n  기각 조건: {'R2 빗나감 → 게이트는 top-3 좁은 풀에 기대고 있었다' if not r2 else '미발동 (R2 유지)'}")
if not r3: print("  R3 빗나감 → 풀 확대가 실제로 다른 선택을 만들지 못했다")
