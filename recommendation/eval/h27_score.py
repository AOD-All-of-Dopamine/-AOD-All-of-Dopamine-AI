"""H27 — 7차 채점 + 사전 등록(h26, md5 b914a967…) 판정."""
import sys, math, statistics
sys.path.insert(0,'.')
from h27_funtags import S
G=[[1,0,3,2,3,2,2,1,1,0],[2,1,2,3,3,0,3,1,1,0],[1,0,2,1,0,0,0,1,0,0],
   [2,1,1,0,3,0,2,0,0,1],[2,2,0,0,0,0,0,2,2,1],[0,1,1,0,3,1,2,2,1,3],
   [1,1,1,0,0,2,1,3,3,1],[2,1,1,0,1,0,0,0,1,1],[1,2,1,0,0,2,1,1,2,0],
   [0,2,0,1,0,0,0,1,1,1],[1,0,2,0,0,0,2,0,0,2],[1,2,1,0,1,0,0,0,0,0]]
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
    return sel,ranks
print(f"{'시드':28s} {'top-3':>8s} {'top-10':>9s} {'선택':>6s}")
for (nm,stg,cts),gs in zip(S,G):
    J3=[jac(stg,c) for c in cts[:3]]; J10=[jac(stg,c) for c in cts]
    b3=max(range(3),key=lambda i:J3[i]); b10=max(range(10),key=lambda i:J10[i])
    f=lambda J,b,g: ("통과 "+("O" if g>=2 else "X")) if J>=0.34 else "차단"
    print(f"{nm[:26]:28s} {f(J3[b3],b3,gs[b3]):>8s} {f(J10[b10],b10,gs[b10]):>9s} {b10+1:4d}위 J={J10[b10]:.2f}")
res={}
for K in (3,10):
    sel,ranks=run(K); k,n=sum(1 for g in sel if g>=2),len(sel); lo,hi=wilson(k,n)
    res[K]=(k,n,ranks)
    print(f"\n=== top-{K} ===  통과 {n}/12 = {100*n/12:.0f}%   적합 {k}/{n} = {100*k/n if n else 0:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
    if ranks: print(f"  선택 순위 {sorted(ranks)}")
k3,n3,_=res[3]; k10,n10,r10=res[10]
r3rate=k3/n3 if n3 else 0; r10rate=k10/n10 if n10 else 0
deep=sum(1 for r in r10 if r>=4)/len(r10) if r10 else 0
t1=n10>n3; t2=r10rate>=r3rate; t3=deep>=0.25; t4=r10rate>=0.70
print("\n=== 사전 등록 판정 ===")
print(f"  T1 top-10 통과율 > top-3      {n10}/12 vs {n3}/12          {'적중' if t1 else '빗나감'}")
print(f"  T2 top-10 적합률 >= top-3     {100*r10rate:.1f}% vs {100*r3rate:.1f}%     {'적중' if t2 else '빗나감'}")
print(f"  T3 4위 이하 선택 >= 25%       {100*deep:.0f}%             {'적중' if t3 else '빗나감'}")
print(f"  T4 top-10 적합률 >= 70%       {100*r10rate:.1f}%           {'적중' if t4 else '빗나감'}")
print(f"\n  기각 조건: " + ("발동 — '풀 확대는 커버리지와 품질을 동시에 올린다'를 버린다" if not(t1 and t2)
      else ("T3 빗나감 — 풀 확대 효과의 원인을 다시 찾아야 한다" if not t3 else "미발동")))
# 6+7차 top-10 풀링
print("\n=== 6+7차 top-10 풀링 (24시드) ===")
K,N=7+k10,9+n10; lo,hi=wilson(K,N)
print(f"  통과 {N}/24 = {100*N/24:.0f}%   적합 {K}/{N} = {100*K/N:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
K3,N3=4+k3,6+n3; lo,hi=wilson(K3,N3)
print(f"  같은 24시드 top-3:  통과 {N3}/24 = {100*N3/24:.0f}%   적합 {K3}/{N3} = {100*K3/N3:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
