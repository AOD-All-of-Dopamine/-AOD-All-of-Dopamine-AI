"""H31 — 9차 채점 + 사전 등록(h30, md5 82b6c2ad…) 판정."""
import sys, math
sys.path.insert(0,'.')
from h31_funtags import S
# 시드별 {후보 인덱스(0-base): 등급}. 게이트 선택 + 동점 후보만.
GR=[{9:2},{5:1,6:1},{0:0,3:0,4:0,5:1,6:0,7:0,8:1},{6:1},{1:3,5:3,6:2},{0:1,2:2},
    {8:1},{3:2,5:1},{5:2},{7:1},{4:1},{3:2,8:1}]
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
passed=[]; tie_hi=[]; tie_lo=[]; nties=0
print(f"{'시드':24s} {'J':>5s} {'상태':>6s} {'선택':>5s} {'등급':>4s}")
for i,(nm,st,cts) in enumerate(S):
    J=[jac(st,c) for c in cts]; m=max(J)
    ties=[j for j,v in enumerate(J) if abs(v-m)<1e-9]
    if len(ties)>1: nties+=1
    b=ties[0]                      # 현행: 유사도 상위 우선
    ok=m>=0.34
    if ok:
        passed.append(GR[i][b])
        if len(ties)>1:
            tie_hi.append(GR[i][ties[0]]); tie_lo.append(GR[i][ties[-1]])
    print(f"{nm[:22]:24s} {m:5.2f} {'통과' if ok else '차단':>6s} {b+1:4d}위 {GR[i][b]:4d}"
          + (f"   동점 {len(ties)}개, 하위선택 {ties[-1]+1}위 등급 {GR[i][ties[-1]]}" if len(ties)>1 and ok else ""))
k,n=sum(1 for g in passed if g>=2),len(passed); lo,hi=wilson(k,n)
print(f"\n=== 9차 top-10 ===  통과 {n}/12 = {100*n/12:.0f}%   적합 {k}/{n} = {100*k/n:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
hr=sum(1 for g in tie_hi if g>=2)/len(tie_hi) if tie_hi else 0
lr=sum(1 for g in tie_lo if g>=2)/len(tie_lo) if tie_lo else 0
v1=nties>=3; v2=lr>=hr; v3=(k/n if n else 0)>=0.70
print(f"\n=== 사전 등록 판정 ===")
print(f"  V1 동점 시드 >= 3건          {nties}/12             {'적중' if v1 else '빗나감'}")
print(f"  V2 하위선택 적합률 >= 상위     {100*lr:.0f}% vs {100*hr:.0f}% (n={len(tie_hi)})  {'적중' if v2 else '빗나감'}")
print(f"  V3 통과 후 적합률 >= 70%      {100*k/n:.1f}%          {'적중' if v3 else '빗나감'}")
print(f"\n  V2 빗나감 → 현행 유사도 우선 동점 처리를 유지한다" if not v2 else "")
print(f"  V3 빗나감 → top-10 운영 지점을 다시 의심해야 한다" if not v3 else "")
# 6·7·9차 top-10 풀링
K,N=7+8+k, 9+9+n; lo,hi=wilson(K,N)
print(f"\n=== 6·7·9차 top-10 풀링 (36시드) ===  통과 {N}/36 = {100*N/36:.0f}%   적합 {K}/{N} = {100*K/N:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
