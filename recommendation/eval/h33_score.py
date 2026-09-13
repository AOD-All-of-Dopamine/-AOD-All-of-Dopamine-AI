"""H33 — 10차 채점 + 사전 등록(h32, md5 623ba988…) 판정."""
import sys, math, statistics
sys.path.insert(0,'.')
from h33_funtags import S
# 시드별 {후보 인덱스: 등급} — 게이트가 고른 것만
GR=[{0:3},{7:1},{1:2},{4:3},{9:2},{6:2},{1:2},{1:2},{0:2},{0:3},{2:2},{5:0}]
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
cur=[]; dual=[]; ok_r=[]; bad_r=[]
for i,(nm,st,cts) in enumerate(S):
    J=[jac(st,c) for c in cts]; m=max(J)
    b=[j for j,v in enumerate(J) if abs(v-m)<1e-9][0]
    if m>=0.34:
        g=GR[i][b]; cur.append(g)
        (ok_r if g>=2 else bad_r).append(b+1)
        if b<5: dual.append(g)
k1,n1=sum(1 for g in cur if g>=2),len(cur); lo1,hi1=wilson(k1,n1)
k2,n2=sum(1 for g in dual if g>=2),len(dual); lo2,hi2=wilson(k2,n2)
print(f"현행 게이트 (J>=0.34)          통과 {n1}/12 = {100*n1/12:3.0f}%   적합 {k1}/{n1} = {100*k1/n1:5.1f}%  [{100*lo1:3.0f}%, {100*hi1:3.0f}%]")
print(f"이중 게이트 (J>=0.34 & 순위<=5) 통과 {n2}/12 = {100*n2/12:3.0f}%   적합 {k2}/{n2} = {100*k2/n2:5.1f}%  [{100*lo2:3.0f}%, {100*hi2:3.0f}%]")
print(f"\n성공 시드 순위 {sorted(ok_r)}   실패 시드 순위 {sorted(bad_r)}")
mo=statistics.median(ok_r) if ok_r else 0; mb=statistics.median(bad_r) if bad_r else None
w1 = mb is not None and mb>mo
w2 = (k2/n2 if n2 else 0) > (k1/n1 if n1 else 0)
w3 = n2/12>=0.50
w4 = (k1/n1 if n1 else 0)>=0.60
print(f"\n=== 사전 등록 판정 ===")
print(f"  W1 실패 순위 중앙값 > 성공     {mb} vs {mo} (실패 n={len(bad_r)})   {'적중' if w1 else '빗나감/판정불가'}")
print(f"  W2 이중 적합률 > 현행          {100*k2/n2 if n2 else 0:.1f}% vs {100*k1/n1:.1f}%     {'적중' if w2 else '빗나감'}")
print(f"  W3 이중 커버리지 >= 50%        {100*n2/12:.1f}%              {'적중' if w3 else '빗나감'}")
print(f"  W4 현행 적합률 >= 60%          {100*k1/n1:.1f}%              {'적중' if w4 else '빗나감'}")
print(f"\n  기각 조건: " + ("미발동" if (w1 and w2 and w3) else
      ("발동 — " + ", ".join(x for x,ok in [("W1 빗나감→이중게이트 폐기",w1),("W2 빗나감→유사도 보강 무효",w2),("W3 빗나감→커버리지로 산 것",w3)] if not ok))))
# top-10 전체 풀링 (6,7,9,10차)
K,N=7+8+5+k1, 9+9+9+n1; lo,hi=wilson(K,N)
print(f"\n=== 6·7·9·10차 top-10 풀링 (48시드) ===  통과 {N}/48 = {100*N/48:.0f}%   적합 {K}/{N} = {100*K/N:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
