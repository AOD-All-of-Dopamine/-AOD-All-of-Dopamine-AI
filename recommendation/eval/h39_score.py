"""H39 — 14차(20시드 신규) 채점 + 사전 등록(h39, md5 9ff0ee1c…) 판정."""
import sys, math
sys.path.insert(0,'.')
from h39_funtags import S
GR={0:{1:2},2:{0:3},3:{1:1},4:{0:2},5:{0:3},7:{1:1},8:{9:1},9:{0:1},10:{0:2},11:{0:1},
    12:{7:1},13:{6:1,3:2},14:{4:1},16:{1:2},17:{2:1},18:{1:2},19:{4:2}}
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
cur=[]; dual=[]
for i,(g,nm,st,cts) in enumerate(S):
    J=[jac(st,c) for c in cts]; m=max(J)
    b=[j for j,v in enumerate(J) if abs(v-m)<1e-9][0]
    if m>=0.34: cur.append((nm,b+1,GR[i][b]))
    e=[j for j in range(5) if J[j]>=0.34]
    if e:
        eb=max(e,key=lambda j:J[j]); dual.append((nm,eb+1,GR[i][eb]))
k1,n1=sum(1 for *_,g in cur if g>=2),len(cur); lo1,hi1=wilson(k1,n1)
k2,n2=sum(1 for *_,g in dual if g>=2),len(dual); lo2,hi2=wilson(k2,n2)
print(f"현행 J>=0.34        {k1}/{n1} = {100*k1/n1:5.1f}%  [{100*lo1:3.0f}%, {100*hi1:3.0f}%]   커버리지 {n1}/20 = {100*n1/20:.0f}%")
print(f"이중 (J & 순위<=5)   {k2}/{n2} = {100*k2/n2:5.1f}%  [{100*lo2:3.0f}%, {100*hi2:3.0f}%]   커버리지 {n2}/20 = {100*n2/20:.0f}%")
a1=(k2/n2 if n2 else 0)>=0.75; a2=(k2/n2 if n2 else 0)>=(k1/n1 if n1 else 0)
a3=n2/20>=0.40; a4=0.65<=(k1/n1 if n1 else 0)<=0.90
print(f"\n=== 사전 등록 판정 ===")
print(f"  A1 이중 적합률 >= 75%       {100*k2/n2:.1f}%          {'적중' if a1 else '빗나감'}")
print(f"  A2 이중 >= 현행             {100*k2/n2:.1f}% vs {100*k1/n1:.1f}%  {'적중' if a2 else '빗나감'}")
print(f"  A3 이중 커버리지 >= 40%      {100*n2/20:.0f}%            {'적중' if a3 else '빗나감'}")
print(f"  A4 현행 적합률 65~90%       {100*k1/n1:.1f}%          {'적중' if a4 else '빗나감'}")
print(f"\n  기각 조건: " + ("미발동" if (a1 and a2 and a3) else "발동 — " +
      ("A1 빗나감 → 이중 게이트 채택 철회" if not a1 else "") +
      (" · A2 빗나감" if not a2 else "") + (" · A3 빗나감" if not a3 else "")))
if not a4: print("  A4 빗나감 → 79.2% 기준선 자체를 다시 의심한다")
print(f"\n실패 건:")
for nm,r,g in cur:
    if g<2: print(f"  {nm[:26]:28s} {r:2d}위 등급 {g}")
# 회고 72 + 신규 20 풀링
K,N=42+k1, 53+n1; lo,hi=wilson(K,N)
print(f"\n=== 현행 게이트 92시드 풀링 === {K}/{N} = {100*K/N:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
K,N=31+k2, 36+n2; lo,hi=wilson(K,N)
print(f"=== 이중 게이트 92시드 풀링 === {K}/{N} = {100*K/N:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
