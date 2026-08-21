"""H41 — 15차 채점 + 사전 등록(h40, md5 763afd15…) 판정."""
import sys, math
sys.path.insert(0,'.')
from h41_funtags import S, tags15
# 시드별 {후보 인덱스: 등급} — 14축/15축이 고른 것 전부
GR={1:{1:2},2:{7:2,3:3},5:{0:1},6:{0:2},7:{4:3,2:3},8:{1:2},9:{1:2},10:{4:1},
    11:{0:1,1:3},12:{0:3},13:{5:1},14:{8:3},15:{3:1},18:{1:3},19:{1:0}}
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
def pick(st,cts):
    J=[jac(st,c) for c in cts]; m=max(J)
    return [j for j,v in enumerate(J) if abs(v-m)<1e-9][0], m
a14=[]; a15=[]; sk14=[]; sk15=[]; ndiff=0
for i,(g,nm,st,sk,cs) in enumerate(S):
    b14,m14=pick(st,[c[0] for c in cs])
    b15,m15=pick(tags15(st,sk),[tags15(c[0],c[1]) for c in cs])
    if b14!=b15 or (m14>=0.34)!=(m15>=0.34): ndiff+=1
    if m14>=0.34:
        a14.append(GR[i][b14])
        if sk: sk14.append(GR[i][b14])
    if m15>=0.34:
        a15.append(GR[i][b15])
        if sk: sk15.append(GR[i][b15])
k1,n1=sum(1 for x in a14 if x>=2),len(a14); lo1,hi1=wilson(k1,n1)
k2,n2=sum(1 for x in a15 if x>=2),len(a15); lo2,hi2=wilson(k2,n2)
print(f"14축 게이트   {k1}/{n1} = {100*k1/n1:5.1f}%  [{100*lo1:3.0f}%, {100*hi1:3.0f}%]   커버리지 {n1}/20 = {100*n1/20:.0f}%")
print(f"15축 게이트   {k2}/{n2} = {100*k2/n2:5.1f}%  [{100*lo2:3.0f}%, {100*hi2:3.0f}%]   커버리지 {n2}/20 = {100*n2/20:.0f}%")
m14s=sum(sk14)/len(sk14) if sk14 else 0; m15s=sum(sk15)/len(sk15) if sk15 else 0
print(f"\n숙련·경쟁 시드 {len(sk14)}건  등급 평균  14축 {m14s:.2f}  →  15축 {m15s:.2f}   (등급 14축 {sk14} / 15축 {sk15})")
b1=ndiff>=2; b2=(k2/n2 if n2 else 0)>=(k1/n1 if n1 else 0); b3=m15s>=m14s; b4=(k2/n2 if n2 else 0)>=0.60
print(f"\n=== 사전 등록 판정 ===")
print(f"  B1 게이트 결과가 다른 시드 >= 2   {ndiff}건          {'적중' if b1 else '빗나감'}")
print(f"  B2 15축 적합률 >= 14축           {100*k2/n2:.1f}% vs {100*k1/n1:.1f}%  {'적중' if b2 else '빗나감'}")
print(f"  B3 숙련경쟁 시드 등급 15축 >= 14축 {m15s:.2f} vs {m14s:.2f}     {'적중' if b3 else '빗나감'}")
print(f"  B4 15축 적합률 >= 60%            {100*k2/n2:.1f}%          {'적중' if b4 else '빗나감'}")
print(f"\n  기각 조건: " + ("미발동" if (b1 and b2 and b4) else "발동"))
