"""H45 — 18차 채점 + 사전 등록(h44, md5 2c194d0b…) 판정."""
import sys, math
sys.path.insert(0,'.')
from h45_funtags import S, tags
GR={0:{8:1},1:{1:2},2:{3:0,0:3},3:{9:1},4:{6:1},5:{},6:{0:1},7:{},8:{7:1,0:3},9:{6:2},
    10:{0:2},11:{4:1},12:{},13:{9:2},14:{},15:{},16:{},17:{0:3},18:{0:0},19:{3:1}}
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
sel={}; ch={'15축':0,'16축':0}
prev=None
for use_sk,use_mg,name in ((0,0,'14축'),(1,0,'15축'),(1,1,'16축')):
    g=[]; cur=[]
    for i,(gen,nm,st,sk,mg,cs) in enumerate(S):
        b,m = pick(tags(st,sk,mg,use_sk,use_mg), [tags(c[0],c[1],c[2],use_sk,use_mg) for c in cs])
        cur.append((b, m>=0.34))
        if m>=0.34: g.append(GR[i][b])
    sel[name]=g
    if prev is not None:
        ch[name]=sum(1 for a,b in zip(prev,cur) if a!=b)
    prev=cur
for name in ('14축','15축','16축'):
    g=sel[name]; k,n=sum(1 for x in g if x>=2),len(g); lo,hi=wilson(k,n)
    print(f"{name}  {k}/{n} = {100*k/n:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]   커버리지 {n}/20 = {100*n/20:.0f}%"
          + (f"   변경 {ch[name]}건" if name!='14축' else ""))
r={n:(sum(1 for x in sel[n] if x>=2)/len(sel[n])) for n in sel}
e1=ch['16축']>=3; e2=(r['16축']-r['14축'])>=0.05; e3=ch['15축']<=1; e4=0.60<=r['14축']<=0.85
print(f"\n=== 사전 등록 판정 ===")
print(f"  E1 MG 선택 변경 >= 3건        {ch['16축']}건            {'적중' if e1 else '빗나감'}")
print(f"  E2 16축이 14축보다 +5%p 이상  {100*(r['16축']-r['14축']):+.1f}%p        {'적중' if e2 else '빗나감'}")
print(f"  E3 SK 선택 변경 <= 1건        {ch['15축']}건            {'적중' if e3 else '빗나감'}")
print(f"  E4 14축 적합률 60~85%        {100*r['14축']:.1f}%          {'적중' if e4 else '빗나감'}")
print(f"\n  기각 조건(E1 또는 E2): {'발동 — 경영운영 축 폐기, 14축 확정' if not(e1 and e2) else '미발동'}")
print(f"  E3 빗나감 → 숙련·경쟁 폐기 결정을 재검토해야 한다" if not e3 else "")
print(f"  E4 빗나감 → 이 배치 자체가 비정상이라 E1~E3 해석이 어렵다" if not e4 else "")
