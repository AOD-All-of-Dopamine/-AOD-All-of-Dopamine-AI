"""H43 — 16차 채점 + 사전 등록(h42, md5 68f624c0…) 판정."""
import sys, math
sys.path.insert(0,'.')
from h43_funtags import S, tags
GR={0:{}, 1:{3:1,2:3}, 2:{1:2,0:1}, 3:{2:2}, 4:{3:1}, 5:{5:3}, 6:{8:1}, 7:{},
    8:{7:2}, 9:{5:2}, 10:{1:2}, 11:{2:2}, 12:{3:2}, 13:{}, 14:{0:2}, 15:{0:3},
    16:{}, 17:{1:2}, 18:{1:0}, 19:{}}
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
res={}; changes={}
for use_sk,use_mg,name in ((0,0,'14축'),(1,0,'15축'),(1,1,'16축')):
    g=[]
    for i,(gen,nm,st,sk,mg,cs) in enumerate(S):
        b,m = pick(tags(st,sk,mg,use_sk,use_mg), [tags(c[0],c[1],c[2],use_sk,use_mg) for c in cs])
        if m>=0.34: g.append(GR[i][b])
    res[name]=g
prev=None
for use_sk,use_mg,name in ((0,0,'14축'),(1,0,'15축'),(1,1,'16축')):
    n_ch=0
    for i,(gen,nm,st,sk,mg,cs) in enumerate(S):
        b,m = pick(tags(st,sk,mg,use_sk,use_mg), [tags(c[0],c[1],c[2],use_sk,use_mg) for c in cs])
        cur=(b, m>=0.34)
        if prev is not None and prev[i]!=cur: n_ch+=1
    changes[name]=n_ch
    prev=[pick(tags(st,sk,mg,use_sk,use_mg), [tags(c[0],c[1],c[2],use_sk,use_mg) for c in cs]) for (gen,nm,st,sk,mg,cs) in S]
    prev=[(b, m>=0.34) for b,m in prev]
for name in ('14축','15축','16축'):
    g=res[name]; k,n=sum(1 for x in g if x>=2),len(g); lo,hi=wilson(k,n)
    print(f"{name}  {k}/{n} = {100*k/n:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]   커버리지 {n}/20 = {100*n/20:.0f}%"
          + (f"   앞 단계 대비 선택 변경 {changes[name]}건" if name!='14축' else ""))
r14=sum(1 for x in res['14축'] if x>=2)/len(res['14축'])
r15=sum(1 for x in res['15축'] if x>=2)/len(res['15축'])
r16=sum(1 for x in res['16축'] if x>=2)/len(res['16축'])
c1=changes['16축']>=2; c2=r16>=r15; c3=r15>=r14; c4=r16>=0.60
print(f"\n=== 사전 등록 판정 ===")
print(f"  C1 16축이 선택 바꾸는 시드 >= 2   {changes['16축']}건        {'적중' if c1 else '빗나감'}")
print(f"  C2 16축 적합률 >= 15축           {100*r16:.1f}% vs {100*r15:.1f}%  {'적중' if c2 else '빗나감'}")
print(f"  C3 15축 >= 14축 (이득 재현)      {100*r15:.1f}% vs {100*r14:.1f}%  {'적중' if c3 else '빗나감'}")
print(f"  C4 16축 적합률 >= 60%           {100*r16:.1f}%          {'적중' if c4 else '빗나감'}")
print(f"\n  ** C3 주의: 15축이 14축 대비 선택을 바꾼 시드가 {changes['15축']}건이다.")
print(f"     '>=' 는 만족하지만 **이득은 {100*(r15-r14):+.1f}%p** 로, 15차의 +6.7%p 는 재현되지 않았다.")
