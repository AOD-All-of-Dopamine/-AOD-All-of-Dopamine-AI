"""H35 — 11차(24시드) 채점 + 사전 등록(h34, md5 9e232e87…) 판정."""
import sys, math
sys.path.insert(0,'.')
from h35_funtags import S
GR={1:{5:3},2:{5:3},3:{9:2},4:{0:3},6:{8:0},7:{6:2},9:{1:1},10:{1:2},11:{4:2},12:{4:2},
    14:{7:2},15:{0:3},16:{3:2},17:{0:2},19:{0:2},22:{6:1}}
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
GRP={"무협":"무협","판타지":"판타지·현판","현판":"판타지·현판","로판":"로판·로맨스",
     "로맨스":"로판·로맨스","BL":"기타","미스터리":"기타"}
from collections import defaultdict
per=defaultdict(lambda:[0,0,0])   # 시드수, 통과, 적합
allg=[]
for i,(g,nm,st,cts) in enumerate(S):
    J=[jac(st,c) for c in cts]; m=max(J)
    b=[j for j,v in enumerate(J) if abs(v-m)<1e-9][0]
    k=GRP[g]; per[k][0]+=1
    if m>=0.34:
        grade=GR[i][b]; per[k][1]+=1; per[k][2]+= grade>=2; allg.append(grade)
k,n=sum(1 for x in allg if x>=2),len(allg); lo,hi=wilson(k,n)
print(f"=== 11차 24시드 ===  통과 {n}/24 = {100*n/24:.0f}%   적합 {k}/{n} = {100*k/n:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]  폭 {100*(hi-lo):.0f}%p")
print(f"\n{'묶음':12s} {'시드':>4s} {'통과':>4s} {'적합률':>10s}")
rates=[]
for g,(ns,gp,gh) in per.items():
    r=gh/gp if gp else None
    if r is not None: rates.append(r)
    print(f"{g:12s} {ns:4d} {gp:4d} {(f'{gh}/{gp} = {100*r:.0f}%' if gp else '—'):>12s}")
spread=(max(rates)-min(rates)) if len(rates)>1 else 0
x1=0.65<=k/n<=0.90; x2=(hi-lo)<=0.35; x3=spread>=0.30
print(f"\n=== 사전 등록 판정 ===")
print(f"  X1 적합률 65~90%           {100*k/n:.1f}%           {'적중' if x1 else '빗나감'}")
print(f"  X2 구간 폭 <= 35%p         {100*(hi-lo):.0f}%p            {'적중' if x2 else '빗나감'}")
print(f"  X3 장르 묶음 최대-최소 >= 30%p {100*spread:.0f}%p            {'적중' if x3 else '빗나감'}")
print(f"\n  기각 조건: " + ("미발동 — 장르가 배치 변동의 원인 후보로 남는다" if x3
      else "발동 — 배치 변동은 장르가 아니라 표본 잡음. 앞으로 배치별 숫자를 인용하지 않고 누적 풀링만 보고한다."))
if not x1: print("  X1 빗나감 → 48시드 풀링 78.4% 자체를 다시 의심한다")
# top-10 전체 풀링
K,N=7+8+5+9+k, 9+9+9+10+n; lo,hi=wilson(K,N)
print(f"\n=== 6·7·9·10·11차 top-10 풀링 (72시드) ===  통과 {N}/72 = {100*N/72:.0f}%   적합 {K}/{N} = {100*K/N:.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")
