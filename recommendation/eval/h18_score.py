"""H18 — 프로토콜 청정 배치. fun_tag 는 h18_funtags.json(md5 1712148f…)에 먼저 확정했고,
등급은 그 파일을 다시 읽지 않고 매겼다. 라벨이 등급을 향해 조정될 여지를 없앤 유일한 배치.
"""
import json, math
G=[[0,0,0],[0,0,0],[1,0,1],[2,2,2],[2,1,1],[2,3,0],[1,3,1],[0,0,0],[0,0,1],[0,2,1],[1,0,0],[1,2,0]]
d=json.load(open("h18_funtags.json"))["seeds"]
S=[[c["s"] for c in x["c"]] for x in json.load(open("h18_e2e3.json"))]
assert len(G)==len(d)==len(S)==12
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
def rep(l,k,n,cov=None):
    lo,hi=wilson(k,n)
    print(f"  {l:26s} {k:2d}/{n:2d} = {100*k/n if n else 0:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]"
          +(f"   커버리지 {100*n/cov:.0f}%" if cov else ""))
a=b=c=cn=0; rows=[]
print(f"{'시드':24s} {'A 유사도1위':>8s}  {'B fun_tag1위':>24s} {'J':>5s}  게이트")
for s,g,sim in zip(d,G,S):
    pairs=list(zip(s["c"],g))
    J=[jac(s["t"],cc[1]) for cc,_ in pairs]
    bi=max(range(3),key=lambda i:J[i])
    a+= g[0]>=2; b+= g[bi]>=2
    ok=J[bi]>=0.34
    if ok: cn+=1; c+= g[bi]>=2
    print(f"{s['wn'][:22]:24s} {'O' if g[0]>=2 else 'X':>8s}  {s['c'][bi][0][:20]:>24s} {J[bi]:5.2f}  "
          +("통과 "+("O" if g[bi]>=2 else "X") if ok else "차단"))
    rows.append((J,g,sim))
print("\n=== 3차 12시드 (프로토콜 청정) ===")
rep("A 유사도 top-1",a,12); rep("B fun_tag 재선택",b,12); rep("C +게이트 J>=0.34",c,cn,12)
print("\n=== 1+2+3차 36시드 풀링 ===")
rep("A 유사도 top-1",5+a,36); rep("B fun_tag 재선택",9+b,36); rep("C +게이트",9+c,11+cn,36)
print("\n=== 상관 (36쌍, 청정 배치만) ===")
def pear(x,y):
    n=len(x);mx=sum(x)/n;my=sum(y)/n
    num=sum((p-mx)*(q-my) for p,q in zip(x,y))
    den=math.sqrt(sum((p-mx)**2 for p in x)*sum((q-my)**2 for q in y))
    return num/den if den else 0
gg=[v for g in G for v in g]; jj=[v for J,_,_ in rows for v in J]; ss=[v for _,_,s in rows for v in s]
crit=1.96/math.sqrt(len(gg)-3)
for l,v in [("유사도 vs 등급",ss),("fun_tag Jaccard vs 등급",jj)]:
    r=pear(v,gg); z=0.5*math.log((1+r)/(1-r))
    lo,hi=math.tanh(z-crit),math.tanh(z+crit)
    print(f"  {l:24s} r={r:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  {'유의' if lo>0 or hi<0 else '유의하지 않음'}")
