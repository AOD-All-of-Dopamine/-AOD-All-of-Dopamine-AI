"""H21 — 4차 배치 채점 + 사전 등록(h20, md5 915d57a6…) 예측 판정."""
import json, math
G=[[2,2,1],[2,0,3],[1,0,0],[2,2,2],[2,0,1],[0,2,0],[2,0,0],[2,2,2],[1,2,2],[0,0,2],[0,0,1],[0,0,1]]
d=json.load(open("h21_funtags.json"))["seeds"]
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
B4=[[(g,jac(s["t"],c[1])) for c,g in zip(s["c"],gs)] for s,gs in zip(d,G)]
print(f"{'시드':30s} {'A':>3s} {'B fun_tag1위':>26s} {'J':>5s}")
for s,row in zip(d,B4):
    bi=max(range(3),key=lambda i:row[i][1])
    print(f"{s['wn'][:28]:30s} {'O' if row[0][0]>=2 else 'X':>3s} {s['c'][bi][0][:22]:>26s} {row[bi][1]:5.2f}")

def run(sets,T):
    sel=[max(r,key=lambda t:t[1]) for r in sets]
    sel=[g for g,j in sel if j>=T]
    return sum(1 for g in sel if g>=2), len(sel)
print(f"\n=== 4차 12시드 (프로토콜 청정) ===")
for l,k,n in [("A 유사도 top-1",sum(r[0][0]>=2 for r in B4),12),
              ("B fun_tag 재선택",sum(max(r,key=lambda t:t[1])[0]>=2 for r in B4),12)]:
    lo,hi=wilson(k,n); print(f"  {l:22s} {k:2d}/{n:2d} = {100*k/n:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]")
for T in (0.34,0.50):
    k,n=run(B4,T); lo,hi=wilson(k,n)
    print(f"  C 게이트 T={T:<5.2f}     {k:2d}/{n:2d} = {100*k/n if n else 0:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]   커버리지 {100*n/12:.0f}%")

print("\n=== 사전 등록 예측 판정 ===")
k34,n34=run(B4,0.34); k50,n50=run(B4,0.50)
r34=k34/n34 if n34 else 0; r50=k50/n50 if n50 else 0
p1 = r50>=r34
p2 = n50<n34
hi_pairs=[g for r in B4 for g,j in r if j>=0.50]
lo_pairs=[g for r in B4 for g,j in r if 0.25<=j<=0.40]
p3r=sum(1 for g in hi_pairs if g>=2)/len(hi_pairs) if hi_pairs else 0
p4r=sum(1 for g in lo_pairs if g>=2)/len(lo_pairs) if lo_pairs else 0
print(f"  P1 T=0.50 적합률 >= T=0.34   {100*r50:.1f}% vs {100*r34:.1f}%   {'적중' if p1 else '빗나감'}")
print(f"  P2 T=0.50 커버리지 < T=0.34  {n50}/12 vs {n34}/12            {'적중' if p2 else '빗나감'}")
print(f"  P3 J>=0.50 쌍 적합률 >= 70%  {100*p3r:.1f}% (n={len(hi_pairs)})      {'적중' if p3r>=0.70 else '빗나감'}")
print(f"  P4 J in[0.25,0.40] < 40%     {100*p4r:.1f}% (n={len(lo_pairs)})      {'적중' if p4r<0.40 else '빗나감'}")
print(f"\n  기각 조건(P3<70% 또는 P4>=40%) → {'발동: 0.5 계단은 우연이었다' if (p3r<0.70 or p4r>=0.40) else '미발동'}")

# 전 배치 풀링
B1=[[(1,.25),(1,.33),(0,.00)],[(1,.25),(0,.25),(2,.25)],[(0,.50),(0,.33),(2,1.0)],
    [(0,.00),(0,.33),(1,.33)],[(0,.00),(1,.25),(0,.00)],[(2,1.0),(1,.33),(0,.33)],
    [(0,.00),(0,.00),(0,.00)],[(0,.50),(2,1.0),(1,.00)],[(3,.67),(2,.25),(1,.00)],
    [(1,.33),(1,.00),(0,.33)],[(1,1.0),(1,.33),(0,.33)],[(2,1.0),(0,.00),(2,1.0)]]
B2=[[(3,1.0),(2,1.0),(2,.50)],[(0,.00),(1,.33),(0,.00)],[(1,.33),(0,.00),(0,.00)],
    [(0,.00),(0,.00),(0,.00)],[(0,.00),(1,.33),(2,.33)],[(2,.50),(1,.33),(3,.67)],
    [(0,.00),(3,1.0),(0,.00)],[(0,.00),(0,.00),(0,.00)],[(1,.25),(1,.40),(0,.00)],
    [(0,.00),(0,.00),(0,.00)],[(1,.33),(2,.33),(3,1.0)],[(1,.33),(2,.33),(0,.00)]]
G3=[[0,0,0],[0,0,0],[1,0,1],[2,2,2],[2,1,1],[2,3,0],[1,3,1],[0,0,0],[0,0,1],[0,2,1],[1,0,0],[1,2,0]]
d3=json.load(open("h18_funtags.json"))["seeds"]
B3=[[(g,jac(s["t"],c[1])) for c,g in zip(s["c"],gs)] for s,gs in zip(d3,G3)]
ALL=B1+B2+B3+B4
print(f"\n=== 1~4차 풀링 {len(ALL)}시드 ===")
for l,k,n in [("A 유사도 top-1",sum(r[0][0]>=2 for r in ALL),len(ALL)),
              ("B fun_tag 재선택",sum(max(r,key=lambda t:t[1])[0]>=2 for r in ALL),len(ALL))]:
    lo,hi=wilson(k,n); print(f"  {l:22s} {k:2d}/{n:2d} = {100*k/n:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]")
for T in (0.34,0.50):
    k,n=run(ALL,T); lo,hi=wilson(k,n)
    print(f"  C 게이트 T={T:<5.2f}     {k:2d}/{n:2d} = {100*k/n:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]   커버리지 {100*n/len(ALL):.0f}%")
