"""H23 — 5차 장르 층화 배치 채점 + 사전 등록(h22, md5 adb7fa5e…) 판정."""
import json, math
G=[[2,0,0],[3,2,1],[0,0,0],[2,0,2],[0,0,0],[0,0,1],[2,1,2],[2,0,0],[1,1,1],[0,1,0],[0,0,0],[2,1,1]]
d=json.load(open("h23_funtags.json"))["seeds"]
def jac(a,b):
    A,B=set(a),set(b); return len(A&B)/len(A|B) if A|B else 0.0
def wilson(k,n):
    if not n: return (0,0)
    p=k/n; z=1.96; q=1+z*z/n
    c=(p+z*z/(2*n))/q; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/q
    return max(0,c-h),min(1,c+h)
B5=[[(g,jac(s["t"],c[1])) for c,g in zip(s["c"],gs)] for s,gs in zip(d,G)]
from collections import defaultdict
per=defaultdict(lambda:[0,0,0])
print(f"{'장르':6s} {'시드':26s} {'A':>3s} {'B fun_tag1위':>26s} {'J':>5s}  게이트")
for s,row in zip(d,B5):
    bi=max(range(3),key=lambda i:row[i][1]); J=row[bi][1]; ok=J>=0.34
    p=per[s["g"]]; p[0]+=1; p[1]+=ok; p[2]+= ok and row[bi][0]>=2
    print(f"{s['g']:6s} {s['wn'][:24]:26s} {'O' if row[0][0]>=2 else 'X':>3s} {s['c'][bi][0][:22]:>26s} {J:5.2f}  "
          +("통과 "+("O" if row[bi][0]>=2 else "X") if ok else "차단"))
print("\n=== 장르별 ===")
for g,(n,gp,gh) in per.items():
    print(f"  {g:6s} n={n}  게이트 통과 {gp}/{n}  통과 후 적합 {gh}/{gp}" if gp else f"  {g:6s} n={n}  게이트 통과 0/{n}")
sel=[max(r,key=lambda t:t[1]) for r in B5]; sel=[g for g,j in sel if j>=0.34]
k,n=sum(1 for g in sel if g>=2),len(sel); lo,hi=wilson(k,n)
print(f"\n=== 5차 12시드 ===")
for l,kk,nn in [("A 유사도 top-1",sum(r[0][0]>=2 for r in B5),12),
                ("B fun_tag 재선택",sum(max(r,key=lambda t:t[1])[0]>=2 for r in B5),12)]:
    a,b=wilson(kk,nn); print(f"  {l:22s} {kk:2d}/{nn:2d} = {100*kk/nn:5.1f}%  [{100*a:3.0f}%, {100*b:3.0f}%]")
print(f"  C +게이트              {k:2d}/{n:2d} = {100*k/n if n else 0:5.1f}%  [{100*lo:3.0f}%, {100*hi:3.0f}%]   커버리지 {100*n/12:.0f}%")

print("\n=== 사전 등록 판정 ===")
q1=per["BL"][1]<=1; q2=per["무협"][1]<=2; q3=per["판타지"][1]>=2; q4=(k/n if n else 0)>=0.70
for l,got,pred,ok in [("Q1 BL 통과 <=1",per["BL"][1],"<=1",q1),("Q2 무협 통과 <=2",per["무협"][1],"<=2",q2),
                      ("Q3 판타지 통과 >=2",per["판타지"][1],">=2",q3),("Q4 통과 후 적합률 >=70%",f"{100*k/n if n else 0:.1f}%",">=70%",q4)]:
    print(f"  {l:26s} 관측 {str(got):>6s} (예측 {pred})   {'적중' if ok else '빗나감'}")
miss=sum(1 for x in (q1,q2,q3) if not x)
print(f"\n  기각 조건: Q1~Q3 중 {miss}개 빗나감 → "
      + ("발동: '장르별 카탈로그 겹침' 해석을 버린다" if miss>=2 else "미발동"))
print(f"  Q4 {'적중 — 게이트 자체는 층화 표본에서도 유지된다' if q4 else '빗나감 — 파이프라인 결론 하향'}")

# 전체 풀링
B1=[[(1,.25),(1,.33),(0,.00)],[(1,.25),(0,.25),(2,.25)],[(0,.50),(0,.33),(2,1.0)],
    [(0,.00),(0,.33),(1,.33)],[(0,.00),(1,.25),(0,.00)],[(2,1.0),(1,.33),(0,.33)],
    [(0,.00),(0,.00),(0,.00)],[(0,.50),(2,1.0),(1,.00)],[(3,.67),(2,.25),(1,.00)],
    [(1,.33),(1,.00),(0,.33)],[(1,1.0),(1,.33),(0,.33)],[(2,1.0),(0,.00),(2,1.0)]]
B2=[[(3,1.0),(2,1.0),(2,.50)],[(0,.00),(1,.33),(0,.00)],[(1,.33),(0,.00),(0,.00)],
    [(0,.00),(0,.00),(0,.00)],[(0,.00),(1,.33),(2,.33)],[(2,.50),(1,.33),(3,.67)],
    [(0,.00),(3,1.0),(0,.00)],[(0,.00),(0,.00),(0,.00)],[(1,.25),(1,.40),(0,.00)],
    [(0,.00),(0,.00),(0,.00)],[(1,.33),(2,.33),(3,1.0)],[(1,.33),(2,.33),(0,.00)]]
G3=[[0,0,0],[0,0,0],[1,0,1],[2,2,2],[2,1,1],[2,3,0],[1,3,1],[0,0,0],[0,0,1],[0,2,1],[1,0,0],[1,2,0]]
G4=[[2,2,1],[2,0,3],[1,0,0],[2,2,2],[2,0,1],[0,2,0],[2,0,0],[2,2,2],[1,2,2],[0,0,2],[0,0,1],[0,0,1]]
B3=[[(g,jac(s["t"],c[1])) for c,g in zip(s["c"],gs)] for s,gs in zip(json.load(open("h18_funtags.json"))["seeds"],G3)]
B4=[[(g,jac(s["t"],c[1])) for c,g in zip(s["c"],gs)] for s,gs in zip(json.load(open("h21_funtags.json"))["seeds"],G4)]
ALL=B1+B2+B3+B4+B5
sel=[max(r,key=lambda t:t[1]) for r in ALL]; sel=[g for g,j in sel if j>=0.34]
print(f"\n=== 1~5차 풀링 {len(ALL)}시드 ===")
for l,kk in [("A 유사도 top-1",sum(r[0][0]>=2 for r in ALL)),("B fun_tag 재선택",sum(max(r,key=lambda t:t[1])[0]>=2 for r in ALL))]:
    a,b=wilson(kk,len(ALL)); print(f"  {l:22s} {kk:2d}/{len(ALL)} = {100*kk/len(ALL):5.1f}%  [{100*a:3.0f}%, {100*b:3.0f}%]")
k,n=sum(1 for g in sel if g>=2),len(sel); a,b=wilson(k,n)
print(f"  C +게이트              {k:2d}/{n:2d} = {100*k/n:5.1f}%  [{100*a:3.0f}%, {100*b:3.0f}%]   커버리지 {100*n/len(ALL):.0f}%")
