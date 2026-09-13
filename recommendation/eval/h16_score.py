"""H16 — end-to-end 파이프라인 표본 확대(12 → 24 시드).
등급/fun_tag 라벨은 후보를 보기 전에 시드 fun_tag 를 먼저 적고, 그 다음 후보를 판정했다.
적합 = grade >= 2 (전 세션 P@k 기준과 동일).
"""
import math

# (시드명, 시드 fun_tags, [(후보명, 후보 fun_tags, grade), x3])
B2 = [
 ("불확정성의 원리", {"미스터리추리"}, [
   ("플레밍 저택의 죽음", {"미스터리추리"}, 3),
   ("Tokachi탐정", {"미스터리추리"}, 2),
   ("Kumitantei", {"미스터리추리","생존긴장"}, 2)]),
 ("경찰청장 박전전", {"유머경쾌","사이다응징"}, [
   ("Accident Investigator", {"미스터리추리","전략운영"}, 0),
   ("Police Tactics: Imperio", {"전략운영","사이다응징"}, 1),
   ("Whispers of the Luminaries", {"미스터리추리","피폐고통"}, 0)]),
 ("그 마물에 대한 논문 외전", {"성장역전","세계관로어"}, [
   ("Revive", {"세계관로어","탐험발견"}, 1),
   ("Blood of Inquisition", {"전략운영","생존긴장"}, 0),
   ("Malus Code", {"미스터리추리"}, 0)]),
 ("황태자의 처음을 훔쳐버렸다", {"관계로맨스"}, [
   ("Derail: Sacrifice", {"미스터리추리","피폐고통"}, 0),
   ("Himegimi Detective", {"미스터리추리"}, 0),
   ("Lost in Loss", {"미스터리추리"}, 0)]),
 ("초월급 풍수사", {"성장역전","사이다응징"}, [
   ("ONTOS", {"미스터리추리","세계관로어"}, 0),
   ("Another Day Another Dollar", {"전략운영","성장역전"}, 1),
   ("Kung Fu Time Travel", {"성장역전","세계관로어"}, 2)]),
 ("최애의 며느리가 될 예정", {"관계로맨스","성장역전"}, [
   ("별빛 아래", {"관계로맨스"}, 2),
   ("Come to my party!", {"힐링평온","관계로맨스"}, 1),
   ("스위트 클락워크", {"관계로맨스","성장역전","피폐고통"}, 3)]),
 ("북천의 칼잡이", {"사이다응징","세계관로어"}, [
   ("귀신 탐정", {"유머경쾌","미스터리추리"}, 0),
   ("义闻录：轮回", {"사이다응징","세계관로어"}, 3),
   ("Come to my party!", {"힐링평온","관계로맨스"}, 0)]),
 ("Munchkin", {"성장역전","사이다응징"}, [
   ("MunchKing", {"유머경쾌"}, 0),
   ("Escape from Monkey Island", {"유머경쾌","탐험발견"}, 0),
   ("Return to Monkey Island", {"유머경쾌","탐험발견"}, 0)]),
 ("신화급 유물이 내게 집착함", {"성장역전","세계관로어","탐험발견"}, [
   ("Hindsight", {"힐링평온","탐험발견"}, 1),
   ("OPUS: Rocket of Whispers", {"탐험발견","힐링평온","세계관로어"}, 1),
   ("ひとりだるま", {"잔혹공포","생존긴장"}, 0)]),
 ("봉추운", {"성장역전","유머경쾌"}, [
   ("Bravium", {"수집육성","전략운영"}, 0),
   ("Winner's Break", {"관계로맨스"}, 0),
   ("匣中少女的第二次末日", {"잔혹공포","생존긴장"}, 0)]),
 ("반드시 로맨틱", {"관계로맨스","피폐고통"}, [
   ("L'Agence", {"관계로맨스","미스터리추리"}, 1),
   ("아무나 나랑 결혼해줘!", {"관계로맨스","유머경쾌"}, 2),
   ("Dear Nobody", {"관계로맨스","피폐고통"}, 3)]),
 ("빈곤지독", {"성장역전","세계관로어"}, [
   ("LOOM", {"탐험발견","세계관로어"}, 1),
   ("Moonlight thief", {"사이다응징","성장역전"}, 2),
   ("The Meat Factory", {"생존긴장","잔혹공포"}, 0)]),
]
GATE = 0.34
def jac(a,b): return len(a&b)/len(a|b) if a|b else 0.0

a_hit=b_hit=c_hit=c_n=0
print(f"{'시드':22s} {'A 유사도1위':>10s} {'B fun_tag1위':>26s} {'J':>5s} {'게이트':>6s}")
for nm, st, cs in B2:
    A = cs[0]
    best = max(cs, key=lambda c: jac(st,c[1]))
    J = jac(st,best[1])
    a_hit += A[2]>=2
    b_hit += best[2]>=2
    passed = J>=GATE
    if passed:
        c_n+=1; c_hit += best[2]>=2
    print(f"{nm[:20]:22s} {'O' if A[2]>=2 else 'X':>10s} {best[0][:22]:>26s} {J:5.2f} "
          f"{('통과 '+('O' if best[2]>=2 else 'X')) if passed else '차단':>6s}")

def wilson(k,n):
    if n==0: return (0,0)
    p=k/n; z=1.96; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (max(0,c-h), min(1,c+h))

print(f"\n=== 2차 12시드 ===")
for lbl,k,n in [("A 유사도 top-1",a_hit,12),("B fun_tag 재선택",b_hit,12),("C +게이트",c_hit,c_n)]:
    lo,hi=wilson(k,n)
    print(f"  {lbl:18s} {k:2d}/{n:2d} = {100*k/n if n else 0:5.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]")

print(f"\n=== 1차+2차 24시드 풀링 ===")
for lbl,k1,n1,k2,n2 in [("A 유사도 top-1",3,12,a_hit,12),("B fun_tag 재선택",5,12,b_hit,12),("C +게이트",5,6,c_hit,c_n)]:
    k,n=k1+k2,n1+n2; lo,hi=wilson(k,n)
    print(f"  {lbl:18s} {k:2d}/{n:2d} = {100*k/n:5.1f}%  [{100*lo:.0f}%, {100*hi:.0f}%]"
          + (f"   커버리지 {100*n/24:.0f}%" if lbl.startswith("C") else ""))
