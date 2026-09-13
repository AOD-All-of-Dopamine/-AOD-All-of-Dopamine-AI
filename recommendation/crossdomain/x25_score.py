"""X-25 집계 — 사전등록 80a3fbc4… + 부록 61e99841… 문턱대로만 판정."""
import json, glob, collections, numpy as np
K=json.load(open("eval/x25_key.json")); rows=K["rows"]; todo=K["todo"]
key={t["id"]:t for t in todo}
votes=collections.defaultdict(list)
for f in sorted(glob.glob("eval/x25/[ABC]_*.json")):
    for i,g in json.load(open(f)).items():
        if i in key: votes[i].append(int(g)>=2)
maj={}; unan=0
for i,v in votes.items():
    if len(v)>=3: maj[i]=sum(v)>=2; unan+=(len(set(v))==1)
print(f"신규 판정 {len(maj)}/{len(todo)} · 만장일치 {unan/max(len(maj),1):.2f}")
pair={(key[i]['pid'],key[i]['plat'],key[i]['item']):m for i,m in maj.items()}
miss=0
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["plat"],r["item"])
    if k in pair: r["maj"]=pair[k]
    else: miss+=1
print(f"미채점 슬롯 {miss}")
if miss: raise SystemExit("미채점이 남아 비교하지 않는다 (사전등록)")
json.dump(dict(rows=rows),open("eval/x25_graded.json","w"),ensure_ascii=False)
P=lambda rs: float(np.mean([r["maj"] for r in rs])) if rs else float("nan")
V={}
print("\n### 팔별 (페르소나 v2 32, M6)")
for a in ("k1","k3_red","k3_div","k5"):
    s=[r for r in rows if r["arm"]==a]
    h=[r for r in s if r["part"]=="top10"]; t=[r for r in s if r["part"]=="tail"]
    V[a]=P(s); print(f"  {a:<7} 1-10 {P(h):.3f} · 11-50 {P(t):.3f} · 합산 {P(s):.4f} (n={len(s)})")
d1=V["k1"]-V["k5"]; d2=V["k3_div"]-V["k3_red"]
print(f"\n축 1 · 개수:   k1 − k5 = {d1:+.4f}")
print("  판정:", "(A1) 개수가 큰 축" if d1<=-0.10 else ("(B1) 약한 효과" if d1<=-0.05 else "(C1) 개수 효과 없음"))
print(f"축 2 · 다양성: k3_div − k3_red = {d2:+.4f}")
print("  판정:", "(A2) 다양성이 따로 작동" if d2>=0.05 else ("(D2) 반대 방향 — 중복이 낫다" if d2<=-0.05 else "(C2) 다양성 효과 없음"))
print("\n### 플랫폼별 합산")
for plat in ("steam","tmdb","wn"):
    print("  "+plat.ljust(6)+"  "+" · ".join(f"{a} {P([r for r in rows if r['arm']==a and r['plat']==plat]):.3f}" for a in ("k1","k3_red","k3_div","k5")))
