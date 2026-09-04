"""X-24 집계 — 사전등록 59cf944f… 문턱대로만 판정한다."""
import json, glob, collections, numpy as np, os
K=json.load(open("eval/x24_key.json")); rows=K["rows"]; todo=K["todo"]
key={t["id"]:t for t in todo}
votes=collections.defaultdict(list)
for f in sorted(glob.glob("eval/x24/[ABC]_*.json")):
    for i,g in json.load(open(f)).items():
        if i in key: votes[i].append(int(g)>=2)
maj={}; unan=0
for i,v in votes.items():
    if len(v)>=3:
        maj[i]=sum(v)>=2
        unan+= (len(set(v))==1)
print(f"신규 판정 {len(maj)}/{len(todo)} · 만장일치 {unan/max(len(maj),1):.2f}")
pair={(key[i]['pid'],key[i]['plat'],key[i]['item']):m for i,m in maj.items()}
miss=0
for r in rows:
    k=(r["pid"],r["plat"],r["item"])
    if k in pair: r["maj"]=pair[k]
    else: miss+=1
print(f"미채점 슬롯 {miss}")
if miss: raise SystemExit("미채점이 남아 비교하지 않는다 (사전등록)")
json.dump(dict(rows=rows),open("eval/x24_graded.json","w"),ensure_ascii=False)
P=lambda rs: float(np.mean([r["maj"] for r in rs])) if rs else float("nan")
def band(rs,lab):
    h=[r for r in rs if r["part"]=="top10"]; t=[r for r in rs if r["part"]=="tail"]
    print(f"  {lab:<10} 1-10 {P(h):.3f} · 11-50 {P(t):.3f} · 합산 {P(rs):.4f}  (n={len(rs)})")
    return P(rs)
print("\n### 근거 페르소나 50 (판정 대상)")
v={}
for tag in ("M6","C1a"):
    v[tag]=band([r for r in rows if r["split"]=="근거" and r["variant"]==tag],tag)
d=v["C1a"]-v["M6"]
print(f"  Δ(C1a − M6) = {d:+.4f}")
print("  판정:", "(A) 조합이 진짜면 C1 이 이긴다" if d>=0.05 else
                ("(C) X-16 결론 유지" if d<0.02 else "보류"))
print("\n### 참조군 페르소나 v2 32 · M6 (판정 아님)")
ref=band([r for r in rows if r["split"]=="참조"],"M6")
print(f"  근거 M6 {v['M6']:.4f} vs 참조 M6 {ref:.4f} → {v['M6']-ref:+.4f} (난이도 교란, 판정 무관)")
print("\n### 플랫폼별 (근거)")
for plat in ("steam","tmdb","wn"):
    line=f"  {plat:<6}"
    for tag in ("M6","C1a"):
        s=[r for r in rows if r["split"]=="근거" and r["variant"]==tag and r["plat"]==plat]
        h=[r for r in s if r["part"]=="top10"]; t=[r for r in s if r["part"]=="tail"]
        line+=f"  {tag} {P(h):.2f}/{P(t):.2f} (n={len(s)})"
    print(line)
print("\n### 종류별 (근거)")
G=json.load(open("crossdomain/x24_personas.json")); kind={p["pid"]:p["kind"] for p in G}
for kd in ("game","novel"):
    line=f"  {kd:<6}"
    for tag in ("M6","C1a"):
        s=[r for r in rows if r["split"]=="근거" and r["variant"]==tag and kind.get(r["pid"])==kd]
        line+=f"  {tag} {P(s):.3f} (n={len(s)})"
    print(line)
