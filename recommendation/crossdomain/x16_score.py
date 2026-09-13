"""X-16 채점 통합 + 지표. 은행(기존) + x16 신규(A/B/C 다수결, 적합 = 등급 ≥2 이 2표 이상)."""
import json, glob, collections, sys, os
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
K=json.load(open("eval/x16_key.json")); rows=K["rows"]; todo={t["id"]:t for t in K["todo"]}
votes=collections.defaultdict(dict)
for role in "ABC":
    for f in [f"eval/x16/{role}.jsonl"]+sorted(glob.glob(f"eval/x16/{role}_*.json")):
        for line in open(f):
            line=line.strip()
            if not line: continue
            try: d=json.loads(line)
            except Exception as e: print("파싱 실패",f,e); continue
            for i,g in d.items(): votes[i][role]=int(g)
new={}
for i,v in votes.items():
    if i not in todo: continue
    if len(v)>=2:
        yes=sum(g>=2 for g in v.values()); new[(todo[i]["pid"],todo[i]["plat"],todo[i]["item"])]=(yes*2>len(v)) if len(v)==3 else (yes==2)
    # 2표 동률(1:1)은 미판정
miss=collections.Counter(); filled=0
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["plat"],r["item"])
    if k in new: r["maj"]=new[k]; filled+=1
    else: miss[(r["split"],r["variant"])]+=1
json.dump(K,open("eval/x16_graded.json","w"),ensure_ascii=False)
unan=sum(1 for i,v in votes.items() if len(v)==3 and len({g>=2 for g in v.values()})==1); n3=sum(1 for v in votes.values() if len(v)==3)
print(f"신규 판정 {len(new)} · 채워짐 {filled} · 만장일치 {unan/max(n3,1):.2f} (n={n3})"); print("미채점:",dict(miss) or 0)
def P(rs, lo, hi):
    s=[r for r in rs if lo<=r["pos"]<=hi and "maj" in r]; return (sum(r["maj"] for r in s)/len(s) if s else float('nan')), len(s)
print(f"\n{'split':<5}{'variant':<8}{'P@10':>7}{'P11-50':>8}{'P@50*':>8}{'n':>6}   플랫폼별 P@50 (steam/tmdb/wn)")
for split in ("dev","val"):
    for var in ("M6","C1a","C1b","C2"):
        rs=[r for r in rows if r["split"]==split and r["variant"]==var]
        if not any("maj" in r for r in rs): continue
        p10,n10=P(rs,1,10); pt,nt=P(rs,11,50)
        p50=(p10*10+pt*40)/50 if nt else float('nan')   # 1-10 전수 + 11-50 표본 가중
        pp=[]
        for pl in ("steam","tmdb","wn"):
            s=[r for r in rs if r["plat"]==pl and "maj" in r]; pp.append(f"{sum(r['maj'] for r in s)/len(s):.2f}" if s else "-")
        print(f"{split:<5}{var:<8}{p10:>7.3f}{pt:>8.3f}{p50:>8.3f}{n10+nt:>6}   {'/'.join(pp)}")
