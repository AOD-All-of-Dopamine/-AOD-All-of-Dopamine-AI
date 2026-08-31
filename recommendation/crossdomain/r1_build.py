"""R-1 3단계: 재정렬된 top-50 으로 M6 통합 목록을 만들고 채점 슬롯을 뽑는다.
  python crossdomain/r1_build.py <split> <variant>
리랭커는 로드하지 않는다(랭커 임베딩만, 웹소설 화수 필요).
"""
import json, random, os, sys, collections
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation"); sys.path.insert(0,"crossdomain")
split, variant = sys.argv[1], sys.argv[2]
SCR="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
TOP=json.load(open("crossdomain/r1_top100.json"))
RR=json.load(open(f"crossdomain/r1_reranked_{split}_{variant}.json"))
pids=sorted({k.split("|")[2] for k in RR})
import xseed
xseed._load_platform("wn", xseed.WN, xseed.ART["wn"])          # 화수·카드용 (웹소설만)
EPS=xseed.wn_episodes()
sys.path.insert(0,"crossdomain"); from mix import RULES

base=json.load(open("crossdomain/x16_lists.json"))             # 기준선 M6 (val) — dev 는 mixed_x11
basedev=json.load(open("crossdomain/mixed_x11.json"))["M6"]
mixed={}
for pid in pids:
    lists={}; seeds={}
    for pl in ("steam","tmdb","wn"):
        k=f"{split}|{pl}|{pid}"
        if k not in TOP or not TOP[k]["top100"]: continue
        seeds[pl]=len(TOP[k]["seeds"])
        lists[pl]=(RR[k][:50] if k in RR else TOP[k]["top100"][:50])
    if len(lists)<2: continue
    mixed[pid]=[dict(plat=p,item=str(it),rank=rk) for p,it,rk in RULES["M6"](lists,seeds,{p:0.6 for p in lists},k=50,episodes=EPS)]
json.dump(mixed,open(f"crossdomain/r1_mixed_{split}_{variant}.json","w"))

# 기준선 슬롯과 비교 대상 정하기: 리랭크된 플랫폼 슬롯만 본다
target={k.split("|")[1] for k in RR}
bank={}
for f in ("eval/x16_graded.json",):
    for r in json.load(open(f))["rows"]:
        if "maj" in r: bank[(r["pid"],r["plat"],str(r["item"]))]=r["maj"]
for f in ("eval/x14_graded.json","eval/x15_graded.json","eval/r0_graded.json","eval/x9_all_600.json"):
    for r in json.load(open(f)):
        if "maj" in r: bank.setdefault((r["pid"],r["plat"],str(r["item"])),bool(r["maj"]))
for x in ("x10","x11","x12"):
    kd=json.load(open(f"eval/{x}_key.json")); kd=kd["rows"] if isinstance(kd,dict) else kd
    key={r["id"]:r for r in kd if "id" in r}; votes=collections.defaultdict(list)
    for role in "ABC":
        for line in open(f"eval/{x}/{role}.jsonl"):
            if line.strip():
                for i,g in json.loads(line).items(): votes[i].append(int(g)>=2)
    for i,v in votes.items():
        if len(v)>=3 and i in key: r=key[i]; bank.setdefault((r["pid"],r["plat"],str(r["item"])),sum(v)>=2)

rng=random.Random(20260831002); rows=[]
for pid,slots in mixed.items():
    sel=[(s,i+1) for i,s in enumerate(slots) if s["plat"] in target]
    pick=[(s,p) for s,p in sel if p<=10]
    tail=[(s,p) for s,p in sel if p>10]
    pick+=rng.sample(tail,min(10,len(tail)))
    for s,pos in pick:
        r=dict(split=split,variant=variant,pid=pid,plat=s["plat"],item=s["item"],rank=s["rank"],pos=pos)
        k=(pid,s["plat"],s["item"])
        if k in bank: r["maj"]=bank[k]
        rows.append(r)
SN={}
for f in ("eval/x9_all_600.json","eval/x15_graded.json"):
    for r in json.load(open(f)): SN[r["pid"]]=r["seed"]
cards=json.load(open("crossdomain/x16_cards.json"))
missing=[r for r in rows if "maj" not in r and f"{r['plat']}|{r['item']}" not in cards]
if missing: print(f"카드 없는 신규 슬롯 {len(missing)}건 — 카드 생성 필요")
need=[]; seen=set()
for r in rows:
    if "maj" in r: continue
    k=(r["pid"],r["plat"],r["item"])
    if k in seen: continue
    seen.add(k); need.append(r)
json.dump(dict(rows=rows,need=[dict(r) for r in need]),open(f"eval/r1_{split}_{variant}_key.json","w"),ensure_ascii=False)
have=sum(1 for r in rows if "maj" in r)
print(f"슬롯 {len(rows)} · 은행 {have} · 신규 {len(need)}")
