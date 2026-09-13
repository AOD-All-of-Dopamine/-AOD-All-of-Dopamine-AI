"""h50 라벨링 입력 — 시드 20 + Steam 200 + TMDB 200 = 420건. 전문 사용(D-10), ID 셔플(D-11)."""
import json, random
SP="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
d=json.load(open(f"{SP}/h50_seeds20.json"))
items=[]
for s,x in enumerate(d):
    items.append(dict(kind="seed", seed=s, dom=None, cand=None, name=x["wn"], text=x["syn"], g=x["g"]))
    for dom in ("steam","tmdb"):
        for ci,c in enumerate(x[dom]):
            items.append(dict(kind="cand", seed=s, dom=dom, cand=ci, name=c["n"], text=c["d"], g=None))
def assert_not_truncated(texts, label):
    L=sorted(len(t) for t in texts); med,mx=L[len(L)//2],L[-1]
    if med==mx: raise SystemExit(f"[절단 감지] {label}: 중앙 {med} == 최대 {mx}.")
    print(f"  {label}: {len(texts)}건 · 중앙 {med} · 최대 {mx} · 절단 없음")
assert_not_truncated([i["text"] for i in items if i["kind"]=="seed"], "시드")
for dom in ("steam","tmdb"):
    assert_not_truncated([i["text"] for i in items if i["dom"]==dom], dom)

rng=random.Random(20260902)
perm=list(range(len(items))); rng.shuffle(perm)      # D-11: ID 가 시드/후보 주기를 드러내지 않게
for pos,i in enumerate(perm): items[i]["id"]=f"M{pos:03d}"
order=list(range(len(items))); rng.shuffle(order)     # 제시 순서도 섞는다
key=[dict(id=items[i]["id"], kind=items[i]["kind"], seed=items[i]["seed"],
          dom=items[i]["dom"], cand=items[i]["cand"], name=items[i]["name"]) for i in range(len(items))]
json.dump(key, open(f"{SP}/h50_key.json","w"), ensure_ascii=False, indent=1)
lines=[]
for i in order:
    it=items[i]
    lines.append(f"## {it['id']}  {it['name']}\n{it['text']}\n")
txt="\n".join(lines); open(f"{SP}/h50_label.md","w").write(txt)
print(f"\n라벨 대상 {len(items)}건 · {len(txt):,}자 → h50_label.md")
ls=txt.split("\n## ")
chunks, cur = [], []
for b in ls:
    cur.append(b)
    if len("\n## ".join(cur))>22000: chunks.append(cur); cur=[]
if cur: chunks.append(cur)
for k,c in enumerate(chunks):
    body=("## " if k else "")+"\n## ".join(c)
    open(f"{SP}/ml{k+1}.txt","w").write(body)
    print(f"  ml{k+1}.txt {body.count('## '):3d}건 {len(body):,}자")
