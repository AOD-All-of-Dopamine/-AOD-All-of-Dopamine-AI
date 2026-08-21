"""h53 신규 라벨 입력 447건. 전문 사용(D-10), ID 셔플(D-11)."""
import json, random, sys
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0,SP)
k50=json.load(open(f"{SP}/h50_key.json")); k51=json.load(open(f"{SP}/h51_key.json"))
d50=json.load(open(f"{SP}/h50_seeds20.json"))
labeled={(k['dom'],k['name']) for k in k50 if k['kind']=='cand'} | {('wn',k['name']) for k in k51} \
         | {('wn',x['wn']) for x in d50}
P=json.load(open(f"{SP}/h53_pools.json"))['pools']
items={}
for s,arms in P.items():
    for a,doms in arms.items():
        for d,cs in doms.items():
            for c in cs:
                if (d,c['n']) not in labeled and (d,c['n']) not in items:
                    items[(d,c['n'])]=c['d']
lst=[dict(dom=d,name=n,text=t) for (d,n),t in items.items()]
L=sorted(len(i['text']) for i in lst)
if L[len(L)//2]==L[-1]: raise SystemExit("[절단 감지]")
print(f"신규 라벨 {len(lst)}건 · 중앙 {L[len(L)//2]} · 최대 {L[-1]} · 절단 없음")
rng=random.Random(20260908)
perm=list(range(len(lst))); rng.shuffle(perm)
for pos,i in enumerate(perm): lst[i]['id']=f"X{pos:03d}"
json.dump([{k:i[k] for k in ('id','dom','name')} for i in lst],
          open(f"{SP}/h53_key.json","w"), ensure_ascii=False, indent=1)
order=list(range(len(lst))); rng.shuffle(order)
txt="\n".join(f"## {lst[i]['id']}  {lst[i]['name']}\n{lst[i]['text']}\n" for i in order)
open(f"{SP}/h53_label.md","w").write(txt)
bl=txt.split("\n## ")
for k in range(0,len(bl),50):
    body=("## " if k else "")+"\n## ".join(bl[k:k+50])
    open(f"{SP}/xl{k//50+1}.txt","w").write(body)
    print(f"  xl{k//50+1}.txt {body.count('## ')}건 {len(body):,}자")
