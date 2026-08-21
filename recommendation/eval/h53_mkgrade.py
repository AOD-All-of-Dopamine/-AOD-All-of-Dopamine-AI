"""h53 눈가림 채점표 — 신규 쌍만. 기존 등급은 h50/h51 에서 재사용. ID 셔플(D-11), 전문 사용(D-10)."""
import json, random, sys, collections
SP='/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad'
sys.path.insert(0, SP)
from h50_grades import G as G50
from h51_grades import G as G51

pools=json.load(open(f'{SP}/h53_pools.json'))['pools']
seeds=json.load(open(f'{SP}/h50_seeds20.json'))

# 기존 등급 지도 (seed, dom, name) -> grade
gm={}
for k in json.load(open(f'{SP}/h50_grade_key.json')):
    if k['pid'] in G50: gm[(k['seed'], k['dom'], k['cand_name'])]=G50[k['pid']]
for k in json.load(open(f'{SP}/h51_grade_key.json')):
    if k['pid'] in G51: gm[(k['seed'], 'wn', k['cand_name'])]=G51[k['pid']]
print(f"기존 등급 지도 {len(gm)}건")

# 필요한 고유 쌍
uniq={}
for s, arms in pools.items():
    for a in ('M','S'):
        for d in ('wn','steam','tmdb'):
            for c in arms[a][d]:
                key=(int(s), d, c['n'])
                if key not in uniq: uniq[key]=c['d']
print(f"고유 쌍 {len(uniq)} · 재사용 {sum(1 for k in uniq if k in gm)} · 신규 {sum(1 for k in uniq if k not in gm)}")

new=[dict(seed=s, dom=d, cand_name=n, text=t) for (s,d,n),t in uniq.items() if (s,d,n) not in gm]
by=collections.Counter((p['dom']) for p in new)
print("  신규 도메인별", dict(by))
L=sorted(len(p['text']) for p in new)
if L[len(L)//2]==L[-1]: raise SystemExit("[절단 감지]")
print(f"  본문 중앙 {L[len(L)//2]} · 최대 {L[-1]} · 절단 없음")

rng=random.Random(20260909)
pm=list(range(len(new))); rng.shuffle(pm)
for pos,i in enumerate(pm): new[i]['pid']=f"Y{pos:03d}"
json.dump([{k:p[k] for k in ('pid','seed','dom','cand_name')} for p in new],
          open(f"{SP}/h53_grade_key.json","w"), ensure_ascii=False, indent=1)

ref="# 시드 참조표\n\n"+"\n".join(f"**[{i}] {x['wn']}** [{x['g']}]\n{x['syn']}\n" for i,x in enumerate(seeds))
order=list(range(len(new))); rng.shuffle(order)
bl=[f"## {new[i]['pid']}   시드▸ [{new[i]['seed']}] {seeds[new[i]['seed']]['wn']}\n"
    f"**후보** {new[i]['cand_name']}\n{new[i]['text']}\n" for i in order]
for k in range(0, len(bl), 40):
    body=ref+"\n---\n\n"+"\n".join(bl[k:k+40])
    open(f"{SP}/yg{k//40+1}.txt","w").write(body)
    print(f"  yg{k//40+1}.txt {len(bl[k:k+40])}쌍 {len(body):,}자")
