"""h50 눈가림 채점표 — 400쌍. 갈래·fun_tag·유사도 순위·점수 모두 비공개.
시드 전문은 각 청크 머리에 참조표로 1회 싣는다(h48 과 동일한 정보량, 반복만 제거)."""
import json, random
SP="/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
d=json.load(open(f"{SP}/h50_seeds20.json"))
pairs=[]
for s,x in enumerate(d):
    for dom in ("steam","tmdb"):
        for ci,c in enumerate(x[dom]):
            pairs.append(dict(seed=s, dom=dom, cand=ci, seed_name=x["wn"],
                              cand_name=c["n"], text=c["d"]))
rng=random.Random(20260903)
perm=list(range(len(pairs))); rng.shuffle(perm)
for pos,i in enumerate(perm): pairs[i]["pid"]=f"P{pos:03d}"
order=list(range(len(pairs))); rng.shuffle(order)
json.dump([{k:p[k] for k in ("pid","seed","dom","cand","seed_name","cand_name")} for p in pairs],
          open(f"{SP}/h50_grade_key.json","w"), ensure_ascii=False, indent=1)

ref="# 시드 참조표\n\n" + "\n".join(
    f"**{x['wn']}** [{x['g']}]\n{x['syn']}\n" for x in d)
blocks=[f"## {pairs[i]['pid']}   시드▸ {pairs[i]['seed_name']}\n**후보** {pairs[i]['cand_name']}\n{pairs[i]['text']}\n"
        for i in order]
N=40
for k in range(0, len(blocks), N):
    body=ref+"\n---\n\n"+"\n".join(blocks[k:k+N])
    fn=f"{SP}/pg{k//N+1}.txt"; open(fn,"w").write(body)
    print(f"  pg{k//N+1}.txt {len(blocks[k:k+N])}쌍 · {len(body):,}자")
print(f"총 {len(pairs)}쌍 · 참조표 {len(ref):,}자")
