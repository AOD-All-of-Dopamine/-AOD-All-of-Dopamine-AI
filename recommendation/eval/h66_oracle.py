"""h66 보조 — 적합_전체가 이 풀에서 도달 가능한가 (기술 통계)."""
exec(open('/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad/h66_agg.py').read().split('print("\\n=== N1')[0])
import numpy as np
tot=[c for ss in pools for c in pools[ss]]
print(f"풀 전체 {len(tot)}건 · 적합_전체 통과 {100*np.mean([c['allfit'] for c in tot]):.1f}% · "
      f"적합_max 통과 {100*np.mean([c['mx']>=2 for c in tot]):.1f}%")
print(f"\n{'시드':>5} {'후보':>4} {'적합_전체 통과 수':>14} {'오라클(도메인별 1칸)':>20}")
orc=[]
for ss in pools:
    p=pools[ss]; n=sum(c['allfit'] for c in p)
    # 도메인마다 적합_전체 통과가 있으면 뽑는다 = 도달 가능 최대
    best=[]
    for d in ('wn','steam','tmdb'):
        cd=[c for c in p if c['dom']==d]
        best.append(1 if any(c['allfit'] for c in cd) else 0)
    o=100*np.mean(best); orc.append(o)
    print(f"{ss:>5} {len(p):>4} {n:>14} {o:>19.0f}%")
print(f"{'평균':>5} {'':>4} {'':>14} {np.mean(orc):>19.1f}%")
print(f"\n코사인 max 쿼터 = 26.7% · 무작위 = 26.4% · **오라클 = {np.mean(orc):.1f}%**")
# 등급 모양 분포
import collections
sh=collections.Counter()
for c in tot:
    g=sorted(c['gs'],reverse=True)
    sh['모두 좋음(최소≥2)' if min(g)>=2 else
       ('전문가(최고3·최저0)' if (max(g)==3 and min(g)==0) else
        ('모두 무난(최고≤1)' if max(g)<=1 else '혼합'))]+=1
print("\n후보 240건의 등급 모양:")
for k,v in sh.most_common(): print(f"  {k:<18} {v:>4}건 ({100*v/len(tot):>4.1f}%)")
