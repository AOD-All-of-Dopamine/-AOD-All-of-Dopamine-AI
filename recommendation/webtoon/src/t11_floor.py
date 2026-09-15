"""T-11 단계 1 — 싫어요 감점 임계값·크기 측정 (채점 없음 · 67프로필 전부). 결과 eval/t11_floor.json.

감점에 쓸 유사도는 좋아요와 같은 `Engine.similarity` (허브 보정 PRODUCTION) 이다.

  표적 싫어요 d  = 프로필 1페이지 1위 (사전등록 시나리오 규칙과 같다)
  무관 싫어요 du = 장르가 다른 다음 프로필의 1페이지 1위
  후보 풀        = 1페이지를 본 뒤 다음에 나올 100개 (제품 경로 recommend k=100, exclude=1페이지)
  "닮은 작품"    = d 의 임베딩 최근접 50

  U = sim(du, 풀 전체)            — 무관 싫어요가 건드리는 유사도
  T = sim(d, 풀 ∩ 닮은 작품)      — 표적 싫어요가 밀어내야 할 유사도

임계값 규칙 (측정 전에 정함): floor = U 의 99백분위를 0.01 단위로 올림.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("AOD_WT_ARTIFACTS", str(ROOT / "artifacts/wt_v1"))
from src.personalized_retrieve import Engine
from src.config import PRODUCTION

eng = Engine()
P = json.load(open(ROOT / "eval/profiles.json"))
ids = eng.ds["item_id"].astype(int).to_numpy()
NN = 50
# 싫어요 경로가 아니라 **현행 경로**의 풀을 잰다 — 감점 설계는 감점 없는 점수 척도 위에서 한다.
assert PRODUCTION["dislike_w"] == 0.0


def neighbors(d):
    s = eng.emb @ eng.emb[eng.row[d]]
    top = np.argpartition(-s, NN + 1)[:NN + 1]
    return {int(ids[t]) for t in top} - {d}


page1, pool = {}, {}
for p in P:
    s = [int(x) for x in p["seeds"]]
    page1[p["pid"]] = [int(x) for x in eng.recommend(s, k=10)["item_id"]]
    pool[p["pid"]] = [int(x) for x in eng.recommend(s, k=100, exclude=page1[p["pid"]])["item_id"]]

U, T, prof_umax, prof_tn, gap_1_20, at, skipped = [], [], [], [], [], {r: [] for r in (1, 10, 20, 50, 100)}, 0
for i, p in enumerate(P):
    pid, seeds = p["pid"], [int(x) for x in p["seeds"]]
    rows = [eng.row[c] for c in pool[pid]]
    vs = np.sort(eng.fold(eng.similarity(seeds), PRODUCTION["strategy"])[rows])[::-1]
    for r in at: at[r].append(vs[min(r, len(vs)) - 1])
    gap_1_20.append(vs[0] - vs[min(19, len(vs) - 1)])
    d = page1[pid][0]
    q = next((o for o in P[i + 1:] + P[:i] if o.get("genre") != p.get("genre")
              and page1[o["pid"]][0] not in set(page1[pid]) | set(seeds) | set(pool[pid])), None)
    if q is None: skipped += 1; continue
    du = page1[q["pid"]][0]
    su = eng.similarity([du])[0][rows]; sd = eng.similarity([d])[0][rows]
    tgt = np.array([c in neighbors(d) for c in pool[pid]])
    U.extend(su.tolist()); prof_umax.append(float(su.max()))
    T.extend(sd[tgt].tolist()); prof_tn.append(int(tgt.sum()))

U, T = np.array(U), np.array(T)
pc = lambda a, x: float(np.percentile(a, x))
floor = float(np.ceil(pc(U, 99) * 100) / 100)
print(f"[webtoon] 프로필 {len(P)} (무관 짝 없음 {skipped}) · hub_lambda {PRODUCTION['hub_lambda']} · 풀 100")
print(f"  U 무관  쌍 {len(U):,} · p50 {pc(U,50):.3f} · p95 {pc(U,95):.3f} · p99 {pc(U,99):.3f} · 최대 {U.max():.3f}")
print(f"  T 표적  쌍 {len(T):,} (프로필당 풀 속 닮은 작품 중앙 {np.median(prof_tn):.0f}) · p10 {pc(T,10):.3f} · p25 {pc(T,25):.3f} · p50 {pc(T,50):.3f}")
print(f"  규칙 floor = ceil(p99(U)) = {floor:.2f}")
print(f"  floor 이상: T {np.mean(T >= floor):.2f} · U {np.mean(U >= floor):.3f} · 무관 싫어요가 풀 100 중 1개라도 건드리는 프로필 {np.mean(np.array(prof_umax) >= floor):.2f}")
print(f"  풀 접힌 유사도 (프로필 중앙): " + " · ".join(f"{r}위 {np.median(at[r]):.3f}" for r in at))
ex = np.clip(T - floor, 0, None); ex = ex[ex > 0]
print(f"  풀 1위−20위 간격 중앙 {np.median(gap_1_20):.3f} · floor 초과분 중앙 {np.median(ex):.3f} · p90 {pc(ex, 90):.3f}")
for w in (1, 2, 3):
    print(f"    w={w}: 감점 중앙 {w*np.median(ex):.3f} = 간격의 {w*np.median(ex)/np.median(gap_1_20):.2f}배")
rng = np.random.default_rng(11)
for n in (1, 10, 50, 200):
    dd = [int(x) for x in rng.choice(ids, n, replace=False)]
    t = time.time()
    for _ in range(5): np.clip(eng.similarity(dd) - floor, 0, None).sum(axis=0)
    print(f"  싫어요 {n:>3}개 감점 계산 {(time.time() - t) / 5 * 1000:.1f}ms")
json.dump(dict(floor=floor, U_p99=pc(U, 99), T_p50=pc(T, 50), T_above=float(np.mean(T >= floor)),
               U_above=float(np.mean(U >= floor)), gap_1_20=float(np.median(gap_1_20)), excess_p50=float(np.median(ex))),
          open(ROOT / "eval/t11_floor.json", "w"), indent=1)
