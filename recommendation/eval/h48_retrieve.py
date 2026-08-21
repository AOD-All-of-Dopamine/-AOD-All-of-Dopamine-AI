"""h48 신규 20시드 검색 — 웹소설 → Steam, 확정 파이프라인 그대로.

  · 허브니스 보정 λ=0.35  → score = (seed - λ·centroid) @ corpus.T
  · 성인물 필터는 **검색 전** (postprocess.py 정의 그대로)
  · top-10
  · 시드는 기존 224개와 겹치지 않고, G6 20시드의 장르 구성에 맞춰 층화 추출
"""
import json, collections
import numpy as np, pandas as pd

SP    = "/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
STEAM = "/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
WN    = "/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
LAM   = 0.35
ADULT_GENRES = {"신체 노출", "선정적 콘텐츠"}
ADULT_DESC   = {3, 4}
QUOTA = {"판타지":6, "현판":4, "로판":4, "무협":3, "로맨스":1, "미스터리":1, "라이트노벨":1}

sd = pd.read_parquet(f"{STEAM}/dataset.parquet",
                     columns=["steam_appid","name","genres","content_descriptorids","semantic_text"])
si = pd.read_parquet(f"{STEAM}/corpus_index.parquet").sort_values("embedding_row")
sd = sd.set_index("steam_appid").loc[si["steam_appid"].to_numpy()].reset_index()

def is_adult(g, d):
    if g is not None and set(map(str, g)) & ADULT_GENRES: return True
    if d is not None and {int(x) for x in d} & ADULT_DESC: return True
    return False
keep = ~np.array([is_adult(g, d) for g, d in zip(sd["genres"], sd["content_descriptorids"])])
print(f"Steam {len(sd):,} → 성인물 {int((~keep).sum()):,}건 제외 → {int(keep.sum()):,}")

wd = pd.read_parquet(f"{WN}/dataset.parquet", columns=["item_id","name","genres","synopsis"])
wi = pd.read_parquet(f"{WN}/corpus_index.parquet").sort_values("embedding_row")
wd = wd.set_index("item_id").loc[wi["item_id"].to_numpy()].reset_index()
used = set(json.load(open(f"{SP}/used_seeds.json")))
def g1(v): return str(v[0]) if v is not None and len(v) else ""
wd["g"] = wd["genres"].map(g1)
# 이름이 20자로 잘린 기존 기록과 대조하려면 접두로 비교한다
fresh = wd[~wd["name"].map(lambda n: any(str(n).startswith(u) or u.startswith(str(n)[:20]) for u in used))]
# 줄거리가 너무 짧으면 라벨링이 불가능하다 — webnovel 관례(min_text_chars=20)의 5배로 잡는다
fresh = fresh[fresh["synopsis"].str.len() >= 100]
print(f"웹소설 {len(wd):,} → 기존 시드 제외·줄거리 100자 이상 → {len(fresh):,}")
print("  가용 장르:", dict(collections.Counter(fresh['g']).most_common(10)))

rng = np.random.default_rng(20260819)
pick = []
for g, q in QUOTA.items():
    pool = fresh[fresh["g"] == g]
    if len(pool) < q: raise SystemExit(f"장르 {g} 부족: {len(pool)} < {q}")
    pick += list(rng.choice(pool.index.to_numpy(), q, replace=False))
pick = sorted(pick)
print(f"\n신규 시드 {len(pick)}개 (난수 20260819):")
for i in pick: print(f"  [{wd.loc[i,'g']:6s}] {wd.loc[i,'name']}")

Ew = np.load(f"{WN}/corpus_embeddings.npy", mmap_mode="r")
Es = np.load(f"{STEAM}/corpus_embeddings.npy", mmap_mode="r")
Q  = np.asarray(Ew[pick], dtype=np.float32)

# 중심은 전체 코퍼스 기준(candidate_retriever 와 동일). 청크로 누적해 메모리를 아낀다.
CH = 20000
cen = np.zeros(Es.shape[1], dtype=np.float64)
for i in range(0, len(Es), CH): cen += np.asarray(Es[i:i+CH], dtype=np.float64).sum(axis=0)
cen = (cen / len(Es)).astype(np.float32)
Q = Q - LAM * cen[None, :]

best = np.full((len(pick), 10), -1e9, dtype=np.float32)
bidx = np.zeros((len(pick), 10), dtype=np.int64)
for i in range(0, len(Es), CH):
    blk = np.asarray(Es[i:i+CH], dtype=np.float32)
    sim = Q @ blk.T
    sim[:, ~keep[i:i+CH]] = -1e9                      # 성인물은 검색 전에 제외
    cat_v = np.concatenate([best, sim], axis=1)
    cat_i = np.concatenate([bidx, np.arange(i, i+blk.shape[0])[None, :].repeat(len(pick), 0)], axis=1)
    top = np.argsort(-cat_v, axis=1)[:, :10]
    best = np.take_along_axis(cat_v, top, axis=1)
    bidx = np.take_along_axis(cat_i, top, axis=1)

out = []
for r, wrow in enumerate(pick):
    out.append({
        "g": wd.loc[wrow, "g"], "wn": wd.loc[wrow, "name"],
        "syn": wd.loc[wrow, "synopsis"],                          # 전문. 자르지 않는다
        "c": [{"r": k+1, "n": sd.loc[int(bidx[r, k]), "name"],
               "d": sd.loc[int(bidx[r, k]), "semantic_text"],     # 전문
               "sim": float(best[r, k])} for k in range(10)],
    })
json.dump(out, open(f"{SP}/h48_fresh20.json", "w"), ensure_ascii=False, indent=1)
L = [len(x["syn"]) for x in out]; D = [len(c["d"]) for x in out for c in x["c"]]
print(f"\n저장 h48_fresh20.json")
print(f"  시드 줄거리  중앙 {sorted(L)[len(L)//2]} 최대 {max(L)}   {'절단 없음' if sorted(L)[len(L)//2]!=max(L) else '** 절단 의심 **'}")
print(f"  후보 텍스트  중앙 {sorted(D)[len(D)//2]} 최대 {max(D)}   {'절단 없음' if sorted(D)[len(D)//2]!=max(D) else '** 절단 의심 **'}")
