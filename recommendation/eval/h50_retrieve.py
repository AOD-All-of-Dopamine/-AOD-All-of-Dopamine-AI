"""h50 검색 — 웹소설 신규 20시드 → Steam · TMDB 두 target.
사전 등록 md5 3a72b9be3a1378e356b9a1927abd52f2. 파이프라인은 h48 과 완전 동일(λ=0.35, top-10)."""
import json, collections
import numpy as np, pandas as pd

SP    = "/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad"
STEAM = "/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/artifacts/tags_full"
TMDB  = "/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb/artifacts/tmdb_v1"
WN    = "/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4"
LAM   = 0.35
ADULT_GENRES = {"신체 노출", "선정적 콘텐츠"}
ADULT_DESC   = {3, 4}
QUOTA = {"판타지":6, "현판":4, "로판":4, "무협":3, "로맨스":1, "미스터리":1, "라이트노벨":1}

# ---------- 시드 ----------
wd = pd.read_parquet(f"{WN}/dataset.parquet", columns=["item_id","name","genres","synopsis"])
wi = pd.read_parquet(f"{WN}/corpus_index.parquet").sort_values("embedding_row")
wd = wd.set_index("item_id").loc[wi["item_id"].to_numpy()].reset_index()
used = set(json.load(open(f"{SP}/used_seeds_h50.json")))
wd["g"] = wd["genres"].map(lambda v: str(v[0]) if v is not None and len(v) else "")
fresh = wd[~wd["name"].map(lambda n: any(str(n).startswith(u) or u.startswith(str(n)[:20]) for u in used))]
fresh = fresh[fresh["synopsis"].str.len() >= 100]
print(f"웹소설 {len(wd):,} → 기존 시드 {len(used)}개 제외·줄거리 100자 이상 → {len(fresh):,}")

rng = np.random.default_rng(20260901)
pick = []
for g, q in QUOTA.items():
    pool = fresh[fresh["g"] == g]
    if len(pool) < q: raise SystemExit(f"장르 {g} 부족: {len(pool)} < {q}")
    pick += list(rng.choice(pool.index.to_numpy(), q, replace=False))
pick = sorted(pick)
print(f"\n신규 시드 {len(pick)}개 (난수 20260901):")
for i in pick: print(f"  [{wd.loc[i,'g']:6s}] {wd.loc[i,'name']}")

Ew = np.load(f"{WN}/corpus_embeddings.npy", mmap_mode="r")
Qbase = np.asarray(Ew[pick], dtype=np.float32)

def retrieve(E, keep, k=10, ch=20000):
    cen = np.zeros(E.shape[1], dtype=np.float64)
    for i in range(0, len(E), ch): cen += np.asarray(E[i:i+ch], dtype=np.float64).sum(axis=0)
    cen = (cen/len(E)).astype(np.float32)
    Q = Qbase - LAM*cen[None, :]
    best = np.full((len(pick), k), -1e9, np.float32); bidx = np.zeros((len(pick), k), np.int64)
    for i in range(0, len(E), ch):
        blk = np.asarray(E[i:i+ch], dtype=np.float32)
        s = Q @ blk.T
        s[:, ~keep[i:i+blk.shape[0]]] = -1e9
        cv = np.concatenate([best, s], 1)
        ci = np.concatenate([bidx, np.arange(i, i+blk.shape[0])[None,:].repeat(len(pick),0)], 1)
        t = np.argsort(-cv, 1)[:, :k]
        best = np.take_along_axis(cv, t, 1); bidx = np.take_along_axis(ci, t, 1)
    return best, bidx

# ---------- Steam ----------
sd = pd.read_parquet(f"{STEAM}/dataset.parquet",
        columns=["steam_appid","name","genres","content_descriptorids","semantic_text"])
si = pd.read_parquet(f"{STEAM}/corpus_index.parquet").sort_values("embedding_row")
sd = sd.set_index("steam_appid").loc[si["steam_appid"].to_numpy()].reset_index()
def is_adult(g,d):
    if g is not None and set(map(str,g)) & ADULT_GENRES: return True
    if d is not None and {int(x) for x in d} & ADULT_DESC: return True
    return False
sk = ~np.array([is_adult(g,d) for g,d in zip(sd["genres"], sd["content_descriptorids"])])
print(f"\nSteam {len(sd):,} → 성인물 {int((~sk).sum()):,} 제외 → {int(sk.sum()):,}")
Es = np.load(f"{STEAM}/corpus_embeddings.npy", mmap_mode="r")
sbest, sidx = retrieve(Es, sk)

# ---------- TMDB (등록된 제한: 한국어 줄거리 · 100자 이상 · adult=False) ----------
td = pd.read_parquet(f"{TMDB}/dataset.parquet",
        columns=["item_id","name","media","overview","overview_len","adult","semantic_text"])
ti = pd.read_parquet(f"{TMDB}/corpus_index.parquet").sort_values("embedding_row")
td = td.set_index("item_id").loc[ti["item_id"].to_numpy()].reset_index()
tk = (td["overview"].fillna("").str.contains(r"[가-힣]").to_numpy()
      & (td["overview_len"] >= 100).to_numpy()
      & (~td["adult"].fillna(False).to_numpy()))
print(f"TMDB {len(td):,} → 한국어·100자·비성인 → {int(tk.sum()):,}")
Et = np.load(f"{TMDB}/corpus_embeddings.npy", mmap_mode="r")
tbest, tidx = retrieve(Et, tk)

# ---------- 저장 ----------
out = []
for r, w in enumerate(pick):
    out.append({
        "g": wd.loc[w,"g"], "wn": wd.loc[w,"name"], "syn": wd.loc[w,"synopsis"],
        "steam": [{"r":k+1, "n": sd.loc[int(sidx[r,k]),"name"],
                   "d": sd.loc[int(sidx[r,k]),"semantic_text"], "sim": float(sbest[r,k])} for k in range(10)],
        "tmdb":  [{"r":k+1, "n": td.loc[int(tidx[r,k]),"name"], "media": td.loc[int(tidx[r,k]),"media"],
                   "d": td.loc[int(tidx[r,k]),"overview"], "sim": float(tbest[r,k])} for k in range(10)],
    })
json.dump(out, open(f"{SP}/h50_seeds20.json","w"), ensure_ascii=False, indent=1)
L=[len(x["syn"]) for x in out]
for nm in ("steam","tmdb"):
    D=[len(c["d"]) for x in out for c in x[nm]]
    print(f"  {nm:5s} 후보 텍스트 중앙 {sorted(D)[len(D)//2]} 최대 {max(D)} "
          f"{'절단 없음' if sorted(D)[len(D)//2]!=max(D) else '** 절단 의심 **'}")
print(f"  시드 줄거리 중앙 {sorted(L)[len(L)//2]} 최대 {max(L)} "
      f"{'절단 없음' if sorted(L)[len(L)//2]!=max(L) else '** 절단 의심 **'}")
print("저장 h50_seeds20.json")
