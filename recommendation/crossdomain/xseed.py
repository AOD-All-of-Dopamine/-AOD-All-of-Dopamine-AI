"""X-16 변형 생성기 — 세 플랫폼을 **모델 없이** 한 프로세스에서 돌린다.

    M6 : 그 플랫폼 시드만          · PRODUCTION 정렬   (현행 재현)
    C1 : 모든 플랫폼 시드 전부     · PRODUCTION 정렬   (교차 시드)
    C2 : 모든 플랫폼 시드 전부     · 순수 유사도 (허브 보정·보정항 없음)

세 저장소가 전부 `src` 패키지라 같은 프로세스에서 이름이 충돌한다. 플랫폼별 모듈 집합을
따로 들고 있다가 호출 직전에 `sys.modules` 를 바꿔 끼운다(`_swap`). 함수 안의 지연 임포트
(`from src.postprocess import …`)까지 그 플랫폼 것으로 풀리게 하려면 이 방법뿐이다.

시드 결합(교차 시드일 때):
    a) raw : 시드별 유사도 그대로
    b) z   : 시드별 유사도를 코퍼스 전체에 대해 표준화한 뒤, 그 플랫폼 자기 시드들의
             (평균, 표준편차) 눈금으로 되돌린다 — 도메인 안/밖 시드의 스케일을 맞추면서
             랭커가 기대하는 양수 범위를 유지한다.
"""
from __future__ import annotations
import os, sys, contextlib
from pathlib import Path
import numpy as np, pandas as pd

AOD = Path("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
WN = Path("/home/ubuntu/aod-webnovel/recommendation/webnovel")
ART = {"steam": AOD / "steam/artifacts/tags_full", "tmdb": AOD / "tmdb/artifacts/tmdb_v1",
       "wn": WN / os.environ.get("AOD_WN_ARTIFACTS", "artifacts/wn_v4")}
_MODS: dict[str, dict] = {}


def _load_platform(name: str, root: Path, artifacts: Path):
    """`root` 의 src 패키지를 임포트해 모듈 집합을 저장하고 sys.modules 에서 걷어낸다."""
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    sys.path.insert(0, str(root))
    os.environ["AOD_ARTIFACTS"] = str(artifacts)
    cwd = os.getcwd(); os.chdir(root)
    try:
        if name == "steam":
            from src.personalized_retrieve import build_components
            from src.personalization.personalized_ranker import PersonalizedRanker
            from src.postprocess import postprocess
            comps = build_components(artifacts=str(artifacts))
            pure = PersonalizedRanker(rec_boost=0.0, artifacts=str(artifacts), quality_w=0.0,
                                      tag_w=0.0, mc_w=0.0)
            obj = dict(comps=comps, pure_ranker=pure, postprocess=postprocess)
        elif name == "tmdb":
            from src.personalized_retrieve import build_components
            from src.personalization.personalized_ranker import PersonalizedRanker
            from src.postprocess import postprocess
            from src.config import PRODUCTION, PRODUCTION_POSTPROCESS
            comps = build_components(artifacts=str(artifacts),
                                     **{k: v for k, v in PRODUCTION.items() if k != "strategy"})
            pure = PersonalizedRanker(str(artifacts))
            obj = dict(comps=comps, pure_ranker=pure, postprocess=postprocess,
                       PROD=PRODUCTION, PP=PRODUCTION_POSTPROCESS)
        else:
            from src.wn_eval import Engine, pp
            from src.postprocess import drop_seed_series
            from src.config import PRODUCTION
            obj = dict(eng=Engine(), pp=pp, drop_seed_series=drop_seed_series, PROD=PRODUCTION)
    finally:
        os.chdir(cwd); sys.path.pop(0)
    obj["modules"] = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in obj["modules"]: del sys.modules[k]
    _MODS[name] = obj
    return obj


@contextlib.contextmanager
def _swap(name: str):
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    sys.modules.update(_MODS[name]["modules"])
    try:
        yield _MODS[name]
    finally:
        for k in _MODS[name]["modules"]: sys.modules.pop(k, None)


def load_all():
    if "steam" not in _MODS: _load_platform("steam", AOD / "steam", ART["steam"])
    if "tmdb" not in _MODS: _load_platform("tmdb", AOD / "tmdb", ART["tmdb"])
    if "wn" not in _MODS: _load_platform("wn", WN, ART["wn"])


# ── 시드 벡터 ────────────────────────────────────────────────────────────────
def seed_vec(plat: str, sid) -> np.ndarray:
    """플랫폼 코퍼스 임베딩에서 시드 벡터(원시, 보정 없음)를 꺼낸다."""
    m = _MODS[plat]
    if plat == "steam":
        return m["comps"][0].load([int(sid)])[int(sid)]
    if plat == "tmdb":
        return m["comps"][0].load([int(sid)])[int(sid)]
    eng = m["eng"]
    return eng.emb[eng.id_to_row[int(sid)]].copy()


def build_seed_dict(target: str, seeds: dict[str, list], cross: bool) -> dict:
    """{키: 벡터}. 자기 플랫폼 시드는 원래 id 를 키로(태그·장르·후처리가 그걸 본다),
    외부 시드는 'plat:id' 문자열 키. dominant_seed 는 어느 키든 상관없다."""
    d = {}
    for sid in seeds.get(target, []):
        d[int(sid)] = seed_vec(target, sid)
    if cross:
        for p, ids in seeds.items():
            if p == target: continue
            for sid in ids:
                d[f"{p}:{sid}"] = seed_vec(p, sid)
    return d


def _zscale(sim: np.ndarray, own_rows: list[int]) -> np.ndarray:
    """(b) z: 행별 표준화 후 자기 시드들의 눈금으로 복원."""
    mu = sim.mean(axis=1, keepdims=True); sd = sim.std(axis=1, keepdims=True) + 1e-9
    z = (sim - mu) / sd
    m_own = float(mu[own_rows].mean()); s_own = float(sd[own_rows].mean())
    return m_own + s_own * z


# ── 플랫폼별 top-k ────────────────────────────────────────────────────────────
def rank_steam(seeds, variant: str, combine: str = "a", k: int = 50) -> list[int]:
    own = [int(x) for x in seeds.get("steam", [])]
    if not own: return []
    cross, pure = variant in ("C1", "C2"), variant == "C2"
    with _swap("steam") as m:
        loader, retriever, agg, ranker = m["comps"]
        sd = build_seed_dict("steam", seeds, cross)
        sim = retriever.compute_similarity_matrix(sd, hub_lambda=(0.0 if pure else None))
        if cross and combine == "b":
            sim = _zscale(sim, [i for i, key in enumerate(sd) if isinstance(key, int)])
        fr = agg.aggregate_all(sim, sd, retriever.full_corpus_frame(), strategies=["top2_mean"])["top2_mean"]
        rk = (m["pure_ranker"] if pure else ranker)
        ranked = rk.rank(fr, exclude_appids=set(own), top_n=k * 5, seed_appids=own)
        out = m["postprocess"](ranked, ranker.dataset.reset_index(), top_n=k, seed_appids=own)
        col = next(c for c in ("steam_appid", "appid", "candidate_appid") if c in out.columns)
        return [int(x) for x in out[col].head(k)]


def rank_tmdb(seeds, variant: str, combine: str = "a", k: int = 50) -> list[int]:
    own = [int(x) for x in seeds.get("tmdb", [])]
    if not own: return []
    cross, pure = variant in ("C1", "C2"), variant == "C2"
    with _swap("tmdb") as m:
        loader, retriever, agg, ranker = m["comps"]
        sd = build_seed_dict("tmdb", seeds, cross)
        sim = retriever.compute_similarity_matrix(sd, hub_lambda=(0.0 if pure else None))
        if cross and combine == "b":
            sim = _zscale(sim, [i for i, key in enumerate(sd) if isinstance(key, int)])
        strategy = "top2_mean" if pure else m["PROD"]["strategy"]
        fr = agg.aggregate_all(sim, sd, retriever.full_corpus_frame(), strategies=[strategy])[strategy]
        rk = (m["pure_ranker"] if pure else ranker)
        seed_pct = float(np.median(ranker.vote_pct[own]))
        seed_medias = {ranker.dataset.iloc[r]["media"] for r in own}
        seed_genres = None
        if rk.genre_w:
            seed_genres = []
            for rr in own:
                g = ranker.dataset.iloc[rr]["genres"]
                seed_genres.append(frozenset(g.tolist() if hasattr(g, "tolist") else (g or [])))
        ranked = rk.rank(fr, exclude_rows=set(own), top_n=k * 8, servable_mask=retriever.servable,
                         seed_pct=seed_pct, seed_medias=seed_medias, seed_genres=seed_genres)
        out = m["postprocess"](ranked, ranker.dataset, top_n=k, seed_rows=own, **m["PP"])
        return [int(x) for x in out["row"].head(k)]


def rank_wn(seeds, variant: str, combine: str = "a", k: int = 50) -> list[int]:
    own = [int(x) for x in seeds.get("wn", [])]
    if not own: return []
    cross, pure = variant in ("C1", "C2"), variant == "C2"
    with _swap("wn") as m:
        eng = m["eng"]
        sd = build_seed_dict("wn", seeds, cross)
        keys = list(sd); V = np.array([sd[key] for key in keys], dtype=np.float32)
        sim = V @ eng.emb.T                       # PRODUCTION hub_lambda = 0
        if cross and combine == "b":
            sim = _zscale(sim, [i for i, key in enumerate(keys) if isinstance(key, int)])
        s = sim.mean(axis=0) if sim.shape[0] <= 2 else np.sort(sim, axis=0)[-2:].mean(axis=0)
        pop = 0.0 if pure else m["PROD"]["pop_boost"]
        final = s * (1 + eng.pop_pct * pop)
        df = eng.ds[["item_id", "name"]].copy()
        df["seed_similarity"] = s; df["final_score"] = final
        df["dominant_seed"] = [keys[i] for i in sim.argmax(axis=0)]
        df = df[~df["item_id"].isin(set(own))].sort_values("final_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        out = m["drop_seed_series"](df.head(k * 8), eng.ds, own)
        out = eng.mmr_by_seed(out, 1.0)
        out = m["pp"](out, eng.ds, top_n=k).head(k)
        return [int(x) for x in out["item_id"]]


RANK = {"steam": rank_steam, "tmdb": rank_tmdb, "wn": rank_wn}


def wn_episodes() -> dict:
    eng = _MODS["wn"]["eng"]
    return dict(zip(eng.ds["item_id"].astype(int), eng.ds["episode_count"].fillna(0).astype(int)))


def platform_lists(seeds: dict, variant: str, combine: str = "a", k: int = 50) -> dict:
    return {p: RANK[p](seeds, variant, combine, k) for p in ("steam", "tmdb", "wn") if seeds.get(p)}


def mix_m6(lists: dict, seeds: dict, k: int = 50, episodes: dict | None = None):
    sys.path.insert(0, str(AOD / "crossdomain"))
    from mix import RULES
    eps = episodes if episodes is not None else wn_episodes()
    n = {p: len(seeds[p]) for p in lists}
    return [dict(plat=p, item=str(it), rank=rk)
            for p, it, rk in RULES["M6"](lists, n, {p: 0.6 for p in lists}, k=k, episodes=eps)]
