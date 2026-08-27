"""웹소설 추천기 평가 하네스 — Steam·TMDB 와 같은 기준(52프로필 × k=50 · 0.85).

왜 프로덕션 랭커를 안 건드리고 여기서 점수를 다시 계산하는가:
측정으로 이길 조합이 정해지기 전에 서빙 경로를 바꾸지 않는다. 확정된 뒤에
`PersonalizedRanker` 로 옮긴다. (TMDB 에서 λ 가 해로울 거라 적어놓고 측정에서
+4.1%p 도움으로 뒤집힌 전례가 있다 — 기본값을 끄고 재는 쪽이 옳았다.)

**등급 은행은 `(profile_id, item_id)` 로 키를 잡는다.** 이름이 아니다(D-20).
변형끼리 후보가 겹치면 채점을 재사용하고, **미채점이 0 이 되기 전에는 비교하지 않는다.**
미채점을 집계에서 빼면 새 후보를 많이 가져오는 변형일수록 기저선과 겹치는
부분만으로 평가되어 편향된다.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import artifact_dir, PRODUCTION  # noqa: E402
from src.postprocess import postprocess as pp  # noqa: E402

ART = artifact_dir()
P1 = ART.parent / "p1"


# ── 파이프라인 ───────────────────────────────────────────────────────────────
class Engine:
    def __init__(self) -> None:
        self.emb = np.asarray(np.load(ART / "corpus_embeddings.npy"), dtype=np.float32)
        self.ix = pd.read_parquet(ART / "corpus_index.parquet")
        self.ds = pd.read_parquet(ART / "dataset.parquet")
        self.id_to_row = dict(zip(self.ix["item_id"], self.ix["embedding_row"]))
        self.centroid = self.emb.mean(axis=0)

        # 관심 수: 0 은 결측이 아니라 진짜 무명이다. 확인함 —
        # interest=0 집단은 화수 중앙 6(vs 145) · 댓글 중앙 0(vs 32) · 평점 미보고 61%(vs 2.5%).
        # 따라서 하위에 놓는 것이 옳다. Steam 의 87% 미보고와는 다른 상황이다.
        self.pop_pct = self.ds["interest_count"].rank(pct=True, ascending=True).to_numpy()

        # 평점: **0 은 "0점"이 아니라 "평점 없음"이다**(1,207건 · 17.1%).
        # 그대로 (r-5)/5 를 쓰면 -1.0 이 되어 평가가 없다는 이유로 최대 감점을 받는다.
        # 이건 Steam D-6 과 같은 모양이므로 중립(0.0)으로 둔다.
        r = self.ds["rating"].to_numpy(dtype=np.float32)
        self.rating_norm = np.where(r > 0, np.clip((r - 5.0) / 5.0, -1.0, 1.0), 0.0)

    def mmr(self, df: pd.DataFrame, k: int, lam: float) -> pd.DataFrame:
        """MMR — 관련성과 **이미 고른 것과의 중복** 을 맞바꾼다 (D-53).

            argmax_c  λ·rel(c) − (1−λ)·max_{s∈선택됨} cos(c, s)

        `rel` 은 풀 안에서 min-max 정규화한 현행 점수다. 두 항의 눈금을 맞추지
        않으면 λ 가 해석 불가능해진다 — `final_score` 는 보정으로 1 을 넘을 수 있는데
        cos 는 그렇지 않다.

        **ILS 를 낮추는 것 자체가 목적이 아니다.** §20·§21 에서 `quality_w` 를
        양방향으로 움직였을 때 ILS 가 좋아 보이는 쪽이 적합률은 나빴다. 주장할 수 있는
        것은 "적합 개수를 유지한 채 중복만 줄였다"뿐이고, 그건 P@k 관문이 지킨다.
        """
        if df.empty or lam >= 1.0:
            return df.head(k).reset_index(drop=True)
        rows = [self.id_to_row[i] for i in df["item_id"]]
        V = self.emb[rows]
        V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
        rel = df["final_score"].to_numpy(dtype=np.float32)
        lo, hi = float(rel.min()), float(rel.max())
        rel = (rel - lo) / (hi - lo) if hi > lo else np.zeros_like(rel)
        n = len(df)
        picked, maxsim = [], np.zeros(n, dtype=np.float32)
        alive = np.ones(n, dtype=bool)
        for _ in range(min(k, n)):
            score = lam * rel - (1.0 - lam) * maxsim
            score[~alive] = -np.inf
            j = int(np.argmax(score))
            picked.append(j); alive[j] = False
            maxsim = np.maximum(maxsim, V @ V[j])
        return df.iloc[picked].reset_index(drop=True)

    def mmr_by_seed(self, df: pd.DataFrame, lam: float) -> pd.DataFrame:
        """`dominant_seed` 묶음 **안에서만** MMR 로 재정렬한다 (D-54).

        D-53 에서 전역 MMR 은 `interleave_by_seed` 를 밀어냈고, top-50 이 덮는 시드
        종류가 twenty_library 에서 **19 → 9** 로 반토막 나면서 목록이 **더** 뭉쳤다.
        시드 커버리지와 임베딩 분산은 다른 다양성이다. 여기서는 각 묶음의 **원소
        집합을 바꾸지 않고 순서만** 바꾸므로 커버리지가 설계상 보존된다.
        """
        if df.empty or lam >= 1.0 or "dominant_seed" not in df:
            return df
        parts = []
        for _, g in df.groupby("dominant_seed", sort=False):
            parts.append(self.mmr(g.reset_index(drop=True), len(g), lam))
        # 묶음 **순서**는 손대지 않는다. `interleave_by_seed` 는 groupby(sort=False) 로
        # 첫 등장 순서를 버킷 우선순위로 쓰므로, 여기서 묶음을 재정렬하면 어느 시드가
        # 1위 자리를 갖는지가 바뀐다. 위 루프도 sort=False 라 순서가 보존된다.
        return pd.concat(parts).reset_index(drop=True)

    def recommend(self, seed_ids, *, strategy="top2_mean",
                  # **`None` 이면 `config.PRODUCTION` 의 확정값**(D-66).
                  # 예전 기본값 0.0 이 이 로그의 모든 웹소설 숫자를 만들었고,
                  # 서빙은 0.03/0.15 로 돌고 있었다. 갈라진 채로 두지 않는다.
                  pop_boost=None,
                  rating_boost=0.0, hub_lambda=0.0, k=10,
                  # `mmr_lambda` 는 **기각된 축이다** (D-53 전역 · D-54 시드묶음).
                  # 1.0 = 끔. 켜지 않는다. 재현·재검증용으로만 남긴다.
                  mmr_lambda=1.0,
                  postprocess_on=True, exclude=None) -> pd.DataFrame:
        pop_boost = PRODUCTION["pop_boost"] if pop_boost is None else pop_boost
        rows = [self.id_to_row[i] for i in seed_ids]
        V = self.emb[rows]
        if hub_lambda:
            # 재정규화하지 않는다. h67 에서 재정규화가 h62 파이프라인을 재현하지
            # 못해 30건 중 18건이 어긋난 적이 있다.
            V = V - hub_lambda * self.centroid[None, :]
        sim = V @ self.emb.T                                   # (시드, 코퍼스)

        if strategy == "max":
            s = sim.max(axis=0)
        elif strategy == "mean":
            s = sim.mean(axis=0)
        elif strategy == "top2_mean":
            s = sim.mean(axis=0) if sim.shape[0] <= 2 else np.sort(sim, axis=0)[-2:].mean(axis=0)
        else:
            raise ValueError(strategy)

        final = s * (1 + self.pop_pct * pop_boost + self.rating_norm * rating_boost)

        df = self.ds[["item_id", "name"]].copy()
        df["seed_similarity"] = s
        df["final_score"] = final
        # dominant_seed 는 점수와 무관하게 항상 채운다 — 인터리빙이 이걸로 묶는다
        df["dominant_seed"] = [seed_ids[i] for i in sim.argmax(axis=0)]

        drop = set(seed_ids) | set(exclude or ())
        df = df[~df["item_id"].isin(drop)]
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        if postprocess_on:
            from src.postprocess import drop_seed_series
            out = drop_seed_series(df.head(k * 8), self.ds, seed_ids)
            out = self.mmr_by_seed(out, mmr_lambda)      # D-54. 묶음 **안**에서만
            return pp(out, self.ds, top_n=k).head(k).reset_index(drop=True)
        return self.mmr(df.head(k * 8), k, mmr_lambda)


# ── 프로필 / 등급 은행 ───────────────────────────────────────────────────────
def load_profiles(split=None) -> pd.DataFrame:
    p = pd.read_parquet(P1 / "profiles.parquet")
    if split and split not in ("all", ""):
        p = p[p["split"] == split]
    return p.reset_index(drop=True)


def load_bank() -> dict:
    f = P1 / "grades.py"
    if not f.exists():
        return {}
    ns: dict = {}
    exec(f.read_text(encoding="utf-8"), ns)
    return {tuple(k.split("\t")): v for k, v in ns["BANK"].items()}


def save_bank(bank: dict) -> None:
    lines = ["# 등급 은행 — (profile_id, item_id) → 등급 (3/2/1/0, 적합=2 이상)",
             "BANK = {"]
    for (pid, iid), g in sorted(bank.items()):
        lines.append(f'    "{pid}\\t{iid}": {g},')
    lines.append("}")
    (P1 / "grades.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def variant_recs(variant: dict, k: int, profiles: pd.DataFrame, eng: Engine) -> dict:
    # 지정하지 않은 축은 **확정값**(config.PRODUCTION)이 들어간다 (D-66).
    # 예전에는 pop_boost 기본이 0.0 이라 하네스가 서빙과 다른 것을 재고 있었다.
    v = {k: PRODUCTION[k] for k in ("strategy", "pop_boost", "rating_boost",
                                    "hub_lambda", "mmr_lambda")}
    v.update(variant)
    out = {}
    for _, p in profiles.iterrows():
        out[p["profile_id"]] = eng.recommend(list(p["seed_ids"]), k=k, **v)
    return out


def intra_list_similarity(vecs) -> float:
    """리스트 내부 유사도(ILS) — top-k 임베딩의 대각 제외 평균 쌍유사도.

    **P@k 하나로는 부족하다(D-32).** 적합률은 "같은 책 10권"을 만점으로 센다.
    웹소설 52프로필 실측에서 corr(적합률, ILS) = **+0.308** — 지표가 중복을
    보상한다. 적합률 1.00 인 rule_rf_none 은 ILS 0.704 인데, 적합률 0.80 인
    coh_talent 는 0.586 으로 **후자가 추천으로서 더 낫다.**

    그래서 적합률과 항상 같이 낸다. 낮을수록 다양하다.
    """
    import numpy as _np
    if vecs is None or len(vecs) < 2:
        return float("nan")
    V = _np.asarray(vecs, dtype=_np.float32)
    V = V / _np.clip(_np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    S = V @ V.T
    n = len(V)
    return float((S.sum() - _np.trace(S)) / (n * (n - 1)))


def score(recs: dict, bank: dict, k: int, vec_of=None) -> dict:
    """`vec_of(ids) -> (n, d)` 를 주면 프로필별 ILS 도 같이 낸다 (D-32)."""
    per, ungraded, ils = {}, 0, {}
    for pid, df in recs.items():
        if vec_of is not None:
            try: ils[pid] = intra_list_similarity(vec_of(list(df["item_id"])[:k]))
            except Exception: ils[pid] = float("nan")
        gs = []
        for iid in df["item_id"]:
            g = bank.get((pid, str(iid)))
            if g is None:
                ungraded += 1
            else:
                gs.append(g)
        per[pid] = (np.mean([x >= 2 for x in gs]) if gs else 0.0,
                    np.mean(gs) if gs else 0.0, len(gs))
    fit = float(np.mean([v[0] for v in per.values()]))
    mg = float(np.mean([v[1] for v in per.values()]))
    out = {"fit": fit, "mean_grade": mg, "ungraded": ungraded,
           "graded": sum(v[2] for v in per.values()), "per": per}
    if ils:
        vals = [v for v in ils.values() if v == v]
        out["ils"] = float(np.mean(vals)) if vals else float("nan")
        out["per_ils"] = ils
    return out
