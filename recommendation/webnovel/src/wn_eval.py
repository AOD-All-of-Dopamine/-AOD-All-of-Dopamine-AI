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
from src.config import artifact_dir  # noqa: E402
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

    def recommend(self, seed_ids, *, strategy="top2_mean", pop_boost=0.0,
                  rating_boost=0.0, hub_lambda=0.0, k=10,
                  postprocess_on=True, exclude=None) -> pd.DataFrame:
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
            return pp(df.head(k * 8), self.ds, top_n=k).head(k).reset_index(drop=True)
        return df.head(k).reset_index(drop=True)


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
    v = {"strategy": "top2_mean", "pop_boost": 0.0, "rating_boost": 0.0, "hub_lambda": 0.0}
    v.update(variant)
    out = {}
    for _, p in profiles.iterrows():
        out[p["profile_id"]] = eng.recommend(list(p["seed_ids"]), k=k, **v)
    return out


def score(recs: dict, bank: dict, k: int) -> dict:
    per, ungraded = {}, 0
    for pid, df in recs.items():
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
    return {"fit": fit, "mean_grade": mg, "ungraded": ungraded,
            "graded": sum(v[2] for v in per.values()), "per": per}
