"""시드 → 후보. 시드별 질의(평균 내지 않음) → 접기 → 정렬 → 후처리.

`next_page()` 계약은 세 플랫폼과 같다: 이미 보여준 것을 `exclude` 로 받아 그 아래를 잇는다.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from src.config import artifact_dir, PRODUCTION, POSTPROCESS
from src.personalization.personalized_ranker import PersonalizedRanker
from src.postprocess import postprocess


class Engine:
    def __init__(self, artifacts=None):
        d = artifact_dir(artifacts)
        self.emb = np.load(d / "corpus_embeddings.npy").astype(np.float32)
        self.emb /= (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9)
        self.ds = pd.read_parquet(d / "dataset.parquet").reset_index(drop=True)
        self.row = {int(v): i for i, v in enumerate(self.ds["item_id"])}
        self.centroid = self.emb.mean(axis=0)
        self.centroid /= (np.linalg.norm(self.centroid) + 1e-9)
        self._tags = [frozenset(t if t is not None and not isinstance(t, str) else ([t] if t else []))
                      for t in self.ds.get("tags", pd.Series([None] * len(self.ds)))]

    # ── 검색 ──
    def similarity(self, seed_ids, hub_lambda=None) -> np.ndarray:
        """시드 × 코퍼스 행렬. 시드마다 따로 — 취향이 갈리는 사람은 평균이 아무 데도 안 가리킨다."""
        hl = PRODUCTION["hub_lambda"] if hub_lambda is None else hub_lambda
        V = []
        for s in seed_ids:
            v = self.emb[self.row[int(s)]].copy()
            if hl:
                v = v - hl * self.centroid
                v /= (np.linalg.norm(v) + 1e-9)
            V.append(v)
        return np.array(V, dtype=np.float32) @ self.emb.T

    @staticmethod
    def fold(sim: np.ndarray, strategy: str):
        """접기: 시드별 유사도를 후보 하나의 점수로."""
        if strategy == "mean" or sim.shape[0] <= 2:
            return sim.mean(axis=0)
        if strategy == "max":
            return sim.max(axis=0)
        return np.sort(sim, axis=0)[-2:].mean(axis=0)      # top2_mean

    def recommend(self, seed_ids, k=50, exclude=None, *, strategy=None, pop_boost=None,
                  star_boost=None, tag_w=None, hub_lambda=None, **pp):
        seed_ids = [int(s) for s in seed_ids if int(s) in self.row]
        if not seed_ids:
            return pd.DataFrame(columns=["item_id", "name", "rank"])
        strategy = strategy or PRODUCTION["strategy"]
        sim = self.similarity(seed_ids, hub_lambda)
        folded = self.fold(sim, strategy)
        dominant = [seed_ids[i] for i in sim.argmax(axis=0)]
        seed_tags = [self._tags[self.row[s]] for s in seed_ids] if (tag_w or PRODUCTION["tag_w"]) else None
        r = PersonalizedRanker(
            self.ds,
            pop_boost=PRODUCTION["pop_boost"] if pop_boost is None else pop_boost,
            star_boost=PRODUCTION["star_boost"] if star_boost is None else star_boost,
            tag_w=PRODUCTION["tag_w"] if tag_w is None else tag_w,
        )
        ex = set(seed_ids) | set(int(x) for x in (exclude or []))
        ranked = r.rank(folded, exclude_ids=ex, seed_tags=seed_tags, top_n=k * 8, dominant=dominant)
        opts = {**POSTPROCESS, **pp}
        return postprocess(ranked, self.ds, top_n=k, seed_ids=seed_ids,
                           series_max=opts["series_max"], artist_max=opts["artist_max"],
                           drop_adult=opts["drop_adult"])

    def next_page(self, seed_ids, k=50, seen=None, **kw):
        """제품 경로. `seen` 아래를 잇는다 — 새로고침해도 앞 페이지가 다시 나오지 않는다."""
        return self.recommend(seed_ids, k=k, exclude=seen or [], **kw)
