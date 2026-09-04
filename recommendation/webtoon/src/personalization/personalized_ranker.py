"""정렬 — `final = 유사도 × (1 + 보정항들)`.

세 플랫폼과 **같은 모양**이다: 곱셈이라 보정항은 더하기가 아니라 배율이다.
웹툰에서 **확정된 항은 아직 하나도 없다.** 아래는 전부 축이고, 각각 사전등록으로 판정한다.

    pop_boost   : 관심 수 백분위    (웹소설 D-66 의 0.03 을 이식, 웹툰 미측정)
    star_boost  : 평점 백분위       (웹소설에선 신호 없었다 D-62, 웹툰 미측정)
    tag_w       : 태그 피복률       (웹툰만의 기회 — 작품당 태그 10~12개. Steam D-49 형태)
                  max_i |후보태그 ∩ 시드ᵢ태그| / |시드ᵢ태그|  — 자카드가 아니다.
                  분모에 시드만 넣는다(후보의 추가 태그를 벌주지 않는다).
"""
from __future__ import annotations
import numpy as np, pandas as pd


class PersonalizedRanker:
    def __init__(self, dataset: pd.DataFrame, pop_boost=0.0, star_boost=0.0, tag_w=0.0):
        self.ds = dataset.reset_index(drop=True)
        self.pop_boost, self.star_boost, self.tag_w = pop_boost, star_boost, tag_w
        fav = pd.to_numeric(self.ds.get("favorite_count"), errors="coerce").fillna(0.0)
        self.pop_pct = fav.rank(pct=True).to_numpy(dtype=np.float32)
        star = pd.to_numeric(self.ds.get("star_score"), errors="coerce").fillna(0.0)
        self.star_pct = star.rank(pct=True).to_numpy(dtype=np.float32)
        self._tags = [frozenset(t if t is not None and not isinstance(t, str) else ([t] if t else []))
                      for t in self.ds.get("tags", pd.Series([None] * len(self.ds)))]

    def rank(self, sim: np.ndarray, *, exclude_ids=None, seed_tags=None, top_n=200,
             dominant=None) -> pd.DataFrame:
        """`sim` 은 후보별 접힌 유사도(행 = 코퍼스 순서)."""
        score = sim.astype(np.float32).copy()
        boost = np.zeros_like(score)
        if self.pop_boost:  boost += self.pop_boost * self.pop_pct
        if self.star_boost: boost += self.star_boost * self.star_pct
        if self.tag_w and seed_tags:
            cov = np.array([max((len(self._tags[i] & s) / len(s) if s else 0.0) for s in seed_tags)
                            for i in range(len(self.ds))], dtype=np.float32)
            boost += self.tag_w * cov
        final = score * (1.0 + boost)
        out = pd.DataFrame({"item_id": self.ds["item_id"].values, "name": self.ds["name"].values,
                            "seed_similarity": score, "final_score": final})
        if dominant is not None: out["dominant_seed"] = dominant
        if exclude_ids: out = out[~out["item_id"].isin(set(int(x) for x in exclude_ids))]
        out = out.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
        out["rank"] = range(1, len(out) + 1)
        return out
