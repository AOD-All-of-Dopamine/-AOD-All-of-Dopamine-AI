"""개인화 랭킹 — 유사도가 주항, 인기도·평점이 보정항.

    final = seed_similarity × (1 + vote_pct×vote_boost + rating_norm×rating_boost)

곱셈이므로 **유사하지 않은 작품은 아무리 인기 있어도 못 올라온다.**

────────────────────────────────────────────────────────────────────────────
**Steam 의 인기도 백분위 결함이 TMDB 에는 없다.**

Steam: `recommendations_total` 이 87% 결측 → `fillna(0).rank(pct=True)` 가
미보고 151,799건을 동점으로 눕히고 pandas 가 평균 순위를 준다. 결과가
  미보고 0.437 · 리뷰 101개 0.879 · 520만개 1.000 — **리뷰 101~520만이 폭 0.12 안에.**
사실상 "미보고냐 아니냐" 이진 플래그였고, `rec_boost` 셋이 그 결함 위에서 맞춰졌다.

TMDB: `vote_count` **비결측 100% · 0인 것 0건.** 분위가
  p10 35 · p25 46 · p50 87 · p75 248 · p90 892 · p99 7,792 · 최대 40,789.
⇒ 백분위가 **전 구간에서 해상도를 갖는다.** Steam 이 못 한 것을 여기서는 할 수 있다.

`vote_average` 도 100% 비결측(중앙 6.5)이라 품질 신호를 따로 쓸 수 있다.
Steam 의 metacritic 은 2.4% 밖에 없어서 못 썼다.
────────────────────────────────────────────────────────────────────────────

보정 계수는 **측정 전까지 0.0** 이다. 근거 없이 켜지 않는다.
"""
from pathlib import Path
import numpy as np, pandas as pd
from src.config import artifact_dir


class PersonalizedRanker:
    def __init__(self, artifacts=None, vote_boost: float = 0.0, rating_boost: float = 0.0):
        d = artifact_dir(artifacts)
        idx = pd.read_parquet(d / "corpus_index.parquet").sort_values("embedding_row")
        ds = pd.read_parquet(d / "dataset.parquet").set_index("item_id")
        self.dataset = ds.loc[idx["item_id"].to_numpy()].reset_index()
        self.dataset["row"] = np.arange(len(self.dataset))
        self.vote_boost = vote_boost
        self.rating_boost = rating_boost
        vc = self.dataset["vote_count"].astype(float)
        self.vote_pct = vc.rank(pct=True).to_numpy()          # 결측 없음 → 눕힐 것이 없다
        va = self.dataset["vote_average"].astype(float)
        self.rating_norm = ((va - 5.0) / 5.0).clip(-1, 1).to_numpy()   # 5점=0, 10점=1

    def rank(self, scored: pd.DataFrame, exclude_rows=None, top_n: int = 300,
             servable_mask: np.ndarray | None = None) -> pd.DataFrame:
        df = scored.copy()
        r = df["row"].to_numpy()
        df["final_score"] = df["seed_similarity"].to_numpy() * (
            1.0 + self.vote_pct[r] * self.vote_boost + self.rating_norm[r] * self.rating_boost)
        if servable_mask is not None:
            df = df[servable_mask[df["row"].to_numpy()]]
        if exclude_rows:
            df = df[~df["row"].isin(set(int(x) for x in exclude_rows))]
        df = df.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df
