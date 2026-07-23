import numpy as np
import pandas as pd

CORPUS_EMBEDDINGS = "artifacts/s1_v2/corpus_embeddings.npy"
CORPUS_INDEX = "artifacts/s1_v2/corpus_index.parquet"
DATASET = "artifacts/s1_v2/dataset.parquet"


class PersonalizedRetriever:
    def __init__(self, rec_boost: float = 0.03):
        self.embeddings = np.load(CORPUS_EMBEDDINGS, mmap_mode="r")
        self.index = pd.read_parquet(CORPUS_INDEX)
        self.dataset = pd.read_parquet(DATASET)
        self.rec_boost = rec_boost

        self._build_recommendations_percentile()

    def _build_recommendations_percentile(self):
        recs = self.dataset["recommendations_total"].fillna(0).values
        ranks = pd.Series(recs).rank(pct=True, ascending=True).values
        self.rec_percentile = ranks

    def retrieve(
        self,
        user_vector: np.ndarray,
        owned_appids: set[int] | None = None,
        exclude_appids: set[int] | None = None,
        top_n: int = 300,
    ) -> pd.DataFrame:
        sims = np.dot(self.embeddings, user_vector)
        top_idx = np.argsort(sims)[::-1]
        exclude = set()
        if owned_appids:
            exclude |= owned_appids
        if exclude_appids:
            exclude |= exclude_appids

        results = []
        for idx in top_idx:
            if len(results) >= top_n:
                break
            row = self.index.iloc[idx]
            appid = int(row["steam_appid"])
            if appid in exclude:
                continue
            sim = float(sims[idx])
            results.append({
                "steam_appid": appid,
                "name": row["name"],
                "cosine_similarity": sim,
            })

        df = pd.DataFrame(results)

        rec_percentiles = self.rec_percentile[[
            self.index[self.index["steam_appid"] == appid].index[0]
            for appid in df["steam_appid"]
        ]]
        df["recommendations_percentile"] = rec_percentiles
        df["final_score"] = df["cosine_similarity"] * (
            1 + df["recommendations_percentile"] * self.rec_boost
        )
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df
