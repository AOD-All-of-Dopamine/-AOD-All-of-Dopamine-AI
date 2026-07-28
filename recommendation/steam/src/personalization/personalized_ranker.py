from pathlib import Path

import pandas as pd

from src.config import artifact_dir


class PersonalizedRanker:
    def __init__(self, rec_boost: float = 0.03, artifacts: str | Path | None = None):
        self.artifacts = artifact_dir(artifacts)
        self.dataset = pd.read_parquet(self.artifacts / "dataset.parquet")
        self.rec_boost = rec_boost
        self.dataset = self.dataset.set_index("steam_appid")

    def rank(
        self,
        candidates: pd.DataFrame,
        exclude_appids: set[int] | None = None,
        top_n: int = 300,
    ) -> pd.DataFrame:
        result = candidates.copy()

        rec_col = self.dataset["recommendations_total"].fillna(0)
        pct = rec_col.rank(pct=True, ascending=True)

        result["recommendations_percentile"] = result["steam_appid"].map(
            lambda x: pct.get(x, 0.0)
        )
        result["final_score"] = result["seed_similarity"] * (
            1 + result["recommendations_percentile"] * self.rec_boost
        )

        if exclude_appids:
            result = result[~result["steam_appid"].isin(exclude_appids)]

        result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        result = result.head(top_n)
        return result
