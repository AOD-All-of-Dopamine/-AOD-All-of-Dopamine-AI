from pathlib import Path

import pandas as pd

from src.config import artifact_dir


class PersonalizedRanker:
    def __init__(self, pop_boost: float = 0.03, artifacts: str | Path | None = None):
        self.artifacts = artifact_dir(artifacts)
        self.dataset = pd.read_parquet(self.artifacts / "dataset.parquet")
        self.pop_boost = pop_boost
        self.dataset = self.dataset.set_index("item_id")

    def rank(
        self,
        candidates: pd.DataFrame,
        exclude_ids: set[int] | None = None,
        top_n: int = 300,
    ) -> pd.DataFrame:
        result = candidates.copy()

        rec_col = self.dataset["interest_count"].fillna(0)
        pct = rec_col.rank(pct=True, ascending=True)

        result["interest_percentile"] = result["item_id"].map(
            lambda x: pct.get(x, 0.0)
        )
        result["final_score"] = result["seed_similarity"] * (
            1 + result["interest_percentile"] * self.pop_boost
        )

        if exclude_ids:
            result = result[~result["item_id"].isin(exclude_ids)]

        result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        result = result.head(top_n)
        return result
