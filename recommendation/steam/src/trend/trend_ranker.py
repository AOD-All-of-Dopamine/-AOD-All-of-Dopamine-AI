from pathlib import Path

import pandas as pd

from personalization.personalized_ranker import PersonalizedRanker
from trend.trend_features import TREND_DIR


class TrendRanker(PersonalizedRanker):
    def __init__(
        self,
        rec_boost: float = 0.03,
        trend_weight: float = 0.01,
        trend_features_path: str | Path | None = None,
    ):
        super().__init__(rec_boost=rec_boost)
        self.trend_weight = trend_weight

        if trend_features_path is None:
            trend_features_path = TREND_DIR / "trend_features.parquet"
        self.trend_features = pd.read_parquet(trend_features_path)
        self.trend_features = self.trend_features.set_index("steam_appid")

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

        result["trend_signal"] = result["steam_appid"].map(
            lambda x: self.trend_features.loc[x, "trend_signal"]
            if x in self.trend_features.index
            else 0.0
        )

        result["final_score"] = result["seed_similarity"] * (
            1
            + result["recommendations_percentile"] * self.rec_boost
            + result["trend_signal"] * self.trend_weight
        )

        if exclude_appids:
            result = result[~result["steam_appid"].isin(exclude_appids)]

        result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        result = result.head(top_n)
        return result
