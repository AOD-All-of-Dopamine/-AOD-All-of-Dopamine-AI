import numpy as np
import pandas as pd


class ScoreAggregator:
    def aggregate_all(
        self,
        sim_matrix: np.ndarray,
        seed_embeddings: dict[int, np.ndarray],
        corpus_df: pd.DataFrame,
        strategies: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        if strategies is None:
            strategies = ["max", "mean", "top2_mean"]

        seed_ids = list(seed_embeddings.keys())
        n_seeds = sim_matrix.shape[0]
        result = {}

        if "max" in strategies:
            max_sim = sim_matrix.max(axis=0)
            dominant_idx = sim_matrix.argmax(axis=0)
            result["max"] = self._build_frame(
                corpus_df, max_sim, "MAX",
                dominant_seed=[seed_ids[i] for i in dominant_idx],
            )

        if "mean" in strategies:
            mean_sim = sim_matrix.mean(axis=0)
            result["mean"] = self._build_frame(corpus_df, mean_sim, "MEAN")

        if "top2_mean" in strategies:
            if n_seeds <= 2:
                top2_mean_sim = sim_matrix.mean(axis=0)
            else:
                sorted_sims = np.sort(sim_matrix, axis=0)
                top2_mean_sim = sorted_sims[-2:, :].mean(axis=0)
            result["top2_mean"] = self._build_frame(corpus_df, top2_mean_sim, "TOP2_MEAN")

        return result

    def _build_frame(
        self,
        corpus_df: pd.DataFrame,
        scores: np.ndarray,
        strategy: str,
        dominant_seed: list[int] | None = None,
    ) -> pd.DataFrame:
        df = corpus_df.copy()
        df["seed_similarity"] = scores
        df["dominant_seed"] = dominant_seed if dominant_seed is not None else None
        df["strategy"] = strategy
        return df
