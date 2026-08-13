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

        seed_appids = list(seed_embeddings.keys())
        n_seeds = sim_matrix.shape[0]
        result = {}

        # `dominant_seed` 는 **점수와 무관하게** "이 후보가 어느 시드 때문에 올라왔나"다.
        # 후처리의 시드 인터리빙이 이걸로 묶기 때문에, 집계 전략과 상관없이 항상 채워야 한다.
        # 예전에는 max 에만 있었고 mean/top2_mean 은 None 이었다 — 그래서 전략을 바꾸면
        # 인터리빙이 조용히 죽어 약한 시드가 굶었다(test_refresh 가 이걸 잡았다).
        dominant = [seed_appids[i] for i in sim_matrix.argmax(axis=0)]

        if "max" in strategies:
            max_sim = sim_matrix.max(axis=0)
            result["max"] = self._build_frame(corpus_df, max_sim, "MAX", dominant_seed=dominant)

        if "mean" in strategies:
            mean_sim = sim_matrix.mean(axis=0)
            result["mean"] = self._build_frame(corpus_df, mean_sim, "MEAN", dominant_seed=dominant)

        if "top2_mean" in strategies:
            if n_seeds <= 2:
                top2_mean_sim = sim_matrix.mean(axis=0)
            else:
                sorted_sims = np.sort(sim_matrix, axis=0)
                top2_mean_sim = sorted_sims[-2:, :].mean(axis=0)
            result["top2_mean"] = self._build_frame(
                corpus_df, top2_mean_sim, "TOP2_MEAN", dominant_seed=dominant)

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
