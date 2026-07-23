import numpy as np
import pandas as pd

from user_profile import UserProfileBuilder
from personalized_retrieve import PersonalizedRetriever


def ndcg_at_k(ranks: list[int], k: int = 10) -> float:
    if not ranks:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 1) for r in ranks if r <= k)
    ideal = min(len(ranks), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


class LeaveOneOutEvaluator:
    def __init__(self, builder: UserProfileBuilder, retriever: PersonalizedRetriever):
        self.builder = builder
        self.retriever = retriever

    def evaluate_user(
        self,
        all_games: list[tuple[int, float]],
        holdout_indices: list[int],
        top_k: list[int] | None = None,
    ) -> dict:
        if top_k is None:
            top_k = [10, 20, 100]
        holdout = [all_games[i] for i in holdout_indices]
        train = [g for i, g in enumerate(all_games) if i not in holdout_indices]

        user_vec = self.builder.build_user_vector(train)
        train_appids = {g[0] for g in train}

        results = self.retriever.retrieve(user_vec, exclude_appids=train_appids, top_n=max(top_k))

        found_ranks = []
        for appid, _ in holdout:
            match = results[results["steam_appid"] == appid]
            if not match.empty:
                found_ranks.append(int(match["rank"].iloc[0]))

        out = {"found_ranks": found_ranks}
        for k in top_k:
            recall = sum(1 for r in found_ranks if r <= k) / len(holdout)
            ndcg = ndcg_at_k(found_ranks, k)
            out[f"Recall@{k}"] = recall
            out[f"NDCG@{k}"] = ndcg
        return out

    def evaluate_users(
        self, user_profiles: list[dict], holdout_per_user: int = 1, top_k: list[int] | None = None
    ) -> pd.DataFrame:
        rows = []
        for profile in user_profiles:
            all_games = profile["games"]
            n = len(all_games)
            if n < holdout_per_user + 1:
                continue
            for _ in range(min(3, n // 2)):
                holdout_idx = np.random.choice(n, holdout_per_user, replace=False).tolist()
                result = self.evaluate_user(all_games, holdout_idx, top_k)
                result["user_id"] = profile.get("user_id", "unknown")
                result["n_games"] = n
                rows.append(result)
        return pd.DataFrame(rows)
