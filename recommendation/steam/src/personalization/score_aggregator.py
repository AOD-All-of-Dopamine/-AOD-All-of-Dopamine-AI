import numpy as np
import pandas as pd


class ScoreAggregator:
    """시드별 유사도 행렬(시드 × 후보)을 후보 하나당 점수 하나로 접는다.

    ────────────────────────────────────────────────────────────────────────
    2026-08-15 — **"공통 축을 강제하는" 집계는 기각됐다. 다시 하지 말 것.**

    동기: `two_rhythm_arcade`(NecroDancer + Just Shapes & Beats)의 Top-20 부적합
    10개가 **전부** 1점(GENRE_ONLY)의 던전 크롤러·벨트스크롤이었다 —
    Crawl · Charlie Murder · Legend of Dungeon · Castle Crashers · Gauntlet · Rotwood.
    NecroDancer 의 태그 15개 중 리듬 계열은 3개뿐이고 나머지는 Roguelike ·
    Dungeon Crawler · RPG · Roguelite · Procedural Generation 이다. 시드가 2개면
    `top2_mean` 은 그냥 평균이라 한쪽만 닮아도 다른 쪽의 낮은 점수를 덮는다.

    그래서 "1등 시드와 2등 시드의 격차"를 벌하는 집계를 넣어 봤다:

        score = top2_mean − α · (top1 − top2)      (n=2 에서 α=0.5 는 정확히 min)

    측정 (two_rhythm_arcade, Top-20 전수 판정):
        α=0     (=top2_mean)   0.50
        α=0.25                 0.30
        α=0.5   (=min)         0.15   ← 0점(무관)까지 등장

    **역효과다.** 밀려 올라온 것은 리듬 게임이 아니라 迷宮校舎(호러) · PAYDAY ·
    Watcher Chronicles(소울라이크) · Gamble With Your Friends 였다.

    이유: **두 시드 벡터의 교집합은 두 장르의 교집합이 아니다.** NecroDancer 와
    JS&B 가 임베딩에서 실제로 공유하는 방향은 "리듬"이 아니라 "로컬 협동 2D 액션"
    이다. 평균 풀링 밀집 벡터는 장르 축으로 분해되지 않으므로, min 을 취하면
    두 게임의 **부수적** 공통점을 정확히 집어낸다. 원하는 것의 반대다.
    ────────────────────────────────────────────────────────────────────────
    """

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
