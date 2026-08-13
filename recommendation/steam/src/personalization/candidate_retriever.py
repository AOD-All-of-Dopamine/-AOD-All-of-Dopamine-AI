from pathlib import Path

import numpy as np
import pandas as pd

from src.config import artifact_dir


class CandidateRetriever:
    def __init__(self, artifacts: str | Path | None = None):
        d = artifact_dir(artifacts)
        self.artifacts = d
        self.embeddings = np.load(d / "corpus_embeddings.npy", mmap_mode="r")
        self.index = pd.read_parquet(d / "corpus_index.parquet")
        self.dataset = pd.read_parquet(d / "dataset.parquet")

    #: 허브니스 보정 계수. 0 이면 끔. 원리는 compute_similarity_matrix docstring.
    #
    # 35프로필 · 미판정 0 · 페어드 부트스트랩 (λ=0 대비, 괄호는 0.8 미만 프로필 수):
    #
    #   λ      k=10           k=20           k=30           k=50           k=75
    #   0      0.9314 (0)     0.9286 (0)     0.9219 (0)     0.9223 (1)     0.9013 (1)
    #   0.15   0.9629 (0)     0.9386 (0)     0.9343 (0)     0.9263 (1)     0.9131 (1)
    #   0.25   0.9686 (0)     0.9500 (0)     0.9410 (0)     0.9309 (1)     0.9177 (2)
    #   0.35   0.9657 (0)     0.9529 (0)     0.9438 (0)     0.9343 (2)     0.9192 (0)
    #   0.5    0.9629 (0)     0.9514 (0)     0.9419 (0)     0.9337 (1)     0.9204 (3)
    #
    #   k=75 Δ:  λ0.15 +0.012 [+0.005,+0.019]   λ0.25 +0.016 [+0.006,+0.026]
    #            λ0.35 +0.018 [+0.007,+0.029]   λ0.5  +0.019 [+0.006,+0.032]   전부 유의
    #
    # **모든 축이 오른다** (λ=0.35, k=75): 저리뷰 +0.027 · 니치 +0.027 · 롱테일 +0.023 ·
    # 혼합 +0.015 · 대작 +0.009. 지금까지 채택한 레버 중 유일하게 전 축 동시 개선이다.
    #
    # 왜 0.35 인가: 0.25~0.5 구간은 서로 통계적으로 구분되지 않는다(Δ 차이 0.003 이내).
    # 평균은 0.35~0.5 에서 평평하고, 미달 개수의 깊이별 출렁임(0.35 가 k=50 에서 2,
    # k=75 에서 0)은 35프로필 × 0.78~0.80 경계의 잡음이다. 그래서 평탄 구간의 중앙을
    # 잡았다. 이 값을 옮기려면 곡선을 다시 재야 한다.
    #
    # λ 를 더 올리면 안 되는 이유(실측): λ=0.5 는 **반허브**를 과하게 올린다. coh_fps 에
    # 축구 게임 3개(Head Goal / Ball 2D Soccer / Strikers Club)가 0점으로 진입했다.
    # "Football·Sports·eSports·Competitive"는 매우 특수한 벡터라 허브도가 낮다.
    hub_lambda: float = 0.35

    def corpus_centroid(self) -> np.ndarray:
        """코퍼스 임베딩의 평균. 한 번만 계산해서 재사용한다."""
        c = getattr(self, "_centroid", None)
        if c is None:
            c = np.asarray(self.embeddings, dtype=np.float32).mean(axis=0)
            self._centroid = c
        return c

    def compute_similarity_matrix(
        self, seed_embeddings: dict[int, np.ndarray], hub_lambda: float | None = None
    ) -> np.ndarray:
        """`hub_lambda > 0` 이면 허브니스를 뺀다.

        **문제.** 고차원 임베딩에서 일부 벡터가 "모든 것과 비슷"해진다(허브). 실측으로
        코퍼스 4,000개 표본에 대한 평균 유사도를 재보니:

            코퍼스 분포        중앙 0.4544   p90 0.4949   p99 0.5241
            奇怪的RPG          0.5162   ← p99 근처. 리뷰 4,065
            Skyrim             0.4256   ← 중앙 미만. 리뷰 215,294
            Fallout 4          0.4246   ← 중앙 미만. 리뷰 288,856

        `coh_arpg`(DS3/Witcher 3/Skyrim)에서 명작들이 194~1,566위로 밀리고 그 자리를
        무명 ARPG 가 채우던 원인이 이것이다. 유사도 격차(0.02~0.07)가 실재하므로 곱셈형
        인기도 부스트로는 못 뒤집었다(0.60·1.50 모두 실패).

        **보정.** 허브도 h(c) = 코퍼스 평균 μ 와의 내적이다. 평균은 선형이라
        mean_i <c, a_i> = <c, μ> 로 정확히 같다 — 표본이 필요 없다. 따라서

            score(s, c) = <s, c> - λ·<μ, c> = <s - λμ, c>

        즉 **쿼리에서 코퍼스 중심을 빼는 것**과 동치다(임베딩 이방성의 표준 교정).
        코퍼스를 다시 임베딩할 필요가 없다.
        """
        seed_vecs = np.array(list(seed_embeddings.values()), dtype=np.float32)
        lam = self.hub_lambda if hub_lambda is None else hub_lambda
        if lam:
            seed_vecs = seed_vecs - lam * self.corpus_centroid()[None, :]
        return seed_vecs @ self.embeddings.T

    def full_corpus_frame(self) -> pd.DataFrame:
        return self.index[["steam_appid", "name"]].copy()
