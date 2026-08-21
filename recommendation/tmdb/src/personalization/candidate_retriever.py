"""후보 검색 — 허브니스 보정 + 서빙 가능 풀 마스크.

────────────────────────────────────────────────────────────────────────────
**허브니스: Steam 의 λ=0.35 를 그대로 가져오면 안 된다. 방향이 반대다.**

Steam 실측 — 유명작이 허브에서 **밀려 있었다**:
    코퍼스 중앙 0.4544 · Skyrim 0.4256 · Fallout 4 0.4246   (둘 다 중앙 미만)
    → 중심을 빼면 유명작이 **올라온다**. λ=0.35 가 전 축을 개선했다.

TMDB 실측 — 방향이 **반대**다:
    허브도 분포  p10 0.5485 · p50 0.6271 · p90 0.6855
    유명 2,000건 중앙 0.6456   vs   무명 2,000건 0.6151
    → 유명작이 허브에 **더 가깝다**. 중심을 빼면 유명작이 **내려간다**.

원인 추정: Steam 은 semantic_text 의 절반이 통제 태그(15개)라 대작일수록 태그가
분산돼 허브도가 낮다. TMDB 는 줄거리가 88% 라 대작일수록 서술이 전형적이다.

⇒ **기본값 0.0 으로 두고 프로필 세트에서 재측정한다.** 근거 없이 켜지 않는다.
────────────────────────────────────────────────────────────────────────────
"""
from pathlib import Path
import numpy as np, pandas as pd
from src.config import artifact_dir


class CandidateRetriever:
    #: 허브니스 보정 계수. **측정 전까지 0.0.** 위 docstring 참조.
    hub_lambda: float = 0.0

    def __init__(self, artifacts=None, min_overview_len: int = 0):
        d = artifact_dir(artifacts)
        self.artifacts = d
        self.embeddings = np.load(d / "corpus_embeddings.npy", mmap_mode="r")
        idx = pd.read_parquet(d / "corpus_index.parquet").sort_values("embedding_row")
        ds = pd.read_parquet(d / "dataset.parquet").set_index("item_id")
        self.dataset = ds.loc[idx["item_id"].to_numpy()].reset_index()
        self.dataset["row"] = np.arange(len(self.dataset))
        # 서빙 가능 풀 — 한국어 줄거리. 영어 줄거리 26,052건은 TMDB 에 한국어 번역이
        # 없어서(크롤이 ko-KR 우선 · en-US 보완) 크롤로 늘릴 수 없다.
        ko = self.dataset["overview"].fillna("").str.contains(r"[가-힣]").to_numpy()
        self.servable = ko & (self.dataset["overview_len"].to_numpy() >= min_overview_len)

    def corpus_centroid(self) -> np.ndarray:
        c = getattr(self, "_centroid", None)
        if c is None:
            c = np.asarray(self.embeddings, dtype=np.float32).mean(axis=0)
            self._centroid = c
        return c

    def compute_similarity_matrix(self, seed_embeddings: dict, hub_lambda=None) -> np.ndarray:
        """score(s,c) = <s − λμ, c>. λ=0 이면 순수 코사인(임베딩은 L2 정규화돼 있다)."""
        V = np.array(list(seed_embeddings.values()), dtype=np.float32)
        lam = self.hub_lambda if hub_lambda is None else hub_lambda
        if lam: V = V - lam * self.corpus_centroid()[None, :]
        return V @ np.asarray(self.embeddings, dtype=np.float32).T

    def full_corpus_frame(self) -> pd.DataFrame:
        return self.dataset[["row", "item_id", "name"]].copy()
