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

    def compute_similarity_matrix(
        self, seed_embeddings: dict[int, np.ndarray]
    ) -> np.ndarray:
        seed_vecs = np.array(list(seed_embeddings.values()), dtype=np.float32)
        return seed_vecs @ self.embeddings.T

    def full_corpus_frame(self) -> pd.DataFrame:
        return self.index[["steam_appid", "name"]].copy()
