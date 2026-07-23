import numpy as np
import pandas as pd

CORPUS_EMBEDDINGS = "artifacts/s1_v2/corpus_embeddings.npy"
CORPUS_INDEX = "artifacts/s1_v2/corpus_index.parquet"
DATASET = "artifacts/s1_v2/dataset.parquet"


class CandidateRetriever:
    def __init__(self):
        self.embeddings = np.load(CORPUS_EMBEDDINGS, mmap_mode="r")
        self.index = pd.read_parquet(CORPUS_INDEX)
        self.dataset = pd.read_parquet(DATASET)

    def compute_similarity_matrix(
        self, seed_embeddings: dict[int, np.ndarray]
    ) -> np.ndarray:
        seed_vecs = np.array(list(seed_embeddings.values()), dtype=np.float32)
        return seed_vecs @ self.embeddings.T

    def full_corpus_frame(self) -> pd.DataFrame:
        return self.index[["steam_appid", "name"]].copy()
