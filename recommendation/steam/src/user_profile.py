import numpy as np
import pandas as pd

CORPUS_EMBEDDINGS = "artifacts/s1_v2/corpus_embeddings.npy"
CORPUS_INDEX = "artifacts/s1_v2/corpus_index.parquet"
DATASET = "artifacts/s1_v2/dataset.parquet"


class UserProfileBuilder:
    def __init__(self):
        self.embeddings = np.load(CORPUS_EMBEDDINGS, mmap_mode="r")
        self.index = pd.read_parquet(CORPUS_INDEX)
        self.appid_to_row = dict(zip(self.index["steam_appid"], self.index["embedding_row"]))
        self.dataset = pd.read_parquet(DATASET)

    def get_embedding(self, steam_appid: int) -> np.ndarray | None:
        row = self.appid_to_row.get(steam_appid)
        if row is None:
            return None
        return self.embeddings[row].copy()

    def build_user_vector(
        self, game_playtimes: list[tuple[int, float]]
    ) -> np.ndarray:
        valid = [(aid, wt) for aid, wt in game_playtimes if aid in self.appid_to_row]
        if not valid:
            raise ValueError("No valid games found in corpus")

        weighted_sum = np.zeros(self.embeddings.shape[1], dtype=np.float64)
        total_weight = 0.0
        for aid, hours in valid:
            weight = np.log1p(max(hours, 0.1))
            emb = self.embeddings[self.appid_to_row[aid]]
            weighted_sum += emb.astype(np.float64) * weight
            total_weight += weight

        user_vec = weighted_sum / total_weight
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec = user_vec / norm
        return user_vec.astype(np.float32)

    @staticmethod
    def cosine_similarity(user_vec: np.ndarray, corpus: np.ndarray) -> np.ndarray:
        return np.dot(corpus, user_vec)
