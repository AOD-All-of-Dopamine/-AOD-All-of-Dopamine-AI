import numpy as np
import pandas as pd

CORPUS_EMBEDDINGS = "artifacts/s1_v2/corpus_embeddings.npy"
CORPUS_INDEX = "artifacts/s1_v2/corpus_index.parquet"


class SeedLoader:
    def __init__(self):
        self.embeddings = np.load(CORPUS_EMBEDDINGS, mmap_mode="r")
        self.index = pd.read_parquet(CORPUS_INDEX)
        self.appid_to_row = dict(zip(self.index["steam_appid"], self.index["embedding_row"]))

    def load(self, liked_appids: list[int]) -> dict[int, np.ndarray]:
        result = {}
        missing = []
        for appid in liked_appids:
            row = self.appid_to_row.get(appid)
            if row is None:
                missing.append(appid)
                continue
            result[appid] = self.embeddings[row].copy()
        if missing:
            raise ValueError(f"Appids not found in corpus: {missing}")
        return result
