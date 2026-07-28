from pathlib import Path

import numpy as np
import pandas as pd

from src.config import artifact_dir


class SeedLoader:
    def __init__(self, artifacts: str | Path | None = None):
        d = artifact_dir(artifacts)
        self.artifacts = d
        self.embeddings = np.load(d / "corpus_embeddings.npy", mmap_mode="r")
        self.index = pd.read_parquet(d / "corpus_index.parquet")
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
