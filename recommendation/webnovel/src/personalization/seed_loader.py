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
        self.id_to_row = dict(zip(self.index["item_id"], self.index["embedding_row"]))

    def load(self, liked_ids: list[int]) -> dict[int, np.ndarray]:
        result = {}
        missing = []
        for item in liked_ids:
            row = self.id_to_row.get(item)
            if row is None:
                missing.append(item)
                continue
            result[item] = self.embeddings[row].copy()
        if missing:
            raise ValueError(f"코퍼스에 없는 item_id: {missing}")
        return result
