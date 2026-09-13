"""시드 로더 — 행 인덱스로 임베딩을 가져온다.

**이름이 아니라 행 인덱스를 키로 쓴다(D-20).** TMDB 코퍼스에는 동명 작품이
2,296개 이름 · 5,269행(8.8%) 있다. 크로스도메인 실험에서 `하모니`(교도소 드라마 /
외계 행성 SF)가 한쪽 벡터로 검색되고 다른 쪽 본문으로 채점되는 사고가 났다.
"""
from pathlib import Path
import numpy as np, pandas as pd
from src.config import artifact_dir


class SeedLoader:
    def __init__(self, artifacts=None):
        d = artifact_dir(artifacts)
        self.artifacts = d
        self.embeddings = np.load(d / "corpus_embeddings.npy", mmap_mode="r")

    def load(self, seed_rows: list[int]) -> dict[int, np.ndarray]:
        n = len(self.embeddings)
        bad = [r for r in seed_rows if not (0 <= int(r) < n)]
        if bad:
            raise ValueError(f"코퍼스 범위 밖의 행: {bad}")
        return {int(r): self.embeddings[int(r)].copy() for r in seed_rows}
