"""아티팩트 없이 도는 테스트용 — 계약을 만족하는 아주 작은 가짜 코퍼스."""
import json
from pathlib import Path
import numpy as np, pandas as pd, pytest

SCHEMAS = Path(__file__).resolve().parents[2] / "schemas"


def write_corpus(d: Path, n: int = 6, dim: int = 1024, *, platform: str = "webnovel") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    e = rng.normal(size=(n, dim)).astype(np.float32); e /= np.linalg.norm(e, axis=1, keepdims=True)
    np.save(d / "corpus_embeddings.npy", e)
    ids = np.arange(100, 100 + n, dtype=np.int64)
    pd.DataFrame({"embedding_row": np.arange(n, dtype=np.int64), "item_id": ids, "name": [f"작품{i}" for i in ids]}
                 ).to_parquet(d / "corpus_index.parquet")
    pd.DataFrame({"item_id": ids, "name": [f"작품{i}" for i in ids], "synopsis": "줄거리", "genres": [["판타지"]] * n,
                  "author": "작가", "publisher": "출판사", "age_limit": 0, "status": "연재", "is_completed": False,
                  "rating": 9.5, "url": "https://example.test", "episode_count": pd.array([30] * n, dtype="Int64"),
                  "interest_count": pd.array([1000] * n, dtype="Int64")}).to_parquet(d / "dataset.parquet")
    return d


@pytest.fixture()
def corpus(tmp_path):
    return write_corpus(tmp_path / "webnovel" / "wn_test")


@pytest.fixture()
def wn_schema():
    return json.loads((SCHEMAS / "webnovel.v1.json").read_text(encoding="utf-8"))
