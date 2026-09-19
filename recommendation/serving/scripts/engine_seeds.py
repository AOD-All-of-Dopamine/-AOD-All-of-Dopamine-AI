"""유효한 시드 키를 표준출력에 한 줄씩 — 스트레스 하네스가 쓴다(엔진을 올리지 않고 코퍼스만 읽는다).

    python scripts/engine_seeds.py webtoon 12
"""
from __future__ import annotations

import sys

from aod_serving.engine.bootstrap import BASELINE_CORPORA, default_artifacts


def seeds(platform: str, n: int = 12) -> list[str]:
    import pandas as pd

    d = default_artifacts(platform, BASELINE_CORPORA[platform])
    # TMDB 의 밖으로 나가는 키(`movie_603`)는 dataset 에만 있다. 나머지는 corpus_index 로 충분하다.
    for name in (("dataset.parquet", "corpus_index.parquet") if platform == "tmdb"
                 else ("corpus_index.parquet", "dataset.parquet")):
        path = d / name
        if path.exists():
            col = pd.read_parquet(path, columns=["item_id"])["item_id"]
            step = max(1, len(col) // max(1, n))
            return [str(v) for v in col.iloc[::step].head(n)]
    raise SystemExit(f"{platform}: 코퍼스 인덱스를 못 찾았다 ({d})")


if __name__ == "__main__":
    platform = sys.argv[1]
    print("\n".join(seeds(platform, int(sys.argv[2]) if len(sys.argv) > 2 else 12)))
