# src/verify_embedding.py
"""저장된 임베딩이 지금 이 환경에서 재현되는지 표본으로 검증한다.

전체 코퍼스 재임베딩은 이 하드웨어(CPU, cgroup 4GiB)에서 ~4.7시간이 걸리는데,
`semantic_text` 와 모델이 그대로면 결과도 그대로다. 그래서 "재현되는가"라는 질문에는
표본 검증으로 답하는 편이 낫다 — 몇 분이면 끝나고 결론은 같다.

    python -m src.verify_embedding [--n 500] [--seed 42]

코사인 유사도가 1.0 에 붙으면 파이프라인이 이 환경에서 재현된다는 뜻이다.
"""
import sys
import time

import numpy as np
import pandas as pd

from src.config import ensure_artifacts_dir, load_config
from src.embed_qwen import resolve_runtime


def _arg(flag: str, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return type(default)(sys.argv[i + 1])
    return default


def main():
    n = _arg("--n", 500)
    seed = _arg("--seed", 42)

    cfg = load_config()
    out = ensure_artifacts_dir()
    df = pd.read_parquet(out / "dataset.parquet")
    index = pd.read_parquet(out / "corpus_index.parquet")
    stored = np.load(out / "corpus_embeddings.npy", mmap_mode="r")

    row_of = dict(zip(index["steam_appid"], index["embedding_row"]))
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(df), size=min(n, len(df)), replace=False)
    sample = df.iloc[picks]

    model, device, batch = resolve_runtime(cfg)
    print(f"device={device} batch={batch} max_seq={model.max_seq_length} | 표본 {len(sample)}건")

    t0 = time.perf_counter()
    fresh = model.encode(
        sample["semantic_text"].tolist(),
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    elapsed = time.perf_counter() - t0

    old = np.stack([stored[row_of[a]] for a in sample["steam_appid"]]).astype("float32")
    cos = np.einsum("ij,ij->i", fresh, old)
    max_abs = float(np.abs(fresh - old).max())

    print(f"\n소요 {elapsed:.1f}s ({len(sample) / elapsed:.2f} items/s) "
          f"→ 전체 {len(df)}건 추정 {elapsed * len(df) / len(sample) / 3600:.1f}시간")
    print(f"\n저장본 대비 코사인 유사도")
    print(f"  min    {cos.min():.6f}")
    print(f"  p1     {np.percentile(cos, 1):.6f}")
    print(f"  median {np.median(cos):.6f}")
    print(f"  mean   {cos.mean():.6f}")
    print(f"  요소별 최대 절대차 {max_abs:.2e}")

    worst = sample.iloc[int(np.argmin(cos))]
    print(f"\n가장 낮은 항목: {worst['name'][:50]} (cos={cos.min():.6f})")

    print()
    if cos.min() > 0.9999:
        print("✅ 재현됨 — 저장된 임베딩과 사실상 동일하다. 전체 재임베딩은 불필요하다.")
    elif cos.min() > 0.99:
        print("△ 거의 일치 — 부동소수 수준의 차이. 순위에 영향을 줄 가능성은 낮지만,")
        print("  결과를 바꿀 변경(모델/텍스트/dtype)이 있었는지 확인할 것.")
    else:
        print("❌ 불일치 — 저장된 임베딩과 다른 벡터가 나온다.")
        print("  모델 버전, semantic_text, dtype, max_seq_length 중 무엇이 달라졌는지 확인할 것.")


if __name__ == "__main__":
    main()
