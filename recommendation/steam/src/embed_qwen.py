# src/embed_qwen.py
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ensure_artifacts_dir, load_config

QUERY_PROMPT = (
    "Instruct: Given a Steam game, retrieve other Steam games "
    "that a player who likes this game would likely find relevant.\n"
    "Query: "
)

EXPERIMENT_ID = "steam_s1_qwen_v2"


def _apply_seq_cap(model, cfg: dict):
    """max_seq_length 를 실제 텍스트 길이에 맞게 제한한다.

    Qwen3-Embedding 의 기본값은 32768 이다. semantic_text 는 p99 가 277토큰, 최대 359토큰
    (전체 코퍼스 표본 기준)이라 그 길이가 필요 없고, CPU 에서는 배치 인코딩 시 거대한
    할당을 유발해 프로세스가 OOM 으로 조용히 죽는다. 512 는 최대 관측치를 충분히 덮는다.
    """
    cap = cfg["runtime"].get("max_seq_length")
    if cap and model.max_seq_length > cap:
        model.max_seq_length = cap
    return model


def resolve_runtime(cfg: dict):
    """(model, device, batch_size) 반환. GPU smoke 실패 시 CPU fallback (고정 정책)."""
    import torch
    from sentence_transformers import SentenceTransformer

    model_name = cfg["embedding"]["model_name"]
    if torch.cuda.is_available():
        device, batch = "cuda", cfg["runtime"]["gpu_batch_size"]
        try:
            model = SentenceTransformer(
                model_name, device=device,
                model_kwargs={"torch_dtype": "float16"},
            )
            _apply_seq_cap(model, cfg)
            model.encode(["smoke test"] * 32, batch_size=batch, show_progress_bar=False)
            return model, device, batch
        except Exception as e:
            print(f"GPU path failed ({type(e).__name__}: {e}); falling back to CPU")
    device, batch = "cpu", cfg["runtime"]["cpu_batch_size"]
    model = SentenceTransformer(
        model_name, device=device,
        model_kwargs={"torch_dtype": "float32"},  # CPU(fp32) 고정 정책 (Qwen3 기본 bf16 방지)
    )
    _apply_seq_cap(model, cfg)
    return model, device, batch


def encode_with_backoff(model, texts, batch: int, **kw) -> np.ndarray:
    while True:
        try:
            return model.encode(
                texts, batch_size=batch, normalize_embeddings=True,
                show_progress_bar=True, **kw,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch > 1:
                batch //= 2
                print(f"OOM: retrying with batch_size={batch}")
                continue
            raise


def _parse_out_dir() -> tuple[Path, Path]:
    """(입력 아티팩트 디렉터리, 출력 디렉터리) — `--out DIR` 로 출력을 분리할 수 있다.

    기존 임베딩을 덮어쓰지 않고 재현성을 검증할 때 쓴다.
    """
    src = ensure_artifacts_dir()
    for i, arg in enumerate(sys.argv):
        if arg == "--out" and i + 1 < len(sys.argv):
            dst = Path(sys.argv[i + 1])
            dst.mkdir(parents=True, exist_ok=True)
            return src, dst
    return src, src


def main():
    cfg = load_config()
    src, out = _parse_out_dir()
    df = pd.read_parquet(src / "dataset.parquet")
    anchors = pd.read_parquet(src / "anchors_40.parquet")
    if out != src:
        print(f"입력 {src} → 출력 {out} (기존 임베딩 보존)")

    model, device, batch = resolve_runtime(cfg)
    print(f"device={device} batch={batch}")

    # --- 500건 benchmark ---
    n_bench = cfg["runtime"]["benchmark_items"]
    sample = df["semantic_text"].tolist()[:n_bench]
    t0 = time.perf_counter()
    encode_with_backoff(model, sample, batch)
    elapsed = time.perf_counter() - t0
    bench = {
        "device": device,
        "batch_size": batch,
        "benchmark_items": len(sample),
        "elapsed_seconds": round(elapsed, 2),
        "items_per_second": round(len(sample) / elapsed, 2),
        "estimated_total_seconds": round(elapsed * len(df) / len(sample), 1),
    }
    with open(out / "runtime_benchmark.json", "w") as f:
        json.dump(bench, f, indent=2)
    print(json.dumps(bench, indent=2))

    # --- corpus embedding ---
    corpus_emb = encode_with_backoff(model, df["semantic_text"].tolist(), batch).astype("float32")
    np.save(out / "corpus_embeddings.npy", corpus_emb)
    corpus_index = df[["steam_appid", "name"]].reset_index(drop=True)
    corpus_index.insert(0, "embedding_row", range(len(corpus_index)))
    corpus_index.to_parquet(out / "corpus_index.parquet", index=False)

    # --- anchor query embedding (custom instruction) ---
    anchor_emb = model.encode(
        anchors["semantic_text"].tolist(),
        prompt=QUERY_PROMPT,
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    np.save(out / "anchor_embeddings.npy", anchor_emb)

    run_config = {
        "experiment_id": EXPERIMENT_ID,
        "model": cfg["embedding"]["model_name"],
        "representation": cfg["representation"]["type"],
        "query_instruction": cfg["embedding"]["query_instruction"],
        "normalize_embeddings": cfg["embedding"]["normalize"],
        "retrieval": "exact_dot_product",
        "candidate_k": cfg["retrieval"]["candidate_k"],
        "device": device,
        "batch_size": batch,
        "corpus_rows": int(corpus_emb.shape[0]),
        "embedding_dim": int(corpus_emb.shape[1]),
    }
    with open(out / "qwen_run_config.json", "w") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(json.dumps(run_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
