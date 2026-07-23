# src/embed_qwen.py
import json
import time

import numpy as np
import pandas as pd

from src.config import ensure_artifacts_dir, load_config

QUERY_PROMPT = (
    "Instruct: Given a Steam game, retrieve other Steam games "
    "that a player who likes this game would likely find relevant.\n"
    "Query: "
)

EXPERIMENT_ID = "steam_s1_qwen_v2"


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
            model.encode(["smoke test"] * 32, batch_size=batch, show_progress_bar=False)
            return model, device, batch
        except Exception as e:
            print(f"GPU path failed ({type(e).__name__}: {e}); falling back to CPU")
    device, batch = "cpu", cfg["runtime"]["cpu_batch_size"]
    model = SentenceTransformer(
        model_name, device=device,
        model_kwargs={"torch_dtype": "float32"},  # CPU(fp32) 고정 정책 (Qwen3 기본 bf16 방지)
    )
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


def main():
    cfg = load_config()
    out = ensure_artifacts_dir()
    df = pd.read_parquet(out / "dataset.parquet")
    anchors = pd.read_parquet(out / "anchors_40.parquet")

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
