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


def encode_with_backoff(model, texts, batch: int, show_progress_bar: bool = True, **kw) -> np.ndarray:
    while True:
        try:
            return model.encode(
                texts, batch_size=batch, normalize_embeddings=True,
                show_progress_bar=show_progress_bar, **kw,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch > 1:
                batch //= 2
                print(f"OOM: retrying with batch_size={batch}")
                continue
            raise


def _dir_arg(flag: str, default: Path) -> Path:
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    return default


def _parse_out_dir() -> tuple[Path, Path]:
    """(입력 디렉터리, 출력 디렉터리).

    `--in DIR` / `--out DIR` 로 표현별 아티팩트를 분리한다. 표현을 바꿔 실험할 때
    기존 임베딩을 덮어쓰면 두 표현을 비교할 수 없으므로 반드시 분리해야 한다.
    """
    src = _dir_arg("--in", ensure_artifacts_dir())
    dst = _dir_arg("--out", src)
    dst.mkdir(parents=True, exist_ok=True)
    return src, dst


PROGRESS_FILE = "_embed_progress.json"


def _load_progress(out: Path) -> dict:
    p = out / PROGRESS_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def _save_progress(out: Path, **kw) -> None:
    (out / PROGRESS_FILE).write_text(json.dumps(kw, ensure_ascii=False))


def encode_to_memmap(
    model, texts: list[str], batch: int, out: Path, dim: int,
    n_existing: int = 0, chunk: int = 2000,
) -> np.ndarray:
    """청크 단위로 인코딩해 디스크에 바로 쓴다.

    17만 건을 한 번에 encode 하면 두 가지가 문제다:
      1) 결과 배열 0.7GB 를 모델(2.4GB)과 함께 들고 있어야 해서 cgroup 4GiB 를 위협한다
      2) 35시간 중 34시간째에 죽으면 전부 날아간다

    memmap 에 청크마다 flush 하고 진행 상황을 기록해, 재실행하면 남은 것부터 이어서 한다.
    """
    total = n_existing + len(texts)
    emb_path = out / "corpus_embeddings.npy"
    done = _load_progress(out).get("rows_done", n_existing)

    if done > n_existing:
        print(f"이어받기: {done - n_existing:,}/{len(texts):,} 건 이미 완료")

    mm = np.lib.format.open_memmap(
        emb_path, mode="r+" if emb_path.exists() else "w+",
        dtype="float32", shape=(total, dim),
    )
    t0 = time.time()
    for start in range(done - n_existing, len(texts), chunk):
        part = texts[start : start + chunk]
        vecs = encode_with_backoff(model, part, batch, show_progress_bar=False)
        mm[n_existing + start : n_existing + start + len(part)] = vecs.astype("float32")
        mm.flush()
        _save_progress(out, rows_done=n_existing + start + len(part), total=total)
        n_done = start + len(part)
        el = time.time() - t0
        rate = n_done / el if el else 0
        eta = (len(texts) - n_done) / rate / 3600 if rate else 0
        print(f"  {n_done:,}/{len(texts):,}  {rate:.2f} items/s  남은 {eta:.1f}h", flush=True)
    return mm


def select_targets(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """이번에 임베딩할 행과, 이미 만들어둔 인덱스를 고른다.

    17만 건을 한 번에 임베딩하면 24시간이 넘는다. 그런데 우리가 실제로 추천하는 것은
    리뷰 수가 알려진 게임뿐이다(`has_recommendations`). 그래서 **가치 있는 것부터
    임베딩하고 나머지는 나중에 이어붙이는** 두 단계로 나눈다.

        1단계: --only-with-reviews   리뷰 있는 것만 (약 6시간)
        2단계: --append             나머지를 이어붙임 (필요해지면)

    `--append` 는 기존 corpus_index 에 있는 appid 를 건너뛴다. 부분 커버리지는
    문제가 되지 않는다 — 후보 생성이 dataset 이 아니라 corpus_index 를 기준으로 돌기
    때문에, 임베딩된 것만 추천 후보가 된다.
    """
    existing_idx = None
    idx_path, emb_path = out / "corpus_index.parquet", out / "corpus_embeddings.npy"

    if "--append" in sys.argv:
        if not (idx_path.exists() and emb_path.exists()):
            raise SystemExit(f"--append 인데 {out} 에 기존 임베딩이 없습니다.")
        existing_idx = pd.read_parquet(idx_path)
        done = set(existing_idx["steam_appid"])
        df = df[~df["steam_appid"].isin(done)]
        print(f"--append: 이미 {len(done):,}건 임베딩됨 → 이번에 {len(df):,}건 추가")
    elif emb_path.exists() and "--force" not in sys.argv:
        prog = _load_progress(out)
        if prog and prog.get("rows_done", 0) < prog.get("total", 0):
            print(f"중단된 작업 발견: {prog['rows_done']:,}/{prog['total']:,} — 이어서 진행합니다")
        else:
            raise SystemExit(
                f"{out} 에 이미 완료된 corpus_embeddings.npy 가 있습니다.\n"
                f"  이어붙이려면 --append, 처음부터 다시 만들려면 --force 를 쓰세요."
            )

    if "--only-with-reviews" in sys.argv:
        before = len(df)
        if "has_recommendations" not in df.columns:
            raise SystemExit("dataset 에 has_recommendations 컬럼이 없습니다.")
        df = df[df["has_recommendations"].fillna(False).astype(bool)]
        print(f"--only-with-reviews: {before:,} → {len(df):,}건 "
              f"({len(df) / before * 100:.0f}%, 리뷰 수가 알려진 게임만)")

    return df.reset_index(drop=True), existing_idx


def main():
    cfg = load_config()
    src, out = _parse_out_dir()
    df = pd.read_parquet(src / "dataset.parquet")
    anchors = pd.read_parquet(src / "anchors_40.parquet")
    if out != src:
        print(f"입력 {src} → 출력 {out} (기존 임베딩 보존)")

    df, existing_idx = select_targets(df, out)
    if df.empty:
        print("임베딩할 대상이 없습니다.")
        return

    model, device, batch = resolve_runtime(cfg)
    print(f"device={device} batch={batch} | 대상 {len(df):,}건")

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

    # --- corpus embedding (청크 단위로 디스크에 직접 기록, 중단 시 이어받기) ---
    n_existing = len(existing_idx) if existing_idx is not None else 0
    dim = int(model.get_sentence_embedding_dimension())
    corpus_emb = encode_to_memmap(
        model, df["semantic_text"].tolist(), batch, out, dim, n_existing=n_existing
    )

    corpus_index = df[["steam_appid", "name"]].reset_index(drop=True)
    corpus_index.insert(0, "embedding_row", range(n_existing, n_existing + len(corpus_index)))
    if existing_idx is not None:
        corpus_index = pd.concat([existing_idx, corpus_index], ignore_index=True)
        print(f"이어붙임: {n_existing:,} + {len(df):,} = {len(corpus_index):,}건")
    corpus_index.to_parquet(out / "corpus_index.parquet", index=False)
    (out / PROGRESS_FILE).unlink(missing_ok=True)  # 완료 표시

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
