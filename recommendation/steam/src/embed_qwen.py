# src/embed_qwen.py
import gc
import json
import os
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


def _arg_value(flag: str, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


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


SHARD_FILE = "_embeddings.raw"
TEXTS_FILE = "_texts.parquet"
INDEX_FILE = "_index.parquet"

# cgroup 4GiB 에서 모델만 2.98GB 다. 실측: 200건 인코딩은 3.11GB 로 통과하지만
# 2000건은 OOM. 순간 할당을 줄이려면 청크를 작게 가져가는 수밖에 없다.
CHUNK = 256


def _shard_rows(path: Path, dim: int) -> int:
    """원시 파일 크기가 곧 진행 상황이다 — 별도 진행 파일이 필요 없다."""
    return path.stat().st_size // (dim * 4) if path.exists() else 0


def stream_texts(texts_path: Path, chunk: int, skip: int = 0):
    """semantic_text 를 parquet 에서 청크 단위로 흘려보낸다.

    17만 건을 `df[...].tolist()` 로 올리면 0.5GB 인데, 모델(2.4GB)과 합치면 cgroup 4GiB
    한도를 넘어 첫 청크도 못 돌고 OOM 이 난다(실측). 텍스트는 메모리에 두지 않는다.
    """
    import pyarrow.parquet as pq

    seen, buf = 0, []
    for rb in pq.ParquetFile(texts_path).iter_batches(batch_size=chunk, columns=["semantic_text"]):
        col = rb.column(0).to_pylist()
        for t in col:
            seen += 1
            if seen <= skip:
                continue
            buf.append(t)
            if len(buf) >= chunk:
                yield buf
                buf = []
    if buf:
        yield buf


def drop_model_page_cache(model) -> float:
    """모델 파일이 차지한 페이지 캐시를 커널에 돌려준다.

    cgroup v2 는 페이지 캐시도 한도에 넣는다. 실측: RSS 2.92GB 인데 cgroup 은 4.25GB
    (한도 4.29GB, 99%). 차이는 safetensors 2.4GB 를 읽으면서 생긴 캐시다. 가중치는 이미
    프로세스 메모리에 올라와 있으므로 이 캐시는 필요 없다.

    posix_fadvise(DONTNEED) 로 반환하면 한도에 여유가 생긴다. 반환량(GB)을 돌려준다.
    """
    import glob
    import os as _os

    before = _cgroup_gb()
    try:
        cache_dir = _os.path.expanduser("~/.cache/huggingface")
        for pat in ("**/*.safetensors", "**/*.bin"):
            for f in glob.glob(_os.path.join(cache_dir, pat), recursive=True):
                try:
                    fd = _os.open(f, _os.O_RDONLY)
                    _os.posix_fadvise(fd, 0, 0, _os.POSIX_FADV_DONTNEED)
                    _os.close(fd)
                except OSError:
                    pass
    except Exception:
        return 0.0
    return before - _cgroup_gb()


def _cgroup_gb() -> float:
    """cgroup 이 실제로 세는 값. RSS 와 다르다 — 페이지 캐시·커널 메모리가 포함된다.

    실측: RSS 2.81GB 인데 OOM 이 났다. 모델 safetensors(2.4GB)를 읽은 페이지 캐시가
    한도를 함께 먹기 때문이다. 그래서 RSS 만 보면 원인을 놓친다.
    """
    try:
        return int(open("/sys/fs/cgroup/memory.current").read()) / 1e9
    except Exception:
        return 0.0


def encode_to_shard(
    model, texts_path: Path, n_total: int, batch: int, out: Path, dim: int,
    chunk: int = CHUNK, max_items: int = 0,
) -> Path:
    """청크마다 원시 float32 바이트를 파일에 append 한다.

    memmap 을 쓰면 안 되는 이유: 더티 페이지가 디스크로 내려가기 전까지 cgroup 메모리로
    잡힌다. append 방식은 청크 하나(2,000 x 1024 x 4 = 8MB)만 들고 있으면 된다.
    파일 크기가 곧 진행 상황이라 재실행하면 자동으로 이어간다.
    """
    shard = out / SHARD_FILE
    done = _shard_rows(shard, dim)
    if done:
        print(f"이어받기: {done:,}/{n_total:,} 건 완료됨", flush=True)
    if done >= n_total:
        return shard

    t0, t_done = time.time(), 0
    with open(shard, "ab") as f:
        for part in stream_texts(texts_path, chunk, skip=done):
            vecs = encode_with_backoff(model, part, batch, show_progress_bar=False)
            f.write(np.ascontiguousarray(vecs, dtype="float32").tobytes())
            f.flush()
            os.fsync(f.fileno())
            t_done += len(part)
            el = time.time() - t0
            rate = t_done / el if el else 0
            remain = (n_total - done - t_done) / rate / 3600 if rate else 0
            rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1024 / 1024
            print(f"  {done + t_done:,}/{n_total:,}  {rate:.2f} items/s  "
                  f"남은 {remain:.1f}h  RSS {rss:.2f}GB cg {_cgroup_gb():.2f}GB", flush=True)
            if max_items and t_done >= max_items:
                print(f"  --max-items {max_items} 도달 — 프로세스를 종료한다(재실행하면 이어감)", flush=True)
                break
    return shard


def shard_to_npy(shard: Path, dest: Path, dim: int, prefix: np.ndarray | None = None,
                 chunk: int = 20000) -> int:
    """원시 파일을 .npy 로 변환한다. 모델을 내린 뒤 호출해야 메모리가 넉넉하다."""
    n_new = _shard_rows(shard, dim)
    n_pre = len(prefix) if prefix is not None else 0
    mm = np.lib.format.open_memmap(dest, mode="w+", dtype="float32", shape=(n_pre + n_new, dim))
    if prefix is not None:
        mm[:n_pre] = prefix
    with open(shard, "rb") as f:
        for start in range(0, n_new, chunk):
            k = min(chunk, n_new - start)
            buf = np.frombuffer(f.read(k * dim * 4), dtype="float32").reshape(k, dim)
            mm[n_pre + start : n_pre + start + k] = buf
            mm.flush()
    del mm
    return n_pre + n_new


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
    df = pd.read_parquet(
        src / "dataset.parquet",
        columns=["steam_appid", "name", "semantic_text", "has_recommendations"],
    )
    anchors = pd.read_parquet(src / "anchors_40.parquet")
    if out != src:
        print(f"입력 {src} → 출력 {out} (기존 임베딩 보존)")

    df, existing_idx = select_targets(df, out)
    if df.empty:
        print("임베딩할 대상이 없습니다.")
        return

    # 텍스트는 메모리에 올리지 않는다 — 임시 parquet 으로 내보내고 청크로 흘려 읽는다.
    # 17만 건 리스트(0.5GB) + 모델(2.4GB) 이면 cgroup 4GiB 를 넘겨 첫 청크도 못 돈다.
    texts_path = out / TEXTS_FILE
    df[["semantic_text"]].to_parquet(texts_path, index=False)
    n_total = len(df)
    # 인덱스도 디스크로 내린다 — 17만 행을 들고 있으면 모델 로드 후 여유가 1GB 밖에 안 남는다
    df[["steam_appid", "name"]].reset_index(drop=True).to_parquet(out / INDEX_FILE, index=False)
    anchor_texts = anchors["semantic_text"].tolist()
    del df, anchors
    gc.collect()

    model, device, batch = resolve_runtime(cfg)
    freed = drop_model_page_cache(model)
    print(f"device={device} batch={batch} | 대상 {n_total:,}건 | "
          f"cgroup {_cgroup_gb():.2f}GB (페이지 캐시 {freed:.2f}GB 반환)", flush=True)

    dim = int(model.get_sentence_embedding_dimension())

    # --- corpus embedding: 청크마다 원시 파일에 append (중단 시 이어받기) ---
    max_items = int(_arg_value("--max-items", 0))
    shard = encode_to_shard(model, texts_path, n_total, batch, out, dim, max_items=max_items)
    if max_items and _shard_rows(shard, dim) < n_total:
        print(f"부분 완료: {_shard_rows(shard, dim):,}/{n_total:,} — 같은 명령으로 재실행하세요")
        return

    # --- anchor query embedding (검색용 지시문 프리픽스) ---
    anchor_emb = model.encode(
        anchor_texts, prompt=QUERY_PROMPT, batch_size=batch,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype("float32")
    np.save(out / "anchor_embeddings.npy", anchor_emb)

    # 모델을 내려야 .npy 변환에 쓸 메모리가 생긴다
    del model
    gc.collect()

    n_existing = len(existing_idx) if existing_idx is not None else 0
    prefix = np.load(out / "corpus_embeddings.npy")[:n_existing] if n_existing else None
    total = shard_to_npy(shard, out / "corpus_embeddings.npy", dim, prefix=prefix)
    corpus_index = pd.read_parquet(out / INDEX_FILE)
    shard.unlink(missing_ok=True)
    texts_path.unlink(missing_ok=True)
    (out / INDEX_FILE).unlink(missing_ok=True)
    corpus_index.insert(0, "embedding_row", range(n_existing, n_existing + len(corpus_index)))
    if existing_idx is not None:
        corpus_index = pd.concat([existing_idx, corpus_index], ignore_index=True)
    corpus_index.to_parquet(out / "corpus_index.parquet", index=False)
    print(f"완료: {total:,}건 x {dim}차원")

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
        "corpus_rows": int(total),
        "embedding_dim": int(dim),
    }
    with open(out / "qwen_run_config.json", "w") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(json.dumps(run_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
