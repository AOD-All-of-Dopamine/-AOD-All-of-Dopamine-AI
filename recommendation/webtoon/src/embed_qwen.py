"""코퍼스 임베딩 — Qwen3-Embedding-0.6B, 1024차원, L2 정규화.

세 플랫폼과 **같은 모델·같은 정규화**를 쓴다. 그래야 나중에 통합 층에 붙일 수 있다
(다만 통합은 순위만 섞으므로 벡터 호환이 필수는 아니다 — X-16/X-23 참조).

메모리 주의: 이 프로세스와 백엔드·리랭커를 **같이 띄우지 않는다.** 과거에 그렇게 하다
cgroup 한계에서 죽었다. 단독으로 돌린다.

    PYTHONPATH=. python -m src.embed_qwen [--out artifacts/wt_v1] [--rep v1|v2]
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np, pandas as pd
from src.config import ensure_artifacts_dir, artifact_dir
from src.text_builder import add_semantic_text

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def encode_with_backoff(model, texts, batch: int) -> np.ndarray:
    while True:
        try:
            return model.encode(texts, batch_size=batch, normalize_embeddings=True,
                                show_progress_bar=True, convert_to_numpy=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch > 1:
                batch //= 2; print(f"OOM → batch {batch}", flush=True)
            else:
                raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--rep", default="v2", choices=["v1", "v2"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-seq", dest="max_seq", type=int, default=512)
    a = ap.parse_args()
    out = ensure_artifacts_dir(a.out)

    df = pd.read_parquet(artifact_dir(None) / "dataset.parquet")
    df = add_semantic_text(df, with_tags=(a.rep == "v2"))
    print(f"대상 {len(df):,}편 · rep_{a.rep} · 텍스트 중앙 {int(df['semantic_text'].str.len().median())}자",
          flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    # **반드시 줄인다.** Qwen3-Embedding 의 기본 max_seq_length 는 32768 이라
    # 짧은 웹툰 텍스트(중앙 187자)에도 거대한 할당을 만들어 CPU 인코딩이 사실상 멈춘다.
    # 실측: 미설정으로 돌렸더니 7분에 배치 1개도 못 끝냈다. 512 면 관측 최대치를 덮는다.
    if model.max_seq_length > a.max_seq:
        print(f"max_seq_length {model.max_seq_length} → {a.max_seq}", flush=True)
        model.max_seq_length = a.max_seq
    emb = encode_with_backoff(model, df["semantic_text"].tolist(), a.batch).astype("float32")

    np.save(out / "corpus_embeddings.npy", emb)
    pd.DataFrame({"embedding_row": range(len(df)), "item_id": df["item_id"].values,
                  "name": df["name"].values}).to_parquet(out / "corpus_index.parquet", index=False)
    df.to_parquet(out / "dataset.parquet", index=False)
    json.dump({"model": MODEL, "dim": int(emb.shape[1]), "rows": int(emb.shape[0]),
               "rep": a.rep, "normalized": True},
              open(out / "qwen_run_config.json", "w"), ensure_ascii=False, indent=1)
    print(f"저장 {emb.shape} → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
