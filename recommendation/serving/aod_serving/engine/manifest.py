"""manifest.json — 코퍼스 폴더의 신분증. 재임베딩 배치가 폴더를 게시할 때 쓰고, 엔진이 기동 시 대조한다."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from aod_serving.engine.contract import sha256_file


def build_manifest(d: Path, schema: dict, *, corpus_version: str, text_builder_version: str = "unknown",
                   source: str = "external-crawl") -> dict:
    import numpy as np
    d = Path(d)
    emb = np.load(d / "corpus_embeddings.npy", mmap_mode="r")
    model = "Qwen3-Embedding-0.6B"
    run_cfg = d / "qwen_run_config.json"
    if run_cfg.exists():
        cfg = json.loads(run_cfg.read_text(encoding="utf-8"))
        model = str(cfg.get("model") or cfg.get("model_name") or model)
    return {"platform": schema["platform"], "corpus_version": corpus_version, "schema_version": schema["schema_version"],
            "embedding_model": model, "dim": int(emb.shape[1]), "rows": int(emb.shape[0]),
            "text_builder_version": text_builder_version, "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": {name: sha256_file(d / name) for name in schema["required_files"]}}


def write_manifest(d: Path, schema: dict, **kw) -> Path:
    """임시 파일에 쓰고 이름을 바꾼다 — 반쯤 쓰인 manifest 를 엔진이 읽지 않게."""
    d = Path(d); out = d / "manifest.json"; tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(build_manifest(d, schema, **kw), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return out
