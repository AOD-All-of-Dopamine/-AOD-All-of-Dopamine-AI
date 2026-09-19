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
    model = "Qwen/Qwen3-Embedding-0.6B"
    run_cfg = d / "qwen_run_config.json"
    if run_cfg.exists():
        cfg = json.loads(run_cfg.read_text(encoding="utf-8"))
        model = str(cfg.get("model") or cfg.get("model_name") or model)
    return {"platform": schema["platform"], "corpus_version": corpus_version, "schema_version": schema["schema_version"],
            "embedding_model": model, "dim": int(emb.shape[1]), "rows": int(emb.shape[0]),
            "text_builder_version": text_builder_version, "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": {name: sha256_file(d / name) for name in schema["required_files"]}}


def write_manifest_dict(d: Path, manifest: dict) -> Path:
    """이미 만든(그리고 검증까지 끝낸) manifest dict 를 임시 파일에 쓰고 이름을 바꾼다 —
    반쯤 쓰인 manifest 를 엔진이 읽지 않게. 쓰기 전에 검증하고 싶으면 `build_manifest` 로
    먼저 dict 를 만들어 `validate_artifacts(..., manifest=그 dict)` 로 본 뒤 이 함수로 쓴다."""
    d = Path(d); out = d / "manifest.json"; tmp = d / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    return out


def write_manifest(d: Path, schema: dict, **kw) -> Path:
    """`build_manifest` + `write_manifest_dict` 를 합친 편의 함수(주로 테스트용) — 쓰기 전 검증은 하지 않는다."""
    return write_manifest_dict(d, build_manifest(d, schema, **kw))
