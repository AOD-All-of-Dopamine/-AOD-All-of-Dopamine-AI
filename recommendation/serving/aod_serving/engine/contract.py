"""아티팩트 계약 — 재임베딩 배치가 지켜야 하고 엔진이 기동 시 검증한다 (REC_TAB_DESIGN §8-6).

하나라도 어긋나면 `/health` 가 준비 안 됨으로 남는다. 반쯤 맞는 코퍼스로 서빙하지 않는다.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

from aod_serving.engine.bootstrap import rec_root


class ArtifactError(Exception):
    """아티팩트가 계약을 어겼다. 메시지는 `/health` 의 reason 으로 그대로 나간다."""


def schema_dir() -> Path:
    return rec_root() / "schemas"


def load_schema(platform: str, version: int = 1) -> dict:
    p = schema_dir() / f"{platform}.v{version}.json"
    if not p.exists():
        raise ArtifactError(f"계약 파일 없음: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _kind_ok(series, kind: str) -> bool:
    from pandas.api import types as T
    if kind == "int":   return T.is_integer_dtype(series.dtype)
    if kind == "float": return T.is_float_dtype(series.dtype)
    if kind == "bool":  return T.is_bool_dtype(series.dtype)
    if kind == "str":   return T.is_string_dtype(series.dtype) and not T.is_bool_dtype(series.dtype) \
                               and all(isinstance(v, str) for v in series.dropna().head(200))
    if kind == "list":  return series.dtype == object
    raise ArtifactError(f"계약 파일의 dtype {kind!r} 를 모른다")


def _is_empty(series) -> bool:
    """전부 결측이거나 전부 빈 목록이면 '비었다'."""
    s = series.dropna()
    if len(s) == 0:
        return True
    if series.dtype == object:
        return all((v is None) or (hasattr(v, "__len__") and len(v) == 0) for v in s)
    return False


def _nonzero(production: dict, key: str) -> bool:
    v = production.get(key)
    return bool(v) and v != 0


def validate_artifacts(d: Path, schema: dict, *, corpus_version: str, production: dict, verify_sha: bool = True) -> dict:
    """계약 검증. 통과하면 요약(dict)을 돌려주고, 아니면 ArtifactError."""
    import numpy as np, pandas as pd
    d = Path(d)
    mf = d / "manifest.json"
    if not mf.exists():
        raise ArtifactError(f"manifest.json 없음: {d} — `python -m aod_serving.tools.make_manifest` 로 만든다")
    manifest = json.loads(mf.read_text(encoding="utf-8"))
    for field, want in (("platform", schema["platform"]), ("corpus_version", corpus_version),
                        ("schema_version", schema["schema_version"])):
        if manifest.get(field) != want:
            raise ArtifactError(f"manifest {field}={manifest.get(field)!r} ≠ {want!r}")

    for name in schema["required_files"]:
        if not (d / name).exists():
            raise ArtifactError(f"필수 파일 없음: {name}")
        if verify_sha:
            want = manifest.get("files", {}).get(name)
            if not want:
                raise ArtifactError(f"manifest 에 {name} 의 sha256 이 없다")
            if sha256_file(d / name) != want:
                raise ArtifactError(f"sha256 불일치: {name} — 파일이 manifest 를 만든 뒤 바뀌었다")

    for cf in schema.get("conditional_files", []):
        w = cf["when"]; v = production.get(w["key"])
        needed = (v in w["in"]) if "in" in w else _nonzero(production, w["key"])
        if needed and not (d / cf["path"]).exists():
            raise ArtifactError(f"{w['key']}={v!r} 인데 부가 파일이 없다: {cf['path']}")

    emb = np.load(d / "corpus_embeddings.npy", mmap_mode="r")
    es = schema["embedding"]
    if emb.ndim != 2 or emb.shape[1] != es["dim"] or str(emb.dtype) != es["dtype"]:
        raise ArtifactError(f"임베딩 모양/타입 {emb.shape} {emb.dtype} ≠ (N, {es['dim']}) {es['dtype']}")
    n = emb.shape[0]
    sample = np.unique(np.linspace(0, n - 1, num=min(n, 512), dtype=np.int64))
    norms = np.linalg.norm(np.asarray(emb[sample], dtype=np.float64), axis=1)
    if np.abs(norms - 1.0).max() > es["norm_tolerance"]:
        raise ArtifactError(f"임베딩 노름이 1 이 아니다 (표본 최대 편차 {np.abs(norms - 1.0).max():.4f})")

    key = schema["key"]["column"]
    index = pd.read_parquet(d / "corpus_index.parquet")
    for col, kind in schema["corpus_index_columns"].items():
        if col not in index.columns or not _kind_ok(index[col], kind):
            raise ArtifactError(f"corpus_index.{col} 없음 또는 타입이 {kind} 가 아니다")
    ds = pd.read_parquet(d / "dataset.parquet")
    if not (len(index) == len(ds) == n == manifest.get("rows")):
        raise ArtifactError(f"행 수 불일치 — 임베딩 {n} · index {len(index)} · dataset {len(ds)} · manifest {manifest.get('rows')}")
    if not (index["embedding_row"].to_numpy() == np.arange(n)).all():
        raise ArtifactError("corpus_index.embedding_row 가 0..N-1 연속이 아니다")

    for col, spec in schema["dataset_columns"].items():
        weights = spec.get("weights", [])
        absent = col not in ds.columns
        if absent or _is_empty(ds[col]):
            if not weights:
                raise ArtifactError(f"dataset.{col} {'없음' if absent else '전부 비었다'}")
            on = [w for w in weights if _nonzero(production, w)]
            if on:
                raise ArtifactError(f"dataset.{col} 이 {'없는데' if absent else '전부 비었는데'} {on} 가 켜져 있다 — config.json 에서 0 으로 끈다")
            continue
        if not _kind_ok(ds[col], spec["dtype"]):
            raise ArtifactError(f"dataset.{col} 타입 {ds[col].dtype} 이 {spec['dtype']} 가 아니다")
    if not ds[key].is_unique:
        raise ArtifactError(f"dataset.{key} 가 유일하지 않다")
    if not (index[key].to_numpy() == ds[key].to_numpy()).all():
        raise ArtifactError(f"corpus_index 와 dataset 의 {key} 순서가 다르다 — 행 번호가 같은 작품을 가리켜야 한다")
    return {"rows": int(n), "dim": int(emb.shape[1]), "corpus_version": corpus_version, "manifest_created_at": manifest.get("created_at")}
