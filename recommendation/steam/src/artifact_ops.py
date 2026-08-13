"""임베딩 아티팩트를 파생·이동시키는 유지보수 연산.

**왜 별도 모듈인가** — 2026-08-09 에 173,691 x 1024 임베딩(CPU 20시간)을 유지보수
스크립트 한 줄로 날렸다. 두 가지가 겹쳤다.

  1. `np.save(p, np.asarray(np.load(p, mmap_mode="r")[:n]))`
     `np.asarray(memmap)` 은 복사가 아니라 **뷰**다. `np.save` 가 읽는 중인 파일을
     스스로 truncate 해서 711MB 가 4KB 로 잘렸다.
  2. 잘린 파일을 유일한 백업본 위로 복사해서 백업까지 같이 죽였다.

그래서 이 모듈의 두 규칙은 타협하지 않는다.
  · 원본과 대상이 같은 파일이면 **거부한다** (경고가 아니라 예외).
  · 아티팩트를 지우지 않는다 — 항상 `.bak-<태그>` 로 밀어놓고, 검증이 끝난 뒤에 교체한다.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

#: 하나의 임베딩 아티팩트를 이루는 파일들. 없는 것은 건너뛴다.
ARTIFACT_FILES = (
    "corpus_embeddings.npy",
    "corpus_index.parquet",
    "dataset.parquet",
    "anchor_embeddings.npy",
    "anchors_40.parquet",
    "qwen_run_config.json",
)


def _same_file(a: Path, b: Path) -> bool:
    """심링크·하드링크·`./x` 대 `x` 까지 뚫고 같은 실체인지 본다."""
    a, b = Path(a), Path(b)
    if not a.exists() or not b.exists():
        return a.resolve() == b.resolve()
    return a.samefile(b)


def slice_npy(src: Path, dst: Path, n: int) -> np.ndarray:
    """`src` 의 앞 `n` 행을 `dst` 로 쓴다. `src is dst` 면 거부한다.

    `mmap` 으로 읽고 `np.array` 로 **실복사**한 뒤에 쓴다 — 뷰를 그대로 넘기면
    쓰기가 읽는 중인 파일을 자른다.
    """
    src, dst = Path(src), Path(dst)
    if _same_file(src, dst):
        raise ValueError(
            f"원본과 대상이 같은 파일입니다: {src}. "
            "제자리 슬라이스는 읽는 중인 파일을 truncate 해서 데이터를 파괴합니다 "
            "— 다른 경로로 쓴 뒤 교체하세요."
        )
    view = np.load(src, mmap_mode="r")
    if n > len(view):
        raise ValueError(f"{src} 는 {len(view):,}행뿐인데 {n:,}행을 요구했습니다.")
    head = np.array(view[:n])          # 뷰가 아니라 복사여야 한다
    del view
    np.save(dst, head)
    return head


def snapshot(art_dir: Path, tag: str) -> Path:
    """아티팩트를 `<이름>.bak-<태그>` 로 복제하고 바이트 수까지 대조한다.

    아티팩트를 건드리는 모든 작업의 첫 줄이어야 한다. 20시간짜리를 날려보면
    "이건 그냥 복사인데" 라는 판단이 얼마나 비싼지 알게 된다.
    """
    art_dir = Path(art_dir)
    dest = art_dir.with_name(f"{art_dir.name}.bak-{tag}")
    if dest.exists():
        raise FileExistsError(f"{dest} 가 이미 있습니다 — 덮어쓰지 않습니다.")
    shutil.copytree(art_dir, dest)
    for f in art_dir.iterdir():
        if f.is_file() and f.stat().st_size != (dest / f.name).stat().st_size:
            raise OSError(f"백업 크기 불일치: {f.name}")
    return dest


def verify(art_dir: Path, expected_rows: int | None = None) -> dict:
    """행수·appid 정렬·행번호 연속·L2 노름을 확인한다. 어긋나면 예외."""
    art_dir = Path(art_dir)
    emb = np.load(art_dir / "corpus_embeddings.npy", mmap_mode="r")
    idx = pd.read_parquet(art_dir / "corpus_index.parquet")
    ds = pd.read_parquet(art_dir / "dataset.parquet", columns=["steam_appid"])

    if not len(emb) == len(idx) == len(ds):
        raise ValueError(f"행수 불일치: 임베딩 {len(emb):,} 인덱스 {len(idx):,} dataset {len(ds):,}")
    if expected_rows is not None and len(emb) != expected_rows:
        raise ValueError(f"행수가 {expected_rows:,} 가 아닙니다: {len(emb):,}")
    if not (idx["steam_appid"].values == ds["steam_appid"].values).all():
        raise ValueError("corpus_index 와 dataset 의 appid 정렬이 다릅니다.")
    if not (idx["embedding_row"].values == np.arange(len(idx))).all():
        raise ValueError("embedding_row 가 0..N-1 연속이 아닙니다.")

    probe = np.array(emb[np.linspace(0, len(emb) - 1, min(16, len(emb))).astype(int)])
    norms = np.linalg.norm(probe, axis=1)
    if np.isnan(probe).any():
        raise ValueError("임베딩에 NaN 이 있습니다.")
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"L2 정규화가 깨졌습니다: {norms.min():.4f}~{norms.max():.4f}")

    return {"rows": len(emb), "dim": int(emb.shape[1]), "norm_min": float(norms.min())}


def stamp_representation(art_dir: Path, representation: str) -> None:
    """`qwen_run_config.json` 의 표현 라벨을 실제와 맞춘다.

    태그를 넣고도 라벨이 `description_genres_modes` 로 남아 있었다 — 산출물이
    자기가 무엇인지 거짓말하면 몇 주 뒤에 어떤 임베딩인지 아무도 모른다.
    """
    p = Path(art_dir) / "qwen_run_config.json"
    cfg = json.loads(p.read_text())
    cfg["representation"] = representation
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
