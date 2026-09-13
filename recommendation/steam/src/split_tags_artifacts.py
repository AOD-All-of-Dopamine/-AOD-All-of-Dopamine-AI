"""전체 태그 임베딩이 끝난 뒤 `tags_full`(173,691) 에서 `tags_v1`(21,883) 을 파생한다.

`tags_v1` 은 리뷰 300+ 코퍼스 실험(`tags_v2_floor_sweep` 등)을 재현하기 위한 것이다.
행 순서가 같으므로 앞 21,883행을 잘라내면 그때와 같은 인덱스가 된다.

**절대 제자리에서 자르지 않는다.** 예전에 `tags_v1` 하나를 append 로 키워 쓰다가
같은 파일을 슬라이스해서 20시간짜리 임베딩을 날렸다 — 자세한 것은 `src/artifact_ops`.

    python -m src.split_tags_artifacts
"""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifact_ops import ARTIFACT_FILES, slice_npy, snapshot, stamp_representation, verify

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL = PROJECT_ROOT / "artifacts" / "tags_full"
V1 = PROJECT_ROOT / "artifacts" / "tags_v1"
FULL_ROWS = 173_691
#: 리뷰 300+ 코퍼스. dataset 은 커밋 fe1be7f 에 남아 있다(임베딩은 소실).
V1_COMMIT = "fe1be7f"
V1_ROWS = 21_883
REPRESENTATION = "description_tags_genres_modes"


def _dataset_at(commit: str) -> pd.DataFrame:
    blob = subprocess.run(
        ["git", "show", f"{commit}:./artifacts/tags_v1/dataset.parquet"],
        cwd=PROJECT_ROOT, capture_output=True, check=True,
    ).stdout
    return pd.read_parquet(io.BytesIO(blob))


def main() -> None:
    print(f"1) {FULL.name} 검증")
    stamp_representation(FULL, REPRESENTATION)
    print("  ", verify(FULL, expected_rows=FULL_ROWS))

    print("2) 백업")
    print("  ", snapshot(FULL, "post-embed"))

    ds_full = pd.read_parquet(FULL / "dataset.parquet")
    idx_full = pd.read_parquet(FULL / "corpus_index.parquet")

    print(f"3) 앞 {V1_ROWS:,}행이 {V1_COMMIT} 의 tags_v1 과 같은 순서인지 확인")
    ds21 = _dataset_at(V1_COMMIT)
    if len(ds21) != V1_ROWS:
        raise SystemExit(f"{V1_COMMIT} 의 dataset 이 {V1_ROWS:,}행이 아닙니다: {len(ds21):,}")
    if ds_full.head(V1_ROWS)["steam_appid"].tolist() != ds21["steam_appid"].tolist():
        raise SystemExit(
            "앞부분 appid 순서가 다릅니다 — 슬라이스로는 옛 실험을 재현할 수 없습니다."
        )
    # 성인 필터용 컬럼은 나중에 붙였으므로 복원본에도 채운다
    for col in ("content_descriptorids", "tags"):
        if col in ds_full.columns:
            ds21[col] = ds_full.head(V1_ROWS)[col].values

    print(f"4) {V1.name} 를 새 경로에 만들고 검증 뒤 교체")
    staged = V1.with_name(V1.name + ".staged")
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    slice_npy(FULL / "corpus_embeddings.npy", staged / "corpus_embeddings.npy", V1_ROWS)
    idx_full.head(V1_ROWS).to_parquet(staged / "corpus_index.parquet", index=False)
    ds21.to_parquet(staged / "dataset.parquet", index=False)
    for f in ARTIFACT_FILES:
        if not (staged / f).exists() and (FULL / f).exists():
            shutil.copy2(FULL / f, staged / f)
    print("  ", verify(staged, expected_rows=V1_ROWS))

    if V1.exists():
        old = V1.with_name(V1.name + ".pre-split")
        if old.exists():
            shutil.rmtree(old)
        V1.rename(old)          # 지우지 않는다 — 밀어놓는다
        print(f"   기존 {V1.name} → {old.name}")
    staged.rename(V1)

    print("5) 최종")
    for d in (FULL, V1):
        print(f"   {d.name:12s} {verify(d)}")

    # 슬라이스가 정말 같은 벡터인지 — 검증 없이 '앞부분이니까 같겠지' 로 넘어가지 않는다
    a = np.load(FULL / "corpus_embeddings.npy", mmap_mode="r")
    b = np.load(V1 / "corpus_embeddings.npy", mmap_mode="r")
    rows = np.linspace(0, V1_ROWS - 1, 8).astype(int)
    assert np.allclose(np.array(a[rows]), np.array(b[rows])), "슬라이스가 원본과 다릅니다"
    print("   슬라이스 대조 통과")


if __name__ == "__main__":
    main()
