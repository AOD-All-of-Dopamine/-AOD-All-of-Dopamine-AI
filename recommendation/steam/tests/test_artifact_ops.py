"""artifact_ops 의 파괴 방지 장치가 실제로 막는지 본다.

여기 있는 테스트는 전부 실제로 데이터를 날렸던 경로를 재현한 것이다 — 2026-08-09,
173,691 x 1024 임베딩(CPU 20시간)이 슬라이스 한 줄에 4KB 로 잘렸다.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.artifact_ops import slice_npy, snapshot, stamp_representation, verify


def _make_artifact(d, n=8, dim=4):
    d.mkdir(parents=True, exist_ok=True)
    e = np.random.default_rng(0).normal(size=(n, dim)).astype("float32")
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    np.save(d / "corpus_embeddings.npy", e)
    appids = list(range(100, 100 + n))
    pd.DataFrame({"embedding_row": range(n), "steam_appid": appids,
                  "name": [f"g{i}" for i in range(n)]}).to_parquet(d / "corpus_index.parquet")
    pd.DataFrame({"steam_appid": appids}).to_parquet(d / "dataset.parquet")
    (d / "qwen_run_config.json").write_text(json.dumps({"representation": "old"}))
    return e


def test_slice_npy_refuses_to_overwrite_its_own_source(tmp_path):
    """제자리 슬라이스가 원본을 파괴하기 전에 막혀야 한다 — 실제 사고의 원인."""
    e = _make_artifact(tmp_path / "art")
    p = tmp_path / "art" / "corpus_embeddings.npy"

    with pytest.raises(ValueError, match="같은 파일"):
        slice_npy(p, p, 4)

    # 거부만 하는 게 아니라 원본이 멀쩡해야 의미가 있다
    assert np.allclose(np.load(p), e)


def test_slice_npy_refuses_symlink_to_same_file(tmp_path):
    """`./x` 대 `x`, 심링크로 우회해도 같은 실체면 막아야 한다."""
    _make_artifact(tmp_path / "art")
    p = tmp_path / "art" / "corpus_embeddings.npy"
    link = tmp_path / "alias.npy"
    link.symlink_to(p)

    with pytest.raises(ValueError, match="같은 파일"):
        slice_npy(p, link, 4)


def test_slice_npy_copies_head_to_a_different_path(tmp_path):
    e = _make_artifact(tmp_path / "art")
    src = tmp_path / "art" / "corpus_embeddings.npy"
    dst = tmp_path / "head.npy"

    head = slice_npy(src, dst, 5)

    assert head.shape == (5, 4)
    assert np.allclose(np.load(dst), e[:5])
    assert np.allclose(np.load(src), e), "원본은 그대로여야 한다"


def test_slice_npy_rejects_n_larger_than_source(tmp_path):
    _make_artifact(tmp_path / "art", n=8)
    with pytest.raises(ValueError, match="행뿐인데"):
        slice_npy(tmp_path / "art" / "corpus_embeddings.npy", tmp_path / "h.npy", 99)


def test_snapshot_copies_and_refuses_to_clobber(tmp_path):
    _make_artifact(tmp_path / "art")

    bak = snapshot(tmp_path / "art", "t1")
    assert bak.name == "art.bak-t1"
    assert (bak / "corpus_embeddings.npy").exists()

    with pytest.raises(FileExistsError):
        snapshot(tmp_path / "art", "t1")


def test_verify_accepts_a_sound_artifact(tmp_path):
    _make_artifact(tmp_path / "art", n=8, dim=4)
    assert verify(tmp_path / "art", expected_rows=8) == {"rows": 8, "dim": 4, "norm_min": pytest.approx(1.0, abs=1e-3)}


def test_verify_catches_a_truncated_embedding(tmp_path):
    """사고 당시 상태 — 임베딩만 잘리고 인덱스는 멀쩡했다."""
    _make_artifact(tmp_path / "art", n=8)
    np.save(tmp_path / "art" / "corpus_embeddings.npy", np.zeros((1, 4), dtype="float32"))

    with pytest.raises(ValueError, match="행수 불일치"):
        verify(tmp_path / "art")


def test_verify_catches_misaligned_appids(tmp_path):
    _make_artifact(tmp_path / "art", n=8)
    ds = pd.read_parquet(tmp_path / "art" / "dataset.parquet")
    ds["steam_appid"] = ds["steam_appid"][::-1].values
    ds.to_parquet(tmp_path / "art" / "dataset.parquet")

    with pytest.raises(ValueError, match="정렬이 다릅니다"):
        verify(tmp_path / "art")


def test_verify_catches_unnormalized_embeddings(tmp_path):
    _make_artifact(tmp_path / "art", n=8)
    np.save(tmp_path / "art" / "corpus_embeddings.npy",
            np.full((8, 4), 3.0, dtype="float32"))

    with pytest.raises(ValueError, match="L2 정규화"):
        verify(tmp_path / "art")


def test_stamp_representation_rewrites_the_label(tmp_path):
    _make_artifact(tmp_path / "art")
    stamp_representation(tmp_path / "art", "description_tags_genres_modes")
    cfg = json.loads((tmp_path / "art" / "qwen_run_config.json").read_text())
    assert cfg["representation"] == "description_tags_genres_modes"
