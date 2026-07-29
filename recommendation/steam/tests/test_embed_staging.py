# tests/test_embed_staging.py
"""단계적 임베딩 — 가치 있는 것부터 하고 나머지는 나중에 이어붙인다.

17만 건을 한 번에 임베딩하면 24시간이 넘는다. 리뷰 수가 알려진 게임만 먼저 하면
6시간이면 되고, 나머지는 필요해질 때 --append 로 붙인다. 버리는 것이 아니라 미루는 것이다.
"""
import sys

import numpy as np
import pandas as pd
import pytest

from src.embed_qwen import select_targets


def _dataset(n_with_reviews=3, n_without=2):
    rows = []
    for i in range(n_with_reviews):
        rows.append({"steam_appid": 100 + i, "name": f"reviewed{i}",
                     "semantic_text": f"t{i}", "has_recommendations": True})
    for i in range(n_without):
        rows.append({"steam_appid": 200 + i, "name": f"obscure{i}",
                     "semantic_text": f"u{i}", "has_recommendations": False})
    return pd.DataFrame(rows)


@pytest.fixture
def argv(monkeypatch):
    def _set(*flags):
        monkeypatch.setattr(sys, "argv", ["embed_qwen", *flags])
    return _set


def test_only_with_reviews_keeps_quality_subset(tmp_path, argv):
    argv("--only-with-reviews")
    df, existing = select_targets(_dataset(), tmp_path)
    assert len(df) == 3
    assert df["has_recommendations"].all()
    assert existing is None


def test_no_flags_embeds_everything(tmp_path, argv):
    argv()
    df, existing = select_targets(_dataset(), tmp_path)
    assert len(df) == 5
    assert existing is None


def test_refuses_to_overwrite_existing_embeddings(tmp_path, argv):
    """실수로 6시간짜리 결과를 날리지 않도록 막는다."""
    np.save(tmp_path / "corpus_embeddings.npy", np.zeros((3, 4), dtype="float32"))
    argv()
    with pytest.raises(SystemExit, match="--append"):
        select_targets(_dataset(), tmp_path)


def test_force_allows_overwrite(tmp_path, argv):
    np.save(tmp_path / "corpus_embeddings.npy", np.zeros((3, 4), dtype="float32"))
    argv("--force")
    df, existing = select_targets(_dataset(), tmp_path)
    assert len(df) == 5


def test_append_skips_already_embedded(tmp_path, argv):
    """1단계에서 한 것을 건너뛰고 나머지만 골라야 한다."""
    np.save(tmp_path / "corpus_embeddings.npy", np.zeros((3, 4), dtype="float32"))
    pd.DataFrame({
        "embedding_row": [0, 1, 2],
        "steam_appid": [100, 101, 102],
        "name": ["reviewed0", "reviewed1", "reviewed2"],
    }).to_parquet(tmp_path / "corpus_index.parquet", index=False)

    argv("--append")
    df, existing = select_targets(_dataset(), tmp_path)
    assert df["steam_appid"].tolist() == [200, 201]   # 아직 안 한 것만
    assert len(existing) == 3


def test_append_without_existing_fails(tmp_path, argv):
    argv("--append")
    with pytest.raises(SystemExit, match="기존 임베딩이 없습니다"):
        select_targets(_dataset(), tmp_path)


def test_append_row_numbers_continue_from_existing():
    """이어붙인 행 번호가 기존과 겹치면 저장된 벡터와 어긋난다."""
    old_emb = np.zeros((3, 4), dtype="float32")
    new_index = pd.DataFrame({"steam_appid": [200, 201], "name": ["a", "b"]})
    new_index.insert(0, "embedding_row", range(len(old_emb), len(old_emb) + len(new_index)))
    assert new_index["embedding_row"].tolist() == [3, 4]
