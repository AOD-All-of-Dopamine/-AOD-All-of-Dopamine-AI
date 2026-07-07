import pytest

from aod_ai.models import Extraction
from aod_ai.pipeline.embed import build_profile_embedding


def _extraction():
    return Extraction(
        fun_tags=[], normalized_summary="s",
        profile_text="임베딩 텍스트", extraction_quality=0.5,
    )


def test_build_profile_embedding_returns_1024_and_passes_profile_text(fake_emb):
    emb = fake_emb([0.0] * 1024)
    vec = build_profile_embedding(emb, _extraction())
    assert len(vec) == 1024
    assert emb.texts == ["임베딩 텍스트"]


def test_build_profile_embedding_rejects_wrong_dimension(fake_emb):
    emb = fake_emb([0.0] * 512)
    with pytest.raises(AssertionError):
        build_profile_embedding(emb, _extraction())
