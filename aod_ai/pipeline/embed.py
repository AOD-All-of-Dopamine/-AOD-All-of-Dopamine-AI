def build_profile_embedding(emb, extraction) -> list[float]:
    vector = emb.embed_text(extraction.profile_text)
    assert len(vector) == 1024, f"expected 1024-dim embedding, got {len(vector)}"
    return vector
