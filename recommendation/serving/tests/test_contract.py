import json
import numpy as np, pandas as pd, pytest
from aod_serving.engine.contract import ArtifactError, load_schema, validate_artifacts
from aod_serving.engine.manifest import write_manifest

PROD = {"pop_boost": 0.0}


def ok(corpus, wn_schema, production=PROD):
    write_manifest(corpus, wn_schema, corpus_version="wn_test")
    return validate_artifacts(corpus, wn_schema, corpus_version="wn_test", production=production)


def test_load_schema_by_platform():
    assert load_schema("steam")["key"]["column"] == "steam_appid"
    with pytest.raises(ArtifactError): load_schema("netflix")


def test_valid_corpus_passes_and_reports_rows(corpus, wn_schema):
    assert ok(corpus, wn_schema)["rows"] == 6


def test_missing_manifest(corpus, wn_schema):
    with pytest.raises(ArtifactError, match="manifest.json"):
        validate_artifacts(corpus, wn_schema, corpus_version="wn_test", production=PROD)


def test_manifest_for_another_corpus_version(corpus, wn_schema):
    write_manifest(corpus, wn_schema, corpus_version="other")
    with pytest.raises(ArtifactError, match="corpus_version"):
        validate_artifacts(corpus, wn_schema, corpus_version="wn_test", production=PROD)


def test_tampered_file_fails_sha(corpus, wn_schema):
    write_manifest(corpus, wn_schema, corpus_version="wn_test")
    ds = pd.read_parquet(corpus / "dataset.parquet"); ds.loc[0, "name"] = "바뀜"; ds.to_parquet(corpus / "dataset.parquet")
    with pytest.raises(ArtifactError, match="sha256"):
        validate_artifacts(corpus, wn_schema, corpus_version="wn_test", production=PROD)


def test_missing_required_column(corpus, wn_schema):
    pd.read_parquet(corpus / "dataset.parquet").drop(columns=["episode_count"]).to_parquet(corpus / "dataset.parquet")
    with pytest.raises(ArtifactError, match="episode_count"): ok(corpus, wn_schema)


def test_wrong_dtype(corpus, wn_schema):
    ds = pd.read_parquet(corpus / "dataset.parquet"); ds["age_limit"] = "성인"; ds.to_parquet(corpus / "dataset.parquet")
    with pytest.raises(ArtifactError, match="age_limit"): ok(corpus, wn_schema)


def test_row_count_mismatch(corpus, wn_schema):
    pd.read_parquet(corpus / "dataset.parquet").iloc[:-1].to_parquet(corpus / "dataset.parquet")
    with pytest.raises(ArtifactError, match="행 수"): ok(corpus, wn_schema)


def test_duplicate_key(corpus, wn_schema):
    ds = pd.read_parquet(corpus / "dataset.parquet"); ds.loc[1, "item_id"] = ds.loc[0, "item_id"]; ds.to_parquet(corpus / "dataset.parquet")
    with pytest.raises(ArtifactError, match="유일"): ok(corpus, wn_schema)


def test_index_and_dataset_key_order_must_match(corpus, wn_schema):
    ds = pd.read_parquet(corpus / "dataset.parquet"); ds.iloc[::-1].reset_index(drop=True).to_parquet(corpus / "dataset.parquet")
    with pytest.raises(ArtifactError, match="순서"): ok(corpus, wn_schema)


def test_embedding_row_not_contiguous(corpus, wn_schema):
    ix = pd.read_parquet(corpus / "corpus_index.parquet"); ix.loc[0, "embedding_row"] = 3; ix.to_parquet(corpus / "corpus_index.parquet")
    with pytest.raises(ArtifactError, match="embedding_row"): ok(corpus, wn_schema)


def test_embeddings_must_be_normalized(corpus, wn_schema):
    e = np.load(corpus / "corpus_embeddings.npy"); np.save(corpus / "corpus_embeddings.npy", e * 3)
    with pytest.raises(ArtifactError, match="노름"): ok(corpus, wn_schema)


def test_empty_weighted_column_requires_zero_weight(corpus, wn_schema):
    ds = pd.read_parquet(corpus / "dataset.parquet"); ds["interest_count"] = pd.array([None] * len(ds), dtype="Int64")
    ds.to_parquet(corpus / "dataset.parquet")
    assert ok(corpus, wn_schema, {"pop_boost": 0.0})["rows"] == 6            # 꺼져 있으면 통과
    with pytest.raises(ArtifactError, match="pop_boost"): ok(corpus, wn_schema, {"pop_boost": 0.03})


def test_weighted_column_may_be_absent_when_weight_is_zero(corpus, wn_schema):
    pd.read_parquet(corpus / "dataset.parquet").drop(columns=["interest_count"]).to_parquet(corpus / "dataset.parquet")
    assert ok(corpus, wn_schema, {"pop_boost": 0.0})["rows"] == 6


def test_conditional_file_required_only_when_enabled(corpus, wn_schema):
    schema = {**wn_schema, "conditional_files": [{"path": "extra.parquet", "when": {"key": "extra_w", "nonzero": True}}]}
    assert ok(corpus, schema, {"pop_boost": 0.0, "extra_w": 0.0})["rows"] == 6
    with pytest.raises(ArtifactError, match="extra.parquet"): ok(corpus, schema, {"pop_boost": 0.0, "extra_w": 0.2})
