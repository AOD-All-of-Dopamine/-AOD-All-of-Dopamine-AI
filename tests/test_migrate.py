from aod_ai.db import connect
from aod_ai.migrate import run_migrations

ALL_VERSIONS = [
    "001_schema_pgvector.sql",
    "002_asset_tables.sql",
    "003_user_and_log_tables.sql",
    "004_hnsw_indexes.sql",
]
ASSET_TABLES = {
    "schema_migrations",
    "fun_tag_dict",
    "content_semantic_profile",
    "content_fun_tag",
    "content_embedding",
    "content_quality_score",
    "user_profile_cache",
    "rec_impression",
    "rec_event",
}


def test_migrate_applies_all_and_is_idempotent(settings):
    c = connect(settings, register_pgvector=False)
    # 자기 격리: 다른 테스트(conn 픽스처)가 공유 컨테이너에 이미 마이그레이션을
    # 적용했을 수 있으므로, ALL_VERSIONS 검증 전에 aod_ai 스키마를 초기화한다.
    c.execute("DROP SCHEMA IF EXISTS aod_ai CASCADE")
    assert run_migrations(c) == ALL_VERSIONS
    assert run_migrations(c) == []

    tables = {
        r[0]
        for r in c.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'aod_ai'"
        ).fetchall()
    }
    assert ASSET_TABLES <= tables

    idx = {
        r[0]
        for r in c.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'aod_ai'"
        ).fetchall()
    }
    assert "idx_content_embedding_hnsw" in idx
    assert "idx_user_profile_vector_hnsw" in idx
    assert "idx_content_fun_tag_tag" in idx
    c.close()
