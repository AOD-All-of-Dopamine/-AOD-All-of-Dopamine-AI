from aod_ai.db import connect
from aod_ai.migrate import run_migrations


def test_migrate_applies_001_and_is_idempotent(settings):
    c = connect(settings, register_pgvector=False)
    c.execute("DROP SCHEMA IF EXISTS aod_ai CASCADE")  # 자기 격리
    first = run_migrations(c)
    assert first == ["001_schema_pgvector.sql"]
    second = run_migrations(c)
    assert second == []
    row = c.execute(
        "SELECT count(*) FROM aod_ai.schema_migrations WHERE version = %s",
        ("001_schema_pgvector.sql",),
    ).fetchone()
    assert row[0] == 1
    c.close()
