from pathlib import Path

ROLES_SQL = Path(__file__).resolve().parent.parent / "sql" / "roles.sql"


def test_roles_sql_grants_readonly_public_and_rw_aod_ai_idempotent(conn, settings):
    sql = ROLES_SQL.read_text(encoding="utf-8").replace(
        ':"ai_user"', f'"{settings.aod_db_user}"'
    )
    conn.execute(sql)
    conn.execute(sql)  # 멱등 재실행

    assert (
        conn.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = 'aod_public_ro'"
        ).fetchone()
        is not None
    )
    # 배치 AI 계정이 public 읽기전용 role 멤버
    assert (
        conn.execute(
            "SELECT pg_has_role(%s, 'aod_public_ro', 'MEMBER')",
            (settings.aod_db_user,),
        ).fetchone()[0]
        is True
    )
    # aod_ai 스키마 USAGE + 테이블 쓰기 권한 (RDS 비-슈퍼유저 계정에서 load-bearing)
    assert (
        conn.execute(
            "SELECT has_schema_privilege(%s, 'aod_ai', 'USAGE')",
            (settings.aod_db_user,),
        ).fetchone()[0]
        is True
    )
    assert (
        conn.execute(
            "SELECT has_table_privilege(%s, 'aod_ai.content_fun_tag', 'INSERT')",
            (settings.aod_db_user,),
        ).fetchone()[0]
        is True
    )
