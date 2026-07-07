from __future__ import annotations

from pathlib import Path

import psycopg

from aod_ai.config import Settings
from aod_ai.db import connect

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def _applied_versions(conn: psycopg.Connection) -> set[str]:
    row = conn.execute("SELECT to_regclass('aod_ai.schema_migrations')").fetchone()
    if row[0] is None:
        return set()
    rows = conn.execute("SELECT version FROM aod_ai.schema_migrations").fetchall()
    return {r[0] for r in rows}


def run_migrations(
    conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR
) -> list[str]:
    applied = _applied_versions(conn)
    newly: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        version = path.name
        if version in applied:
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO aod_ai.schema_migrations (version) VALUES (%s)",
            (version,),
        )
        newly.append(version)
    return newly


def main() -> None:
    settings = Settings()
    conn = connect(settings, register_pgvector=False)
    try:
        print(f"applied migrations: {run_migrations(conn)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
