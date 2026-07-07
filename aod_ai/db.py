from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from aod_ai.config import Settings


def connect(settings: Settings, *, register_pgvector: bool = True) -> psycopg.Connection:
    conn = psycopg.connect(
        host=settings.aod_db_host,
        port=settings.aod_db_port,
        dbname=settings.aod_db_name,
        user=settings.aod_db_user,
        password=settings.aod_db_password,
        autocommit=True,
    )
    conn.execute("SET search_path TO aod_ai, public")
    if register_pgvector:
        register_vector(conn)
    return conn
