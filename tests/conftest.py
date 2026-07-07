from __future__ import annotations

import os
import pathlib

import psycopg
import pytest
from pgvector.psycopg import register_vector
from testcontainers.postgres import PostgresContainer

from aod_ai.config import Settings
from aod_ai.db import connect
from aod_ai.migrate import run_migrations


@pytest.fixture(scope="session")
def pg_container():
    with PostgresContainer("pgvector/pgvector:pg16") as pg:
        yield pg


@pytest.fixture()
def settings(pg_container) -> Settings:
    return Settings(
        _env_file=None,
        aod_db_host=pg_container.get_container_host_ip(),
        aod_db_port=int(pg_container.get_exposed_port(5432)),
        aod_db_name=pg_container.dbname,
        aod_db_user=pg_container.username,
        aod_db_password=pg_container.password,
        openai_base_url="http://test",
        openai_api_key="test",
        llm_model="test-llm",
        embedding_model="test-emb",
    )


@pytest.fixture()
def conn(settings):
    bootstrap = connect(settings, register_pgvector=False)
    run_migrations(bootstrap)
    bootstrap.close()
    c = connect(settings)  # vector 확장 존재 → register_vector 성공
    yield c
    c.close()


# ---------------------------------------------------------------------------
# M1 픽스처 (plan 2026-07-03-aod-rec-m1 Task 1~4 추가분)
# 적응: 로컬 pg 대신 M0의 세션 pgvector 컨테이너 안에 별도 DB(aod_ai_test)를 만들어
# M0 테스트(기본 DB 사용)와 완전히 격리한다. AOD_TEST_DSN 오버라이드는 유지.
# ---------------------------------------------------------------------------

_MIGRATIONS = pathlib.Path(__file__).resolve().parents[1] / "migrations"

_PUBLIC_DDL = """
CREATE TABLE public.contents (
  content_id     bigint PRIMARY KEY,
  domain         varchar(50) NOT NULL,
  master_title   varchar(500) NOT NULL,
  original_title varchar(500),
  release_date   date,
  synopsis       text,
  average_score  double precision,
  review_count   integer,
  created_at     timestamptz DEFAULT now(),
  updated_at     timestamptz DEFAULT now()
);
CREATE TABLE public.webnovel_contents (
  content_id bigint PRIMARY KEY,
  author     varchar(200),
  publisher  varchar(200),
  age_rating varchar(50),
  genres     text[],
  platforms  text[]
);
CREATE TABLE public.external_ranking (
  id                   bigserial PRIMARY KEY,
  platform_specific_id varchar(255),
  content_id           bigint,
  title                varchar(500),
  ranking              integer,
  platform             varchar(100),
  thumbnail_url        varchar(1000),
  watch_providers      jsonb
)
"""


def _run_script(conn, sql: str) -> None:
    for stmt in sql.split(";"):
        if stmt.strip():
            conn.execute(stmt)


@pytest.fixture(scope="session")
def m1_dsn(pg_container) -> str:
    override = os.environ.get("AOD_TEST_DSN")
    if override:
        return override
    host = pg_container.get_container_host_ip()
    port = int(pg_container.get_exposed_port(5432))
    user = pg_container.username
    password = pg_container.password
    admin = psycopg.connect(
        host=host, port=port, dbname=pg_container.dbname,
        user=user, password=password, autocommit=True,
    )
    exists = admin.execute(
        "SELECT 1 FROM pg_database WHERE datname = 'aod_ai_test'"
    ).fetchone()
    if not exists:
        admin.execute("CREATE DATABASE aod_ai_test")
    admin.close()
    return f"postgresql://{user}:{password}@{host}:{port}/aod_ai_test"


# opt-in(NON-autouse): Settings()를 인스턴스화하는 M1 테스트만 명시적으로 요청한다.
# autouse가 아니므로 M0 테스트(test_config/test_clients_*)의 env는 절대 건드리지 않는다.
@pytest.fixture()
def m1_env(monkeypatch):
    monkeypatch.setenv("AOD_DB_HOST", "localhost")
    monkeypatch.setenv("AOD_DB_NAME", "aod_ai_test")
    monkeypatch.setenv("AOD_DB_USER", "postgres")
    monkeypatch.setenv("AOD_DB_PASSWORD", "postgres")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "qwen-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen-embed-test")
    monkeypatch.setenv("VANE_BASE_URL", "http://localhost:3000")


@pytest.fixture(scope="session")
def _schema(m1_dsn):
    conn = psycopg.connect(m1_dsn, autocommit=True)
    conn.execute("DROP SCHEMA IF EXISTS aod_ai CASCADE")
    conn.execute("DROP TABLE IF EXISTS public.contents CASCADE")
    conn.execute("DROP TABLE IF EXISTS public.webnovel_contents CASCADE")
    conn.execute("DROP TABLE IF EXISTS public.external_ranking CASCADE")
    conn.execute("CREATE SCHEMA aod_ai")
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for sql_file in sorted(_MIGRATIONS.glob("*.sql")):
        _run_script(conn, sql_file.read_text(encoding="utf-8"))
    _run_script(conn, _PUBLIC_DDL)
    conn.close()
    yield


class _FakeVane:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, *, query, sources, system_instructions=None):
        self.calls.append(
            {"query": query, "sources": sources, "system_instructions": system_instructions}
        )
        return list(self.results)


@pytest.fixture()
def fake_vane():
    return lambda results: _FakeVane(results)


class _FakeLlm:
    def __init__(self, extraction):
        self.extraction = extraction
        self.calls = []

    def extract(self, *, metadata, sources, active_tags):
        self.calls.append(
            {"metadata": metadata, "sources": sources, "active_tags": active_tags}
        )
        return self.extraction


@pytest.fixture()
def fake_llm():
    return lambda extraction: _FakeLlm(extraction)


class _FakeEmb:
    def __init__(self, vector):
        self.vector = vector
        self.texts = []

    def embed_text(self, text):
        self.texts.append(text)
        return list(self.vector)


@pytest.fixture()
def fake_emb():
    return lambda vector: _FakeEmb(vector)


@pytest.fixture()
def db(_schema, m1_dsn):
    conn = psycopg.connect(m1_dsn, autocommit=True)
    register_vector(conn)
    conn.execute(
        "TRUNCATE aod_ai.content_semantic_profile, aod_ai.content_fun_tag, "
        "aod_ai.content_embedding, aod_ai.content_quality_score, aod_ai.fun_tag_dict"
    )
    conn.execute("TRUNCATE public.contents, public.webnovel_contents, public.external_ranking")
    yield conn
    conn.close()
