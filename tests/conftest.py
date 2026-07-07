from __future__ import annotations

import pytest
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
