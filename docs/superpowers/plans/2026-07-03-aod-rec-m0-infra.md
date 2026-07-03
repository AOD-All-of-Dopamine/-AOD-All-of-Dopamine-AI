# M0 — 인프라 (AOD 추천) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (권장) 또는 superpowers:executing-plans 로 태스크 단위 실행. 스텝은 체크박스(`- [ ]`) 문법으로 추적.

**Goal:** AOD 추천 배치의 그린필드 인프라 — Python `aod_ai` 패키지 스캐폴드, pgvector 포함 `aod_ai` 스키마 마이그레이션, 권한 격리 role, OpenAI 호환 LLM/임베딩·Vane 클라이언트, 시드 fun_tag 사전, Vane 도커 — 를 TDD로 구축해 "기동" 상태를 만든다.

**Architecture:** 무거운 모델(Qwen LLM/Embedding)은 배치에서만 관리형 OpenAI 호환 API로 호출. 벡터는 pgvector `vector(1024)` + HNSW. 데이터는 같은 RDS의 `aod_ai` 스키마(쓰기)와 `public`(읽기전용 role)로 격리. M0/M1은 로컬 PC에서 실행.

**Tech Stack:** Python 3.11+, psycopg3, pgvector, openai SDK(base_url), pydantic / pydantic-settings, tenacity, pytest + respx + testcontainers, Docker(Vane).

**공유 계약:** 이 계획의 모든 이름/타입/스키마/env는 `2026-07-03-aod-rec-00-contracts.md` 를 그대로 인용한다. 실행 전 그 문서(특히 §9 통합 보정 규칙)를 먼저 읽을 것.

---

## M0 — 인프라 (그린필드 Python 배치 스캐폴드 + aod_ai 스키마 + 권한격리 role + 클라이언트 + 시드 사전 + Vane)

**Deliverable/Checkpoint:** `-AOD-All-of-Dopamine-AI/` Python 배치 스캐폴드(`aod_ai` 패키지 + config/db/migrate/models/clients/funtag_dict) + `migrations/001..004` + `sql/roles.sql`(§2.2 데이터 접근 격리 — `public` **읽기전용 role** + `aod_ai` 읽기쓰기) + `resources/seed_fun_tags.yaml` + `docker/vane/`(조작 이미지 태그 대신 **고정 상류 ref로 빌드한 Vane** + SearXNG `settings.yml` JSON 활성화); `python -m pytest -q` = **`18 passed`** + `python -m aod_ai.migrate`가 로컬/RDS pg에 `aod_ai` 스키마·HNSW 인덱스 생성 + DBA가 `sql/roles.sql` 적용(role 격리) + 고정한 `VANE_REF` 버전의 Vane 컨테이너 `/api/search` 스모크가 200 & **top-level `sources[]` 반환**(§10 #9 응답 스키마 확인). (모든 이름·타입·스키마·env는 `_contracts.md` §1~§5 그대로 인용.)

---

### Task 1: pyproject.toml + 레포 스캐폴드 + .env.example

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/pyproject.toml`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/.env.example`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/__init__.py`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/clients/__init__.py`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/pipeline/__init__.py`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/.gitignore`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_package.py`

- [ ] Write FAILING test `tests/test_package.py` with real code:
  ```python
  def test_package_importable():
      import aod_ai
      assert aod_ai.__name__ == "aod_ai"
      assert aod_ai.__version__ == "0.1.0"
  ```
- [ ] Run it & expect FAIL (package not installed yet):
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_package.py -q
  ```
  Expected: `ModuleNotFoundError: No module named 'aod_ai'`
- [ ] Minimal implementation. Create `pyproject.toml` (package `aod_ai`, deps per CONTRACTS §1.1):
  ```toml
  [build-system]
  requires = ["setuptools>=68"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "aod_ai"
  version = "0.1.0"
  description = "AOD recommendation batch pipeline (Content Intelligence)"
  requires-python = ">=3.11"
  dependencies = [
    "psycopg[binary]>=3.2",
    "pgvector>=0.3",
    "openai>=1.40",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "httpx>=0.27",
    "tenacity>=8.4",
    "pyyaml>=6",
    "python-dotenv>=1",
  ]

  [project.optional-dependencies]
  dev = ["pytest>=8", "respx", "testcontainers[postgres]"]

  [tool.setuptools.packages.find]
  include = ["aod_ai*"]

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  ```
  Create `aod_ai/__init__.py`:
  ```python
  __version__ = "0.1.0"
  ```
  Create empty `aod_ai/clients/__init__.py` and `aod_ai/pipeline/__init__.py` (0 bytes). Create `.gitignore`:
  ```gitignore
  .venv/
  __pycache__/
  *.pyc
  .env
  .pytest_cache/
  *.egg-info/
  ```
  Create `.env.example` (exact env keys per CONTRACTS §5; `VANE_REF`는 Settings 키가 아니라 compose 빌드용):
  ```dotenv
  # DB (public 읽기전용 role / aod_ai 읽기쓰기 — §2.2)
  AOD_DB_HOST=localhost
  AOD_DB_PORT=5432
  AOD_DB_NAME=aod
  AOD_DB_USER=aod_ai
  AOD_DB_PASSWORD=change-me
  # OpenAI 호환 (관리형 API)
  OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  OPENAI_API_KEY=sk-change-me
  LLM_MODEL=qwen-plus
  EMBEDDING_MODEL=text-embedding-v3
  EMBEDDING_DIM=1024
  # Vane
  VANE_BASE_URL=http://localhost:3000
  # Vane 자가호스팅 빌드(고정 상류 ref; §10 #9에서 /api/search 스키마 검증한 커밋/태그)
  VANE_REF=<pinned-perplexica-commit-or-tag>
  # 파이프라인 튜닝
  VANE_MAX_SOURCES=8
  EXTRACT_MAX_RETRIES=3
  ```
- [ ] Install editable + run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pip install -e ".[dev]" && python -m pytest tests/test_package.py -q
  ```
  Expected: `1 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git init -b main && git add pyproject.toml .env.example .gitignore aod_ai/ tests/test_package.py && git commit -m "chore: aod_ai batch scaffold, pyproject and .env.example" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: aod_ai/config.py Settings (pydantic-settings, CONTRACTS §5 env 키)

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/config.py`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_config.py`

- [ ] Write FAILING test `tests/test_config.py`:
  ```python
  from aod_ai.config import Settings


  def test_settings_reads_env_keys(monkeypatch):
      monkeypatch.setenv("AOD_DB_HOST", "db.local")
      monkeypatch.setenv("AOD_DB_NAME", "aod")
      monkeypatch.setenv("AOD_DB_USER", "ai")
      monkeypatch.setenv("AOD_DB_PASSWORD", "secret")
      monkeypatch.setenv("OPENAI_BASE_URL", "http://llm")
      monkeypatch.setenv("OPENAI_API_KEY", "key")
      monkeypatch.setenv("LLM_MODEL", "qwen-llm")
      monkeypatch.setenv("EMBEDDING_MODEL", "qwen-emb")
      s = Settings(_env_file=None)
      assert s.aod_db_host == "db.local"
      assert s.aod_db_port == 5432
      assert s.aod_db_name == "aod"
      assert s.embedding_dim == 1024
      assert s.vane_base_url == "http://localhost:3000"
      assert s.vane_max_sources == 8
      assert s.extract_max_retries == 3
  ```
- [ ] Run & expect FAIL:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_config.py -q
  ```
  Expected: `ModuleNotFoundError: No module named 'aod_ai.config'`
- [ ] Minimal implementation `aod_ai/config.py`:
  ```python
  from __future__ import annotations

  from pydantic_settings import BaseSettings, SettingsConfigDict


  class Settings(BaseSettings):
      model_config = SettingsConfigDict(
          env_file=".env", env_file_encoding="utf-8", extra="ignore"
      )

      # DB (읽기: public 읽기전용 role, 쓰기: aod_ai)
      aod_db_host: str
      aod_db_port: int = 5432
      aod_db_name: str
      aod_db_user: str
      aod_db_password: str

      # OpenAI 호환
      openai_base_url: str
      openai_api_key: str
      llm_model: str
      embedding_model: str
      embedding_dim: int = 1024

      # Vane
      vane_base_url: str = "http://localhost:3000"

      # 파이프라인 튜닝
      vane_max_sources: int = 8
      extract_max_retries: int = 3
  ```
  > `extra="ignore"` 이므로 `.env`의 compose 전용 키(`VANE_REF`, `EMBEDDING_DIM` 등)는 Settings 로드를 깨지 않는다.
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_config.py -q
  ```
  Expected: `1 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add aod_ai/config.py tests/test_config.py && git commit -m "feat: Settings reading CONTRACTS env keys" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3: aod_ai/db.py 커넥션 헬퍼 + aod_ai/migrate.py 러너 + migrations/001 (aod_ai.schema_migrations 추적)

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/db.py`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/migrate.py`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/migrations/001_schema_pgvector.sql`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/conftest.py`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_migrate.py`

- [ ] Write FAILING test scaffolding. Create `tests/conftest.py` (임시 pg = testcontainers pgvector 이미지, `settings`/`conn` 픽스처):
  ```python
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
  ```
  > `pg_container`는 세션 공유(속도)라 `conn` 픽스처가 마이그레이션을 컨테이너에 영구 기록한다. 이를 전제로 **마이그레이션 적용 검증 테스트(Task 4)는 자기 격리**(선행 `DROP SCHEMA`)하도록 설계한다.
  Create `tests/test_migrate.py`:
  ```python
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
  ```
- [ ] Run & expect FAIL (collection error, `db`/`migrate` 미존재):
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_migrate.py -q
  ```
  Expected: `ModuleNotFoundError: No module named 'aod_ai.db'`
- [ ] Minimal implementation `aod_ai/db.py`:
  ```python
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
  ```
  `aod_ai/migrate.py`:
  ```python
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
  ```
  Create `migrations/001_schema_pgvector.sql` (CONTRACTS §2 001 그대로):
  ```sql
  CREATE SCHEMA IF NOT EXISTS aod_ai;
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE TABLE IF NOT EXISTS aod_ai.schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
  );
  ```
- [ ] Run & expect PASS (Docker daemon 필요):
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_migrate.py -q
  ```
  Expected: `1 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add aod_ai/db.py aod_ai/migrate.py migrations/001_schema_pgvector.sql tests/conftest.py tests/test_migrate.py && git commit -m "feat: db connect helper and ordered sql migrate runner with 001" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 4: migrations 002/003/004 (asset + user/log + HNSW) — 자기 격리 멱등 적용 + 테이블/HNSW 존재 검증

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/migrations/002_asset_tables.sql`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/migrations/003_user_and_log_tables.sql`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/migrations/004_hnsw_indexes.sql`
- Modify/Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_migrate.py`

- [ ] Rewrite `tests/test_migrate.py` to the FAILING (4-migration) expectation. **자기 격리 필수**: 세션 공유 컨테이너에서 `conn` 픽스처를 쓰는 `test_funtag_dict`/`test_roles`가 알파벳 순서상 `test_migrate`보다 먼저 수집·실행되어 4개 버전을 미리 기록하므로, `ALL_VERSIONS` 검증 직전 스키마를 초기화한다:
  ```python
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
  ```
- [ ] Run & expect FAIL (only 001 present):
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_migrate.py -q
  ```
  Expected: `AssertionError: assert ['001_schema_pgvector.sql'] == ['001_schema_pgvector.sql', '002_asset_tables.sql', '003_user_and_log_tables.sql', '004_hnsw_indexes.sql']`
- [ ] Minimal implementation. Create `migrations/002_asset_tables.sql` (CONTRACTS §2 002 그대로):
  ```sql
  CREATE TABLE aod_ai.fun_tag_dict (
    id          bigserial PRIMARY KEY,
    name        text NOT NULL UNIQUE,
    aliases     text[] NOT NULL DEFAULT '{}',
    description text,
    status      text NOT NULL DEFAULT 'active' CHECK (status IN ('active','proposed','rejected')),
    created_at  timestamptz NOT NULL DEFAULT now()
  );

  CREATE TABLE aod_ai.content_semantic_profile (
    content_id         bigint PRIMARY KEY,
    domain             text NOT NULL,
    normalized_summary text,
    profile_text       text NOT NULL,
    evidence           jsonb NOT NULL DEFAULT '[]'::jsonb,
    extraction_quality real  NOT NULL DEFAULT 0,
    source_count       int   NOT NULL DEFAULT 0,
    content_hash       text  NOT NULL,
    processed_at       timestamptz NOT NULL DEFAULT now()
  );

  CREATE TABLE aod_ai.content_fun_tag (
    content_id     bigint NOT NULL,
    tag            text   NOT NULL,
    tag_score      real   NOT NULL,
    tag_confidence real   NOT NULL,
    PRIMARY KEY (content_id, tag)
  );
  CREATE INDEX idx_content_fun_tag_tag ON aod_ai.content_fun_tag (tag);

  CREATE TABLE aod_ai.content_embedding (
    content_id bigint PRIMARY KEY,
    embedding  vector(1024) NOT NULL,
    model      text NOT NULL,
    dim        int  NOT NULL DEFAULT 1024
  );

  CREATE TABLE aod_ai.content_quality_score (
    content_id               bigint PRIMARY KEY,
    bayesian_score           real,
    platform_rank_score      real,
    review_count_score       real,
    recency_score            real,
    quality_popularity_score real,
    computed_at              timestamptz NOT NULL DEFAULT now()
  );
  ```
  Create `migrations/003_user_and_log_tables.sql` (CONTRACTS §2 003 그대로):
  ```sql
  CREATE TABLE aod_ai.user_profile_cache (
    user_id          bigint PRIMARY KEY,
    fun_tag_profile  jsonb NOT NULL DEFAULT '{}'::jsonb,
    negative_fun_tag jsonb NOT NULL DEFAULT '{}'::jsonb,
    profile_vector   vector(1024),
    positive_count   int NOT NULL DEFAULT 0,
    updated_at       timestamptz NOT NULL DEFAULT now()
  );

  CREATE TABLE aod_ai.rec_impression (
    id                  bigserial PRIMARY KEY,
    request_id          uuid   NOT NULL,
    user_id             bigint,
    location            text   NOT NULL,
    selected_content_id bigint,
    content_id          bigint NOT NULL,
    candidate_source    text,
    rank_position       int,
    score_breakdown     jsonb,
    served_at           timestamptz NOT NULL DEFAULT now()
  );

  CREATE TABLE aod_ai.rec_event (
    id         bigserial PRIMARY KEY,
    request_id uuid,
    user_id    bigint,
    content_id bigint NOT NULL,
    event_type text   NOT NULL,
    value      real,
    created_at timestamptz NOT NULL DEFAULT now()
  );
  ```
  Create `migrations/004_hnsw_indexes.sql` (CONTRACTS §2 004 그대로):
  ```sql
  CREATE INDEX idx_content_embedding_hnsw ON aod_ai.content_embedding
    USING hnsw (embedding vector_cosine_ops);
  CREATE INDEX idx_user_profile_vector_hnsw ON aod_ai.user_profile_cache
    USING hnsw (profile_vector vector_cosine_ops);
  ```
  > **결정 (§2 004 '데이터 적재 후 실행 권장' 오버라이드):** M0은 **의도적으로 빈 테이블 위에 HNSW를 생성**한다. 근거: (1) M0 체크포인트가 인덱스 존재 자체를 산출물로 요구(§9 M0 행), (2) pgvector HNSW는 빈 테이블에 만들어도 유효하며 이후 INSERT마다 점증 구축된다. §2 주석의 "적재 후" 권고는 **대량 벌크 적재 시 빌드시간 최적화** 목적이므로 M1 대량 재적재 전 `REINDEX`로 재빌드하면 된다(자동 러너 001→004 순차 적용은 유지). 이 결정으로 004를 auto-runner에서 분리하지 않는다.
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_migrate.py -q
  ```
  Expected: `1 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add migrations/002_asset_tables.sql migrations/003_user_and_log_tables.sql migrations/004_hnsw_indexes.sql tests/test_migrate.py && git commit -m "feat: aod_ai asset, user/log, and HNSW migrations 002..004" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 5: sql/roles.sql — `public` 읽기전용 role + `aod_ai` 읽기쓰기 grant (DBA/슈퍼유저 실행, 마이그레이션 러너와 분리)

> §2.2/§1 '데이터 접근' + §9 M0 행이 요구하는 **권한 격리**. role 생성은 `CREATEROLE`/슈퍼유저가 필요해 `aod_ai` 계정으로 도는 `migrate.py` 러너로는 못 돌린다. 따라서 계약(§1.2·§6이 grant를 "마이그레이션/DBA 태스크"로 위임) 대로 `migrations/`(자동 러너 대상) **바깥**의 `sql/roles.sql` 로 분리하고, DBA/psql로 적용한다.

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/sql/roles.sql`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_roles.py`

- [ ] Write FAILING test `tests/test_roles.py` (uses `conn` 픽스처 → aod_ai 스키마/테이블 존재 상태; 테스트 컨테이너의 로그인 계정을 배치 AI 계정으로 취급):
  ```python
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
  ```
- [ ] Run & expect FAIL (파일 없음):
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_roles.py -q
  ```
  Expected: `FileNotFoundError: [Errno 2] No such file or directory: ...sql/roles.sql`
- [ ] Minimal implementation `sql/roles.sql` (멱등; psql `-v ai_user=<배치계정>` 로 주입):
  ```sql
  -- 실행: DBA/슈퍼유저 (CREATEROLE 필요) — aod_ai 마이그레이션 러너(aod_ai 계정)와 별개.
  --   psql "postgresql://<dba>@<host>:<port>/<db>" -v ai_user=aod_ai -f sql/roles.sql
  -- 근거: spec §2.2 / §1 '데이터 접근' — 배치 AI 계정 = public 읽기전용 + aod_ai 읽기쓰기.

  -- 1) public 읽기전용 role (멱등)
  DO $$
  BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aod_public_ro') THEN
      CREATE ROLE aod_public_ro NOLOGIN;
    END IF;
  END
  $$;

  GRANT USAGE ON SCHEMA public TO aod_public_ro;
  GRANT SELECT ON ALL TABLES IN SCHEMA public TO aod_public_ro;
  ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO aod_public_ro;

  -- 2) 배치 AI 로그인 계정: public 읽기전용 role 부여 (백엔드 테이블 실수 수정 차단)
  GRANT aod_public_ro TO :"ai_user";

  -- 3) 배치 AI 로그인 계정: aod_ai 읽기쓰기
  GRANT USAGE, CREATE ON SCHEMA aod_ai TO :"ai_user";
  GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA aod_ai TO :"ai_user";
  GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA aod_ai TO :"ai_user";
  ALTER DEFAULT PRIVILEGES IN SCHEMA aod_ai
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"ai_user";
  ALTER DEFAULT PRIVILEGES IN SCHEMA aod_ai
    GRANT USAGE, SELECT ON SEQUENCES TO :"ai_user";
  ```
  > 테스트 컨테이너의 로그인 계정은 슈퍼유저라 aod_ai 권한이 이미 암묵적이므로 grant 자체는 무해·멱등하게 검증된다. **RDS 운영에선 배치 계정이 비-슈퍼유저**이므로 위 grant가 실제로 접근을 결정한다.
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_roles.py -q
  ```
  Expected: `1 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add sql/roles.sql tests/test_roles.py && git commit -m "feat: sql/roles.sql public read-only role and aod_ai read-write grants (data isolation)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 6: aod_ai/models.py (CONTRACTS §3 Pydantic 모델 + §3.1 content_hash)

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/models.py`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_models.py`

- [ ] Write FAILING test `tests/test_models.py`:
  ```python
  from aod_ai.models import (
      Extraction,
      FunTagItem,
      QualityScore,
      ReviewSource,
      SelectedTarget,
      content_hash,
  )


  def test_extraction_round_trips_from_json():
      raw = (
          '{"fun_tags":[{"tag":"회귀","tag_score":0.9,"tag_confidence":0.8,'
          '"evidence":"회귀물","is_new":false}],'
          '"normalized_summary":"요약","profile_text":"프로파일","extraction_quality":0.7}'
      )
      e = Extraction.model_validate_json(raw)
      assert isinstance(e.fun_tags[0], FunTagItem)
      assert e.fun_tags[0].tag == "회귀"
      assert e.extraction_quality == 0.7


  def test_selected_target_and_quality_fields():
      t = SelectedTarget(
          content_id=1,
          domain="WEBNOVEL",
          master_title="t",
          original_title=None,
          synopsis=None,
          genres=["판타지"],
          content_hash="h",
      )
      assert t.original_title is None
      q = QualityScore(
          bayesian_score=1.0,
          platform_rank_score=0.5,
          review_count_score=0.2,
          recency_score=0.1,
          quality_popularity_score=0.9,
      )
      assert q.quality_popularity_score == 0.9
      assert ReviewSource(content="c", url="http://u").url == "http://u"


  def test_content_hash_is_stable_and_genre_order_invariant():
      a = content_hash("주술회전", None, "줄거리", ["액션", "판타지"])
      b = content_hash("주술회전", None, "줄거리", ["판타지", "액션"])
      assert a == b
      assert len(a) == 64
      assert a != content_hash("주술회전", None, "다른줄거리", ["액션", "판타지"])
  ```
- [ ] Run & expect FAIL:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_models.py -q
  ```
  Expected: `ModuleNotFoundError: No module named 'aod_ai.models'`
- [ ] Minimal implementation `aod_ai/models.py` (필드명 §3 고정, 해시 §3.1 공식 그대로):
  ```python
  from __future__ import annotations

  import hashlib

  from pydantic import BaseModel


  class FunTagItem(BaseModel):
      tag: str
      tag_score: float
      tag_confidence: float
      evidence: str
      is_new: bool


  class Extraction(BaseModel):
      fun_tags: list[FunTagItem]
      normalized_summary: str
      profile_text: str
      extraction_quality: float


  class ReviewSource(BaseModel):
      content: str
      url: str


  class SelectedTarget(BaseModel):
      content_id: int
      domain: str
      master_title: str
      original_title: str | None
      synopsis: str | None
      genres: list[str]
      content_hash: str


  class QualityScore(BaseModel):
      bayesian_score: float
      platform_rank_score: float
      review_count_score: float
      recency_score: float
      quality_popularity_score: float


  def content_hash(
      master_title: str,
      original_title: str | None,
      synopsis: str | None,
      genres: list[str],
  ) -> str:
      raw = f"{master_title}|{original_title}|{synopsis}|{sorted(genres)}"
      return hashlib.sha256(raw.encode("utf-8")).hexdigest()
  ```
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_models.py -q
  ```
  Expected: `3 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add aod_ai/models.py tests/test_models.py && git commit -m "feat: pydantic asset models and content_hash helper" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 7: aod_ai/clients/llm.py — LlmClient.extract (§4 고정 시그니처 + json_object + 리터럴 "json" + Pydantic 검증 + tenacity 재시도)

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/clients/llm.py`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_clients_llm.py`

- [ ] Write FAILING test `tests/test_clients_llm.py` (respx mock, no real network; 생성자는 §4 고정 `(base_url, api_key, model)` — 재시도 횟수는 EXTRACT_MAX_RETRIES config에서):
  ```python
  import json

  import respx
  from httpx import Response

  from aod_ai.clients.llm import LlmClient
  from aod_ai.models import Extraction, ReviewSource, SelectedTarget

  _GOOD = {
      "fun_tags": [
          {
              "tag": "회귀",
              "tag_score": 0.9,
              "tag_confidence": 0.8,
              "evidence": "회귀물",
              "is_new": False,
          }
      ],
      "normalized_summary": "요약",
      "profile_text": "프로파일",
      "extraction_quality": 0.7,
  }


  def _target():
      return SelectedTarget(
          content_id=1,
          domain="WEBNOVEL",
          master_title="회귀한 천재",
          original_title=None,
          synopsis="s",
          genres=["판타지"],
          content_hash="h",
      )


  def _chat_response(content: str) -> Response:
      return Response(
          200,
          json={
              "id": "x",
              "object": "chat.completion",
              "created": 0,
              "model": "m",
              "choices": [
                  {
                      "index": 0,
                      "finish_reason": "stop",
                      "message": {"role": "assistant", "content": content},
                  }
              ],
          },
      )


  @respx.mock
  def test_extract_returns_validated_extraction_and_sends_json_and_schema():
      route = respx.post("http://llm.test/chat/completions").mock(
          return_value=_chat_response(json.dumps(_GOOD))
      )
      client = LlmClient(base_url="http://llm.test", api_key="k", model="m")
      result = client.extract(
          metadata=_target(),
          sources=[ReviewSource(content="재밌음", url="http://u")],
          active_tags=["회귀", "먼치킨"],
      )
      assert isinstance(result, Extraction)
      assert result.fun_tags[0].tag == "회귀"
      body = json.loads(route.calls.last.request.content)
      assert body["response_format"] == {"type": "json_object"}
      prompt = body["messages"][-1]["content"]
      assert "json" in prompt
      assert "fun_tags" in prompt


  @respx.mock
  def test_extract_retries_on_invalid_json_then_succeeds(monkeypatch):
      monkeypatch.setenv("EXTRACT_MAX_RETRIES", "3")  # 재시도는 config §5에서
      route = respx.post("http://llm.test/chat/completions").mock(
          side_effect=[
              _chat_response('{"not":"an extraction"}'),
              _chat_response(json.dumps(_GOOD)),
          ]
      )
      client = LlmClient(base_url="http://llm.test", api_key="k", model="m")
      result = client.extract(metadata=_target(), sources=[], active_tags=["회귀"])
      assert result.extraction_quality == 0.7
      assert route.call_count == 2
  ```
- [ ] Run & expect FAIL:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_clients_llm.py -q
  ```
  Expected: `ModuleNotFoundError: No module named 'aod_ai.clients.llm'`
- [ ] Minimal implementation `aod_ai/clients/llm.py` (시그니처 CONTRACTS §4 **그대로** — 생성자에 재시도 인자 없음; EXTRACT_MAX_RETRIES는 내부에서 읽음):
  ```python
  from __future__ import annotations

  import json
  import os

  from openai import OpenAI
  from tenacity import retry, stop_after_attempt, wait_exponential

  from aod_ai.models import Extraction, ReviewSource, SelectedTarget

  _SCHEMA_HINT = json.dumps(
      {
          "fun_tags": [
              {
                  "tag": "str",
                  "tag_score": 0.0,
                  "tag_confidence": 0.0,
                  "evidence": "str",
                  "is_new": False,
              }
          ],
          "normalized_summary": "str",
          "profile_text": "str",
          "extraction_quality": 0.0,
      },
      ensure_ascii=False,
  )


  class LlmClient:
      def __init__(self, base_url: str, api_key: str, model: str):
          self._client = OpenAI(base_url=base_url, api_key=api_key)
          self._model = model

      def extract(
          self,
          *,
          metadata: SelectedTarget,
          sources: list[ReviewSource],
          active_tags: list[str],
      ) -> Extraction:
          prompt = self._build_prompt(metadata, sources, active_tags)
          max_retries = int(os.getenv("EXTRACT_MAX_RETRIES", "3"))  # config §5

          @retry(
              stop=stop_after_attempt(max_retries),
              wait=wait_exponential(multiplier=1, max=10),
              reraise=True,
          )
          def _call() -> Extraction:
              resp = self._client.chat.completions.create(
                  model=self._model,
                  response_format={"type": "json_object"},
                  messages=[
                      {"role": "system", "content": "You extract structured fun_tag JSON."},
                      {"role": "user", "content": prompt},
                  ],
              )
              return Extraction.model_validate_json(resp.choices[0].message.content)

          return _call()

      def _build_prompt(
          self,
          metadata: SelectedTarget,
          sources: list[ReviewSource],
          active_tags: list[str],
      ) -> str:
          evidence = "\n".join(f"- {s.content} ({s.url})" for s in sources)
          return (
              "Return only a JSON object matching this schema (respond in json):\n"
              f"{_SCHEMA_HINT}\n\n"
              f"Active fun_tag dictionary: {active_tags}\n"
              f"Title: {metadata.master_title}\n"
              f"Domain: {metadata.domain}\n"
              f"Synopsis: {metadata.synopsis}\n"
              f"Genres: {metadata.genres}\n"
              f"Review evidence:\n{evidence}\n"
          )
  ```
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_clients_llm.py -q
  ```
  Expected: `2 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add aod_ai/clients/llm.py tests/test_clients_llm.py && git commit -m "feat: LlmClient.extract with fixed signature, json_object, pydantic validate, config-driven tenacity retry" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 8: aod_ai/clients/embedding.py — EmbeddingClient.embed_text (dimensions=1024, len assert)

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/clients/embedding.py`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_clients_embedding.py`

- [ ] Write FAILING test `tests/test_clients_embedding.py` (respx mock):
  ```python
  import pytest
  import respx
  from httpx import Response

  from aod_ai.clients.embedding import EmbeddingClient


  def _embeddings_response(values: list[float]) -> Response:
      return Response(
          200,
          json={
              "object": "list",
              "model": "m",
              "data": [{"object": "embedding", "index": 0, "embedding": values}],
              "usage": {"prompt_tokens": 1, "total_tokens": 1},
          },
      )


  @respx.mock
  def test_embed_text_returns_dim_length_vector_and_sends_dimensions():
      route = respx.post("http://emb.test/embeddings").mock(
          return_value=_embeddings_response([0.0] * 1024)
      )
      client = EmbeddingClient(base_url="http://emb.test", api_key="k", model="m")
      vec = client.embed_text("프로파일 텍스트")
      assert len(vec) == 1024
      import json

      body = json.loads(route.calls.last.request.content)
      assert body["dimensions"] == 1024


  @respx.mock
  def test_embed_text_asserts_on_wrong_length():
      respx.post("http://emb.test/embeddings").mock(
          return_value=_embeddings_response([0.0] * 512)
      )
      client = EmbeddingClient(base_url="http://emb.test", api_key="k", model="m")
      with pytest.raises(AssertionError):
          client.embed_text("x")
  ```
- [ ] Run & expect FAIL:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_clients_embedding.py -q
  ```
  Expected: `ModuleNotFoundError: No module named 'aod_ai.clients.embedding'`
- [ ] Minimal implementation `aod_ai/clients/embedding.py` (시그니처 CONTRACTS §4 그대로 — `dim: int = 1024` 는 §4 고정 인자):
  ```python
  from __future__ import annotations

  from openai import OpenAI


  class EmbeddingClient:
      def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1024):
          self._client = OpenAI(base_url=base_url, api_key=api_key)
          self._model = model
          self.dim = dim

      def embed_text(self, text: str) -> list[float]:
          resp = self._client.embeddings.create(
              model=self._model, input=text, dimensions=self.dim
          )
          vector = list(resp.data[0].embedding)
          assert len(vector) == self.dim, f"expected {self.dim} dims, got {len(vector)}"
          return vector
  ```
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_clients_embedding.py -q
  ```
  Expected: `2 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add aod_ai/clients/embedding.py tests/test_clients_embedding.py && git commit -m "feat: EmbeddingClient.embed_text with dimensions=1024 and length assert" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 9: aod_ai/clients/vane.py — VaneClient.search (§4 고정 시그니처 `__init__(base_url)`; POST /api/search, dedupe(url) + 상한 N개(config) + 길이예산)

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/clients/vane.py`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_clients_vane.py`

- [ ] Write FAILING test `tests/test_clients_vane.py` (respx mock; 생성자는 §4 고정 `(base_url)` — 상한은 config(VANE_MAX_SOURCES), 길이예산은 클라이언트 상수):
  ```python
  import json

  import respx
  from httpx import Response

  from aod_ai.clients.vane import VaneClient, _SOURCE_CHAR_BUDGET


  @respx.mock
  def test_search_dedupes_by_url_caps_and_posts_stream_false(monkeypatch):
      monkeypatch.setenv("VANE_MAX_SOURCES", "2")  # 상한 N개 = config §5
      route = respx.post("http://vane.test/api/search").mock(
          return_value=Response(
              200,
              json={
                  "message": "합성답변",
                  "sources": [
                      {"content": "리뷰1", "url": "http://a"},
                      {"content": "리뷰1-dup", "url": "http://a"},
                      {"content": "리뷰2", "url": "http://b"},
                      {"content": "리뷰3", "url": "http://c"},
                  ],
              },
          )
      )
      client = VaneClient(base_url="http://vane.test")
      out = client.search(
          query="\"오징어 게임\" TV 리뷰 후기 재미",
          sources=["web", "discussions"],
          system_instructions="독자 반응·재미요소 위주",
      )
      assert [s.url for s in out] == ["http://a", "http://b"]
      body = json.loads(route.calls.last.request.content)
      assert body["stream"] is False
      assert body["sources"] == ["web", "discussions"]
      assert body["systemInstructions"] == "독자 반응·재미요소 위주"


  @respx.mock
  def test_search_parses_perplexica_pagecontent_shape(monkeypatch):
      monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)  # 기본 8
      respx.post("http://vane.test/api/search").mock(
          return_value=Response(
              200,
              json={
                  "sources": [
                      {"pageContent": "본문", "metadata": {"url": "http://x"}},
                  ]
              },
          )
      )
      client = VaneClient(base_url="http://vane.test")
      out = client.search(query="q", sources=["web"])
      assert out[0].content == "본문"
      assert out[0].url == "http://x"


  @respx.mock
  def test_search_applies_length_budget_trims_oversized_content(monkeypatch):
      monkeypatch.delenv("VANE_MAX_SOURCES", raising=False)
      long_content = "가" * (_SOURCE_CHAR_BUDGET + 500)
      respx.post("http://vane.test/api/search").mock(
          return_value=Response(
              200,
              json={"sources": [{"content": long_content, "url": "http://a"}]},
          )
      )
      client = VaneClient(base_url="http://vane.test")
      out = client.search(query="q", sources=["web"])
      assert len(out[0].content) == _SOURCE_CHAR_BUDGET
      assert out[0].content == long_content[:_SOURCE_CHAR_BUDGET]
  ```
- [ ] Run & expect FAIL:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_clients_vane.py -q
  ```
  Expected: `ModuleNotFoundError: No module named 'aod_ai.clients.vane'`
- [ ] Minimal implementation `aod_ai/clients/vane.py` (시그니처 CONTRACTS §4 **그대로** `__init__(self, base_url: str)`; §4 주석 3동작 = 상한 N개(config) + dedupe(url) + 길이예산; §4.1 sources[].content 우선, Perplexica pageContent/metadata.url fallback):
  ```python
  from __future__ import annotations

  import os

  import httpx

  from aod_ai.models import ReviewSource

  _DEFAULT_MAX_SOURCES = 8          # VANE_MAX_SOURCES 미설정 시 기본 (config §5)
  _SOURCE_CHAR_BUDGET = 2000        # 길이예산: 소스당 content 상한 (evidence 뭉갬 방지)
  _TIMEOUT = 60.0


  class VaneClient:
      def __init__(self, base_url: str):
          self._base_url = base_url.rstrip("/")
          self._client = httpx.Client(timeout=_TIMEOUT)

      def search(
          self,
          *,
          query: str,
          sources: list[str],
          system_instructions: str | None = None,
      ) -> list[ReviewSource]:
          max_sources = int(os.getenv("VANE_MAX_SOURCES", str(_DEFAULT_MAX_SOURCES)))

          body: dict = {"query": query, "sources": sources, "stream": False}
          if system_instructions is not None:
              body["systemInstructions"] = system_instructions

          resp = self._client.post(f"{self._base_url}/api/search", json=body)
          resp.raise_for_status()
          raw = resp.json().get("sources", []) or []

          seen: set[str] = set()
          out: list[ReviewSource] = []
          for item in raw:
              url = item.get("url") or (item.get("metadata") or {}).get("url")
              content = item.get("content") or item.get("pageContent")
              if not url or not content or url in seen:  # dedupe(url)
                  continue
              seen.add(url)
              content = content[:_SOURCE_CHAR_BUDGET]  # 길이예산 적용
              out.append(ReviewSource(content=content, url=url))
              if len(out) >= max_sources:  # 상한 N개(config)
                  break
          return out
  ```
  > M1 배치 `collect_reviews`는 §1.1/§4 대로 `VaneClient(base_url)` 로 생성하고, 상한은 `.env`의 `VANE_MAX_SOURCES`(config §5)가 그대로 반영된다 — 생성자 인자로 캡을 넘기지 않으므로 계약 시그니처와 일치.
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_clients_vane.py -q
  ```
  Expected: `3 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add aod_ai/clients/vane.py tests/test_clients_vane.py && git commit -m "feat: VaneClient.search POST /api/search with fixed ctor, config cap, dedupe, length budget" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 10: resources/seed_fun_tags.yaml (시드 사전) + aod_ai/funtag_dict.py (upsert status=active, active 목록 조회)

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/resources/seed_fun_tags.yaml`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/aod_ai/funtag_dict.py`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_funtag_dict.py`

- [ ] Write FAILING test `tests/test_funtag_dict.py` (uses `conn` 픽스처 → `fun_tag_dict` 테이블; 계약/스펙이 고정한 건 **30~60개 개수 + alias/name shape**뿐 — 구체 태그명은 원본 기획 `aod_reccomendation.md` 참고 시드일 뿐 계약 고정 아님):
  ```python
  from aod_ai.funtag_dict import list_active_tags, load_seed_tags, sync_seed_tags


  def test_seed_yaml_has_30_to_60_tags_with_alias_shape():
      # spec §1/§10 #6 이 고정하는 것은 '시드 30~60개'뿐. 개수 + 스키마 형태만 검증.
      tags = load_seed_tags()
      assert 30 <= len(tags) <= 60
      names = {t["name"] for t in tags}
      assert len(names) == len(tags)  # 이름 유니크
      for t in tags:
          assert isinstance(t["name"], str) and t["name"]
          assert isinstance(t.get("aliases", []), list)


  def test_sync_upserts_active_and_is_idempotent(conn):
      tags = load_seed_tags()
      inserted = sync_seed_tags(conn, tags)
      assert inserted == len(tags)
      sync_seed_tags(conn, tags)  # 재실행 멱등

      active = list_active_tags(conn)
      assert len(active) == len(tags)
      assert "회귀" in active

      statuses = {
          r[0]
          for r in conn.execute("SELECT DISTINCT status FROM aod_ai.fun_tag_dict").fetchall()
      }
      assert statuses == {"active"}
  ```
- [ ] Run & expect FAIL:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_funtag_dict.py -q
  ```
  Expected: `ModuleNotFoundError: No module named 'aod_ai.funtag_dict'`
- [ ] Minimal implementation. Create `resources/seed_fun_tags.yaml` (45개; 태그 내용은 **원본 기획 `aod_reccomendation.md`의 fun_tag 예시**를 기반으로 확장한 시드):
  ```yaml
  # 시드 fun_tag 사전 (30~60개, spec §1/§9 M0). 태그 내용 출처: 원본 기획 aod_reccomendation.md.
  # status 는 적재 시 전부 active (신규 제안 태그는 배치 런타임에 proposed 로 별도 적재 — spec §4.2).
  tags:
    - { name: "회귀", aliases: ["회귀물"], description: "죽거나 실패한 뒤 과거 시점으로 되돌아가 다시 사는 전개" }
    - { name: "환생", aliases: [], description: "죽은 뒤 다른 존재/세계로 다시 태어나는 전개" }
    - { name: "빙의", aliases: [], description: "다른 인물의 몸/작품 속으로 들어가는 전개" }
    - { name: "회빙환", aliases: ["회귀 빙의 환생"], description: "회귀·빙의·환생 클리셰 묶음" }
    - { name: "먼치킨", aliases: ["압도적 강함"], description: "압도적으로 강한 주인공" }
    - { name: "힘을 숨김", aliases: ["능력 숨김", "숨긴 실력"], description: "진짜 실력을 숨기고 다니는 주인공" }
    - { name: "사이다", aliases: ["통쾌"], description: "막힘없이 통쾌하게 문제를 해결하는 전개" }
    - { name: "고구마", aliases: ["답답 전개"], description: "답답하고 억눌리는 전개" }
    - { name: "피폐", aliases: ["피폐물"], description: "정신적·감정적으로 극한까지 몰아붙이는 어두운 전개" }
    - { name: "혐관", aliases: ["혐관물"], description: "서로 미워하면서 얽히는 관계" }
    - { name: "구원 서사", aliases: ["구원물"], description: "망가진 인물을 구원해가는 서사" }
    - { name: "잔혹한 동화", aliases: [], description: "동화적 외피 속 잔혹한 이야기" }
    - { name: "복수극", aliases: ["복수물"], description: "억울함을 갚아나가는 복수 중심 전개" }
    - { name: "성장 서사", aliases: ["성장물"], description: "약자가 성장해가는 이야기" }
    - { name: "학원물", aliases: ["아카데미"], description: "학교/아카데미 배경 성장·경쟁" }
    - { name: "헌터물", aliases: ["게이트", "던전"], description: "게이트·던전·헌터 세계관 액션" }
    - { name: "정통 판타지", aliases: ["하이 판타지"], description: "정통 검과 마법 판타지" }
    - { name: "현대 판타지", aliases: ["현판"], description: "현대 배경에 초자연 요소" }
    - { name: "무협", aliases: [], description: "무림·강호 세계관" }
    - { name: "로맨스 판타지", aliases: ["로판"], description: "판타지 세계관 로맨스" }
    - { name: "계약 관계", aliases: ["계약 연애"], description: "계약으로 시작되는 관계" }
    - { name: "츤데레", aliases: [], description: "겉은 차갑고 속은 다정한 캐릭터성" }
    - { name: "다정한 남주", aliases: ["순애"], description: "일편단심 다정한 남자주인공" }
    - { name: "집착", aliases: ["집착남", "집착녀"], description: "상대에게 강하게 집착하는 관계성" }
    - { name: "삼각관계", aliases: [], description: "세 인물 사이의 감정 갈등" }
    - { name: "반전", aliases: ["반전 서사"], description: "예상을 뒤엎는 전개/떡밥 회수" }
    - { name: "다크 히어로", aliases: [], description: "어두운 방식으로 정의를 실현하는 주인공" }
    - { name: "안티히어로", aliases: [], description: "선악이 모호한 주인공" }
    - { name: "정치 암투", aliases: ["궁중암투"], description: "권력·정치 중심의 두뇌 싸움" }
    - { name: "두뇌 싸움", aliases: ["지능전"], description: "머리싸움·전략 중심 전개" }
    - { name: "하렘", aliases: [], description: "다수의 이성이 주인공에게 호감" }
    - { name: "역하렘", aliases: [], description: "다수의 남성이 여주인공에게 호감" }
    - { name: "힐링", aliases: ["힐링물"], description: "따뜻하고 편안한 정서" }
    - { name: "일상", aliases: ["일상물"], description: "잔잔한 일상 중심 전개" }
    - { name: "코미디", aliases: ["개그"], description: "웃음 중심 전개" }
    - { name: "하드고어", aliases: ["고어", "잔혹"], description: "잔혹·유혈 묘사 강한 전개" }
    - { name: "감정선 섬세", aliases: ["섬세한 감정선"], description: "인물 감정 묘사가 섬세한 작품" }
    - { name: "신파", aliases: ["최루"], description: "눈물을 자극하는 감정 과잉 전개" }
    - { name: "데스게임", aliases: ["생존게임"], description: "목숨을 건 게임 서바이벌" }
    - { name: "생존물", aliases: ["서바이벌"], description: "극한 환경에서 살아남기" }
    - { name: "느와르", aliases: ["누아르"], description: "범죄·조직 중심의 어두운 분위기" }
    - { name: "성좌물", aliases: ["성좌"], description: "성좌·후원자 시스템 세계관" }
    - { name: "게임판타지", aliases: ["가상현실", "VRMMO"], description: "게임 시스템 기반 판타지" }
    - { name: "빌런 서사", aliases: ["악역", "악녀"], description: "악역·빌런 시점의 이야기" }
    - { name: "세계관 몰입", aliases: ["세계관"], description: "촘촘한 설정으로 몰입시키는 세계관" }
  ```
  Create `aod_ai/funtag_dict.py`:
  ```python
  from __future__ import annotations

  from pathlib import Path

  import psycopg
  import yaml

  SEED_PATH = Path(__file__).resolve().parent.parent / "resources" / "seed_fun_tags.yaml"


  def load_seed_tags(path: Path = SEED_PATH) -> list[dict]:
      data = yaml.safe_load(path.read_text(encoding="utf-8"))
      return data["tags"]


  def sync_seed_tags(conn: psycopg.Connection, tags: list[dict]) -> int:
      count = 0
      for t in tags:
          conn.execute(
              """
              INSERT INTO aod_ai.fun_tag_dict (name, aliases, description, status)
              VALUES (%s, %s, %s, 'active')
              ON CONFLICT (name) DO UPDATE
                SET aliases = EXCLUDED.aliases,
                    description = EXCLUDED.description,
                    status = 'active'
              """,
              (t["name"], t.get("aliases", []), t.get("description")),
          )
          count += 1
      return count


  def list_active_tags(conn: psycopg.Connection) -> list[str]:
      rows = conn.execute(
          "SELECT name FROM aod_ai.fun_tag_dict WHERE status = 'active' ORDER BY name"
      ).fetchall()
      return [r[0] for r in rows]
  ```
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_funtag_dict.py -q
  ```
  Expected: `2 passed`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add resources/seed_fun_tags.yaml aod_ai/funtag_dict.py tests/test_funtag_dict.py && git commit -m "feat: seed fun_tag dictionary yaml and active upsert loader" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 11: docker/vane/ (고정 상류 ref 빌드 Vane + SearXNG JSON settings.yml) + /api/search 스모크(§10 #9)

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/docker/vane/docker-compose.yml`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/docker/vane/searxng/settings.yml`
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/docker/vane/config.toml`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/tests/test_docker_vane.py`

- [ ] Write FAILING test `tests/test_docker_vane.py` — SearXNG JSON은 **실제 메커니즘(마운트된 settings.yml `search.formats`)** 으로 검증(가짜 env 플래그 금지):
  ```python
  from pathlib import Path

  import yaml

  VANE_DIR = Path(__file__).resolve().parent.parent / "docker" / "vane"
  COMPOSE = VANE_DIR / "docker-compose.yml"
  SEARXNG_SETTINGS = VANE_DIR / "searxng" / "settings.yml"


  def test_vane_compose_declares_services_and_maps_port_3000():
      data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
      services = data["services"]
      assert "vane" in services
      assert "searxng" in services
      vane = services["vane"]
      # 조작 이미지 태그가 아니라 고정 상류 ref 로 빌드
      assert "build" in vane
      assert "${VANE_REF}" in yaml.safe_dump(vane["build"])
      assert any(str(p) == "3000:3000" for p in vane["ports"])
      # SearXNG JSON 은 env 플래그가 아니라 마운트된 settings.yml 로 주입
      mounts = [str(v) for v in services["searxng"].get("volumes", [])]
      assert any("./searxng:/etc/searxng" in m for m in mounts)


  def test_searxng_settings_yaml_enables_json_format():
      settings = yaml.safe_load(SEARXNG_SETTINGS.read_text(encoding="utf-8"))
      assert "json" in settings["search"]["formats"]  # spec §4.1
  ```
- [ ] Run & expect FAIL (파일 없음):
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_docker_vane.py -q
  ```
  Expected: `FileNotFoundError: [Errno 2] No such file or directory: ...docker/vane/docker-compose.yml`
- [ ] Minimal implementation. Create `docker/vane/docker-compose.yml` (Vane = Perplexica 자가호스팅을 **고정 상류 ref(`VANE_REF`)로 빌드** — 조작 이미지 태그 배제; SearXNG는 별도 서비스 + JSON settings.yml 마운트):
  ```yaml
  services:
    searxng:
      image: docker.io/searxng/searxng:latest   # 운영 시 dated 릴리스 태그로 고정
      container_name: aod-searxng
      volumes:
        - ./searxng:/etc/searxng:rw
      environment:
        SEARXNG_BASE_URL: "http://localhost:8080/"
      restart: unless-stopped

    vane:
      # Vane = Perplexica 자가호스팅. 조작된 이미지 태그 대신 상류를 '고정 ref'로 빌드한다.
      # VANE_REF = §10 #9에서 /api/search 응답 스키마를 실제 검증한 커밋/태그.
      build:
        context: "https://github.com/ItzCrazyKns/Perplexica.git#${VANE_REF}"
      container_name: aod-vane
      depends_on:
        - searxng
      ports:
        - "3000:3000"
      volumes:
        # 관리형 API(LLM/임베딩) + SearXNG 연결. 키 구조는 VANE_REF 버전의
        # sample.config.toml 기준으로 §10 #9에서 확정.
        - ./config.toml:/home/perplexica/config.toml:ro
        - vane-data:/home/perplexica/data
      restart: unless-stopped

  volumes:
    vane-data:
  ```
  Create `docker/vane/searxng/settings.yml` (SearXNG **실제** JSON 활성화 메커니즘 — spec §4.1 `search.formats` 에 `json`):
  ```yaml
  # SearXNG JSON 출력 활성화 — spec §4.1: search.formats 에 json 포함.
  use_default_settings: true
  server:
    secret_key: "change-me-please"   # 운영에선 강한 랜덤값으로 교체
  search:
    formats:
      - html
      - json
  ```
  Create `docker/vane/config.toml` (Vane 관리형 API 연결 **초안** — 최종 키 구조는 고정 `VANE_REF` 버전 `sample.config.toml` 기준으로 §10 #9에서 확정, 여기서 사실로 단정하지 않음):
  ```toml
  # Vane(Perplexica) 관리형 API 연결 초안. 아래 키 구조는 §10 #9에서
  # 고정한 VANE_REF 버전의 sample.config.toml 과 대조해 확정할 것.
  [GENERAL]
  SIMILARITY_MEASURE = "cosine"

  [MODELS.CUSTOM_OPENAI]
  API_KEY = "sk-REPLACE"          # = .env OPENAI_API_KEY
  API_URL = "https://REPLACE/v1"  # = .env OPENAI_BASE_URL
  MODEL_NAME = "qwen-plus"        # = .env LLM_MODEL

  [API_ENDPOINTS]
  SEARXNG = "http://searxng:8080"
  ```
- [ ] Run & expect PASS:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest tests/test_docker_vane.py -q
  ```
  Expected: `2 passed`
- [ ] **§10 #9 스모크 체크포인트 (수동, `.env`에 검증된 `VANE_REF` 고정 후)** — 고정한 Vane 버전을 빌드·기동하고 `/api/search`가 200 + **top-level `sources[]` 실제 배열**을 반환하며, 각 원소가 `VaneClient.search` 파서(`content`|`pageContent`, `url`|`metadata.url`)와 일치하는지 확인:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && docker compose --env-file .env -f docker/vane/docker-compose.yml up -d --build && curl -s -X POST http://localhost:3000/api/search -H "Content-Type: application/json" -d '{"query":"\"오징어 게임\" TV 리뷰 후기 재미","sources":["web","discussions"],"stream":false}' | python -c "import sys,json;d=json.load(sys.stdin);s=d.get('sources');assert isinstance(s,list) and s,'no sources[]';assert (s[0].get('content') or s[0].get('pageContent')) and (s[0].get('url') or (s[0].get('metadata') or {}).get('url')),'shape mismatch';print('sources_ok',len(s))"
  ```
  Expected: `sources_ok N` (N>=1). 200 아님/`sources[]` 부재/shape 불일치면 §10 #9 미해소로 M0 중단하고 `VANE_REF`·`config.toml`·`VaneClient` 파서를 실제 버전에 맞춰 조정.
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add docker/vane/docker-compose.yml docker/vane/searxng/settings.yml docker/vane/config.toml tests/test_docker_vane.py && git commit -m "chore: vane docker-compose (pinned upstream build) with searxng json settings and api search smoke" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 12: M0 체크포인트 (비-TDD 검증) — 전체 pytest 회귀 + README + migrate CLI + roles.sql(DBA) + Vane §10 #9

> 이 태스크는 **산출물 문서화 + 인프라 검증**이라 실패→구현 리듬이 없다. CONTRACTS §8 TDD 프레이밍을 지키기 위해 **실패테스트 단계로 위장하지 않고**, 명시적으로 **비-TDD 문서화 + 전체 스위트 회귀 체크포인트**로 표기한다.

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI/README.md`

- [ ] **회귀 체크포인트 (비-TDD, 실패테스트 아님)** — 전체 스위트 일괄 실행으로 M0 산출물 회귀 없음 확인:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest -q
  ```
  Expected: `18 passed` (test_package 1, test_config 1, test_migrate 1, test_roles 1, test_models 3, test_clients_llm 2, test_clients_embedding 2, test_clients_vane 3, test_funtag_dict 2, test_docker_vane 2 = **18개 테스트 함수, `passed` 카운트 18**). `test_migrate`가 선행 `conn` 픽스처 오염과 무관하게 통과함을 함께 확인(자기 격리, Task 4).
- [ ] **문서화 (비-TDD)** — `README.md` 작성(기동 순서 + roles.sql DBA 단계 + §10 #9 명시):
  ```markdown
  # AOD AI — Content Intelligence 배치 (M0)

  ## 셋업
  1. `cp .env.example .env` 후 값 채우기 (RDS `aod_ai` 계정 + 관리형 API + `VANE_REF` 고정 ref).
  2. `python -m pip install -e ".[dev]"`
  3. **권한 격리 (DBA/슈퍼유저, 1회)**: `psql "<dba-url>" -v ai_user=aod_ai -f sql/roles.sql`
     → `public` 읽기전용 role(`aod_public_ro`) + `aod_ai` 읽기쓰기 (spec §2.2).
  4. 스키마 마이그레이션(aod_ai 계정): `python -m aod_ai.migrate`
  5. Vane 기동(§10 #9 검증된 `VANE_REF`): `docker compose --env-file .env -f docker/vane/docker-compose.yml up -d --build`
  6. 테스트: `python -m pytest -q` (Docker daemon 필요 — testcontainers pgvector/pgvector:pg16)

  ## 구성
  - `aod_ai/config.py` — Settings (CONTRACTS §5 env)
  - `aod_ai/db.py`, `aod_ai/migrate.py` — psycopg3 + pgvector + `aod_ai.schema_migrations`
  - `migrations/001..004` — CONTRACTS §2 DDL (004 HNSW는 M0에서 빈 테이블 위 생성; Task 4 결정)
  - `sql/roles.sql` — DBA 권한 격리 (자동 러너 밖, CREATEROLE 필요)
  - `aod_ai/models.py` — CONTRACTS §3 Pydantic + content_hash
  - `aod_ai/clients/{llm,embedding,vane}.py` — CONTRACTS §4 (시그니처 고정)
  - `resources/seed_fun_tags.yaml` + `aod_ai/funtag_dict.py` — 시드 사전 (출처: 원본 기획 aod_reccomendation.md)
  - `docker/vane/` — 고정 상류 ref 빌드 Vane + SearXNG JSON settings.yml
  ```
- [ ] **회귀 재확인 (비-TDD)** — 문서 추가 후 스위트 불변:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m pytest -q
  ```
  Expected: `18 passed`
- [ ] **인프라 체크포인트 ① (마이그레이터 CLI)** — 실제 pg에 `aod_ai` 스키마·HNSW 생성:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m aod_ai.migrate
  ```
  Expected: `applied migrations: ['001_schema_pgvector.sql', '
```bash
cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && python -m aod_ai.migrate
```
  Expected: `applied migrations: ['001_schema_pgvector.sql', '002_asset_tables.sql', '003_user_and_log_tables.sql', '004_hnsw_indexes.sql']` (재실행 시 `applied migrations: []`)
- [ ] **인프라 체크포인트 ② (DBA/슈퍼유저, role 격리 §2.2)** — `aod_ai` 마이그레이션 러너가 아니라 `CREATEROLE` 권한 계정으로 실행:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && psql "postgresql://<dba>@$AOD_DB_HOST:$AOD_DB_PORT/$AOD_DB_NAME" -v ai_user="$AOD_DB_USER" -f sql/roles.sql
  ```
  확인: `psql ... -c "SELECT 1 FROM pg_roles WHERE rolname='aod_public_ro'"` → 1행 반환; `\du` 에서 배치 계정(`$AOD_DB_USER`)이 `aod_public_ro` 멤버이고 `public`은 읽기전용·`aod_ai`는 읽기쓰기임을 확인.
- [ ] **인프라 체크포인트 ③ (§10 #9 Vane 스모크)** — Task 11 스모크가 200 + top-level `sources[]` 반환하고 각 원소 shape가 `VaneClient.search` 파서와 일치함을 재확인(미해소면 M0 미완):
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && docker compose --env-file .env -f docker/vane/docker-compose.yml up -d --build && curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:3000/api/search -H "Content-Type: application/json" -d '{"query":"\"오징어 게임\" TV 리뷰 후기 재미","sources":["web","discussions"],"stream":false}'
  ```
  Expected: `200`
- [ ] Commit:
  ```bash
  cd "C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-AI" && git add README.md && git commit -m "docs: M0 setup and checkpoint README (roles, migrate, vane)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
