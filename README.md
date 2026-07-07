# AOD AI — Content Intelligence 배치 (M0)

## 셋업
1. `cp .env.example .env` 후 값 채우기 (RDS `aod_ai` 계정 + 관리형 API + `VANE_REF` 고정 ref).
2. `python -m pip install -e ".[dev]"`
3. **권한 격리 (DBA/슈퍼유저, 1회)**: `psql "<dba-url>" -v ai_user=aod_ai -f sql/roles.sql`
   → `public` 읽기전용 role(`aod_public_ro`) + `aod_ai` 읽기쓰기 (spec §2.2).
4. 스키마 마이그레이션(aod_ai 계정): `python -m aod_ai.migrate`
5. 시드 사전 적재: `python -m aod_ai.funtag_dict --sync`
6. Vane 기동(§10 #9 검증된 `VANE_REF`): `docker compose --env-file .env -f docker/vane/docker-compose.yml up -d --build`
7. 테스트: `python -m pytest -q` (Docker daemon 필요 — testcontainers pgvector/pgvector:pg16)

## 구성
- `aod_ai/config.py` — Settings (CONTRACTS §5 env)
- `aod_ai/db.py`, `aod_ai/migrate.py` — psycopg3 + pgvector + `aod_ai.schema_migrations`
- `migrations/001..004` — CONTRACTS §2 DDL (004 HNSW는 M0에서 빈 테이블 위 생성; Task 4 결정)
- `sql/roles.sql` — DBA 권한 격리 (자동 러너 밖, CREATEROLE 필요)
- `aod_ai/models.py` — CONTRACTS §3 Pydantic + content_hash
- `aod_ai/clients/{llm,embedding,vane}.py` — CONTRACTS §4 (시그니처 고정)
- `resources/seed_fun_tags.yaml` + `aod_ai/funtag_dict.py` — 시드 사전 (출처: 원본 기획 aod_reccomendation.md)
- `docker/vane/` — 고정 상류 ref 빌드 Vane + SearXNG JSON settings.yml
