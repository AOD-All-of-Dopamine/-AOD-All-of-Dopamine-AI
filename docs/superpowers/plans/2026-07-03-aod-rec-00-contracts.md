# AOD 추천 구현 — 공유 계약 (Fixed Contracts)

> 이 문서는 M0/M1/M2 모든 태스크가 **글자 그대로 재사용**해야 하는 이름·타입·인터페이스·스키마다.
> 태스크 초안 작성자는 여기 정의된 테이블/컬럼/함수/env/타입 이름을 **절대 새로 만들지 말고 그대로 인용**할 것.
> 근거 스펙: `docs/superpowers/specs/2026-06-28-aod-recommendation-system-design.md`

## 0. 확정 기술 결정 (근거)

- 제공자: **OpenAI 호환**(base_url + api_key). LLM·Embedding 모두 동일 클라이언트 스타일. provider-agnostic.
- 임베딩 차원: **1024** (`dimensions=1024`, Qwen3-Embedding MRL). pgvector `vector(1024)` + HNSW(cosine). (HNSW `vector`형 상한 2000차원이라 1024는 안전.)
- 구조화 출력: `response_format={"type":"json_object"}` + 프롬프트에 스키마 JSON과 리터럴 단어 `json` 포함 + **Pydantic 검증 후 실패 시 재시도**.
- pgvector: RDS에서 사용 가능(pg15+). 확장은 `aod_ai` 스키마 대상으로 설치.
- M0/M1은 **로컬 PC**에서 실행(RDS는 읽기, Vane은 로컬 도커). M2 서빙은 기존 백엔드 API에 구현.

## 1. 레포 배치 (Python 배치 = AI 모듈, 서빙 = 백엔드 api 모듈)

### 1.1 Python 배치 프로젝트 — 루트: `-AOD-All-of-Dopamine-AI/`
```
-AOD-All-of-Dopamine-AI/
├── pyproject.toml                 # 패키지명: aod_ai
├── .env.example
├── aod_ai/
│   ├── __init__.py
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── db.py                      # psycopg 커넥션 + pgvector 등록 + aod_ai search_path
│   ├── migrate.py                 # 순번 .sql 적용 러너 (aod_ai.schema_migrations 추적)
│   ├── models.py                  # Pydantic: FunTagItem, Extraction, ReviewSource, QualityScore, SelectedTarget
│   ├── funtag_dict.py             # 시드 사전 로드/동기화
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── llm.py                 # LlmClient.extract(...)
│   │   ├── embedding.py           # EmbeddingClient.embed_text(...)
│   │   └── vane.py                # VaneClient.search(...)
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── select_targets.py      # ① select_targets(conn, domain, limit) -> list[SelectedTarget]
│   │   ├── collect_reviews.py     # ② collect_reviews(vane, target) -> list[ReviewSource]
│   │   ├── extract.py             # ③ extract_profile(llm, target, sources, active_tags) -> Extraction
│   │   ├── embed.py               # ④ build_profile_embedding(emb, extraction) -> list[float]
│   │   ├── quality.py             # ⑤ compute_quality(conn, target) -> QualityScore
│   │   └── upsert.py              # ⑥ upsert_assets(conn, target, extraction, vector, quality, sources)
│   └── run.py                     # CLI: python -m aod_ai.run --domain WEBNOVEL --limit 200
├── resources/
│   └── seed_fun_tags.yaml         # 시드 30~60개
├── migrations/
│   ├── 001_schema_pgvector.sql
│   ├── 002_asset_tables.sql
│   ├── 003_user_and_log_tables.sql
│   └── 004_hnsw_indexes.sql
├── docker/
│   └── vane/docker-compose.yml    # Vane 단일 컨테이너 (SearXNG 내장)
└── tests/
    ├── conftest.py                # 픽스처: 임시 pg(testcontainers 또는 로컬 aod_ai_test 스키마), 가짜 클라이언트
    ├── test_config.py
    ├── test_migrate.py
    ├── test_models.py
    ├── test_clients_llm.py
    ├── test_clients_embedding.py
    ├── test_clients_vane.py
    ├── test_select_targets.py
    ├── test_collect_reviews.py
    ├── test_extract.py
    ├── test_embed.py
    ├── test_quality.py
    └── test_upsert.py
```

**Python 의존성(pyproject.toml)**: `psycopg[binary]>=3.2`, `pgvector>=0.3`, `openai>=1.40`, `pydantic>=2.7`, `pydantic-settings>=2.3`, `httpx>=0.27`, `tenacity>=8.4`, `pyyaml>=6`, `python-dotenv>=1`. dev: `pytest>=8`, `pytest-asyncio`(필요시), `respx`(httpx mock), `testcontainers[postgres]`(선택).

### 1.2 서빙 (M2) — 백엔드 api 모듈: `-AOD-All-of-Dopamine-back/-AOD-All-of-Dopamine-api/`
Java 패키지 루트: `com.example.AOD.recommend`
```
com/example/AOD/recommend/
├── RecommendController.java        # GET /api/recommendations
├── RecommendService.java           # orchestration
├── candidate/
│   ├── CandidateGenerator.java     # union → dedupe
│   └── AiAssetRepository.java      # native pgvector/aod_ai 조회 (JPA @Query nativeQuery)
├── feature/FeatureCalculator.java  # 5 피처
├── rank/Ranker.java                # home/related score
├── postprocess/PostProcessor.java  # hard filter, soft penalty, diversity
├── dto/                            # RecommendationItem, RecRequest, ScoreBreakdown
└── log/RecLogWriter.java           # rec_impression insert
```
- 캐시: Spring Cache `@Cacheable("homeRecommendations")` (유저별 Top-N), 프로파일 변경/야간 evict.
- 백엔드 DB 계정에 `aod_ai` 읽기 + `rec_impression`/`rec_event` 쓰기 grant 필요(마이그레이션/DBA 태스크).
- pgvector 벡터 파라미터 바인딩: 네이티브 SQL에 `CAST(:vec AS vector)` 문자열(`'[0.1,0.2,...]'`) 방식.

## 2. aod_ai 스키마 DDL (마이그레이션 파일에 그대로)

`content_id`는 `public.contents.content_id`를 **논리 참조**하되 물리 FK 없음.

### 001_schema_pgvector.sql
```sql
CREATE SCHEMA IF NOT EXISTS aod_ai;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS aod_ai.schema_migrations (
  version    text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);
```

### 002_asset_tables.sql
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

### 003_user_and_log_tables.sql
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

### 004_hnsw_indexes.sql  (데이터 적재 후 실행 권장)
```sql
CREATE INDEX idx_content_embedding_hnsw ON aod_ai.content_embedding
  USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_user_profile_vector_hnsw ON aod_ai.user_profile_cache
  USING hnsw (profile_vector vector_cosine_ops);
```

## 3. Pydantic 모델 (aod_ai/models.py) — 필드명 고정

```python
class FunTagItem(BaseModel):
    tag: str
    tag_score: float          # 0..1
    tag_confidence: float      # 0..1
    evidence: str              # 태그 근거가 된 짧은 리뷰 표현
    is_new: bool               # 활성 사전에 없으면 True → status=proposed

class Extraction(BaseModel):
    fun_tags: list[FunTagItem]
    normalized_summary: str
    profile_text: str          # 임베딩 대상 텍스트
    extraction_quality: float  # 0..1

class ReviewSource(BaseModel):
    content: str               # Vane sources[].content 원본 스니펫
    url: str

class SelectedTarget(BaseModel):
    content_id: int
    domain: str
    master_title: str
    original_title: str | None
    synopsis: str | None
    genres: list[str]
    content_hash: str          # 변경 감지용 (아래 3.1)

class QualityScore(BaseModel):
    bayesian_score: float
    platform_rank_score: float
    review_count_score: float
    recency_score: float
    quality_popularity_score: float
```

### 3.1 content_hash 정의 (변경 감지)
`content_hash = sha256(f"{master_title}|{original_title}|{synopsis}|{sorted(genres)}")` (hexdigest). 프로파일 없음 또는 hash 불일치인 콘텐츠만 재처리.

## 4. 클라이언트 인터페이스 (glass-box, 시그니처 고정)

### aod_ai/clients/llm.py
```python
class LlmClient:
    def __init__(self, base_url: str, api_key: str, model: str): ...
    def extract(self, *, metadata: SelectedTarget, sources: list[ReviewSource],
                active_tags: list[str]) -> Extraction:
        # response_format={"type":"json_object"}; 프롬프트에 스키마 + 리터럴 "json" 포함;
        # 응답을 Extraction.model_validate_json; 실패 시 tenacity로 최대 3회 재시도
```

### aod_ai/clients/embedding.py
```python
class EmbeddingClient:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1024): ...
    def embed_text(self, text: str) -> list[float]:
        # OpenAI 호환 embeddings, dimensions=self.dim; 반환 길이 == self.dim 보장(assert)
```

### aod_ai/clients/vane.py
```python
class VaneClient:
    def __init__(self, base_url: str): ...
    def search(self, *, query: str, sources: list[str],
               system_instructions: str | None = None) -> list[ReviewSource]:
        # POST {base_url}/api/search, stream:false; 응답 sources[]에서 content/url 추출;
        # 상한 N개(config), dedupe(url), 길이예산 적용
```

## 5. 설정 (aod_ai/config.py, pydantic-settings) — env 키 고정

```
# DB (읽기: public, 쓰기: aod_ai) — 로컬 M0/M1은 RDS 읽기 + 로컬 or RDS aod_ai
AOD_DB_HOST, AOD_DB_PORT=5432, AOD_DB_NAME, AOD_DB_USER, AOD_DB_PASSWORD
# OpenAI 호환
OPENAI_BASE_URL, OPENAI_API_KEY, LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM=1024
# Vane
VANE_BASE_URL=http://localhost:3000
# 파이프라인 튜닝
VANE_MAX_SOURCES=8, EXTRACT_MAX_RETRIES=3
```
백엔드 DB의 property 키(참고, 서빙 M2용): `spring.datasource.url|username|password`, `hibernate.default_schema=public`, crawler env `POSTGRES_HOST/PORT/DB/USER/PASSWORD`.

## 6. 백엔드 읽기 스키마 (public) — 배치가 읽는 실제 테이블/컬럼

- **contents** (`Content.java:14`): `content_id bigint PK`, `domain varchar(50)`(MOVIE/TV/GAME/WEBTOON/WEBNOVEL), `master_title varchar(500)`, `original_title varchar(500)`, `release_date date`, `synopsis text`, `average_score double precision`, `review_count integer`, `created_at`, `updated_at`. 인덱스 `idx_contents_lookup(domain, master_title, release_date)`.
- **webnovel_contents** (`WebnovelContent.java:12`, M1 대상): `content_id bigint PK/FK`, `author varchar(200)`, `publisher varchar(200)`, `age_rating varchar(50)`, `genres text[]`, `platforms text[]`. GIN: `idx_webnovel_genres`, `idx_webnovel_platforms`.
- 기타 도메인: `movie_contents`(genres[],platforms[],directors[],cast_members[],runtime), `tv_contents`(genres[],platforms[],cast_members[],season_count,episode_runtime), `game_contents`(genres[],platforms[],developer,publisher,os_platforms jsonb), `webtoon_contents`(genres[],platforms[],author,status,weekday,age_rating). 모두 `content_id` PK/FK.
- **reviews** (`Review.java:15`, api 모듈): `review_id PK`, `content_id FK`, `user_id FK`, `rating double precision`(0.0~5.0), `title varchar(100)`, `review_content text`, `created_at`, `updated_at`. 유니크 `(content_id,user_id)`. 인덱스 `idx_review_content_rating(content_id,rating)`. **단일 테이블(플랫폼 리뷰 별도 테이블 없음).**
- **external_ranking** (`ExternalRanking.java:25`): `id PK`, `platform_specific_id varchar`, `content_id bigint FK(nullable)`, `title varchar`, `ranking integer`(1..N), `platform varchar`(예 "NaverWebtoon","Steam","TMDB_MOVIE"), `thumbnail_url`, `watch_providers jsonb`.
- **platform_data** (`PlatformData.java:17`): `platform_data_id PK`, `content_id FK`, `platform_name varchar(100)`, `platform_specific_id`, `url varchar(1000)`, `attributes jsonb`(rating/comment_count/author/publisher/status/age_rating/genres/synopsis/watch_providers 등), `last_seen_at`. 유니크 `(platform_name,platform_specific_id)`.
- 유저 상호작용(후속 마일스톤): **users**(`id PK`, username, email, roles), **bookmarks**(`bookmark_id PK`, content_id, user_id, created_at; uniq (content_id,user_id)), **content_likes**(`like_id PK`, content_id, user_id, `like_type varchar(10)`=LIKE|DISLIKE, created_at; uniq (content_id,user_id)).
- 마이그레이션: Flyway, `-AOD-All-of-Dopamine-api/.../db/migration/V<n>__<desc>.sql`. 기존 V1(genres GIN), V3(platforms text[]+GIN). **aod_ai는 이 Flyway와 별개**(Python migrate.py로 관리).

## 7. 서빙 계약 (M2) — 요청/응답/쿼리 형태

- 엔드포인트: `GET /api/recommendations?location=home|related&selectedContentId={id?}&page=0&size=20`. 유저는 인증 principal에서.
- 후보 벡터 검색(pgvector): `SELECT content_id FROM aod_ai.content_embedding ORDER BY embedding <=> CAST(:vec AS vector) LIMIT :k`.
- fun_tag 후보: `SELECT content_id FROM aod_ai.content_fun_tag WHERE tag = ANY(:tags)`.
- 랭킹 가중치(초기): home = 0.45·funtag + 0.25·profile_sim + 0.15·quality + 0.10·metadata + 0.05·recency. related = 0.4·home + 0.6·selected_sim.
- 각 노출 → `aod_ai.rec_impression` insert (request_id uuid, location, candidate_source, rank_position, score_breakdown jsonb).
- 콜드스타트 분기: `user_profile_cache.positive_count` (0 / 1~2 / 3+ / 충분).

## 8. 네이밍/규약
- Python: 함수/모듈 snake_case, 위 §1.1·§4 시그니처 그대로. 테스트는 `test_<module>.py`, 함수 `test_<behavior>`.
- 커밋: `feat:`/`test:`/`chore:` 접두, 태스크당 1커밋. 각 커밋 끝에 `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- TDD: 실패 테스트 → 실행(실패 확인) → 최소 구현 → 실행(통과) → 커밋.

## 9. 통합 보정 규칙 (교차 마일스톤 일관성 — 실행 시 반드시 준수)

워크플로우 통합 비평이 M0→M1→M2 조립에서 찾은 교차 이슈의 확정 해소책. 각 마일스톤 태스크는 아래를 우선한다.

- **[정규화 태그 저장]** `upsert_assets`(M1)는 `is_new=false` 태그를 `content_fun_tag.tag`에 저장할 때 **raw `item.tag`가 아니라 정규화 매칭된 `fun_tag_dict.name`(canonical)** 을 넣는다. 이유: M2 후보 쿼리가 `WHERE tag = ANY(:tags)`로 canonical dict 이름과 **정확히** 매칭하므로, LLM이 대소문자/공백만 다르게 뱉으면 최고 가중(funtag 0.45) 후보를 놓친다. → upsert에서 `item.tag`를 active dict에 strip/lower 매칭해 그 canonical name으로 치환 후 저장(매칭 실패 시에만 raw 유지 + 로깅).
- **[funtag_dict --sync 인터페이스 고정]** M0 산출 `aod_ai/funtag_dict.py`는 `python -m aod_ai.funtag_dict --sync` CLI를 제공하고, `resources/seed_fun_tags.yaml` 항목을 `aod_ai.fun_tag_dict`에 **status='active'** 로 upsert한다. M1의 `_load_active_tags`는 status='active'만 읽으므로 이 계약이 성립해야 파이프라인이 동작한다.
- **[마이그레이션 순서·소유권]** M2의 Flyway `V4__grant_aod_ai_access.sql`는 **M0의 aod_ai 마이그레이션이 선행**해야 grant 대상 객체가 존재한다. 또한 grant 실행 주체가 객체 소유자여야 성공한다 — 로컬(단일 postgres 계정)은 자동 충족, **RDS(배치=aod_ai / 서빙=postgres 분리)에선 V4를 aod_ai 소유자로 실행하거나 `sql/roles.sql`에서 서빙 계정에 미리 grant**한다.
- **[HNSW 순서]** M0 Task 4가 빈 테이블에 004 HNSW를 생성하는 것은 **의도된 결정**(§2 004 주석 오버라이드, M0 Task 4에 근거 문서화). 대량 재적재(M5) 시 `REINDEX`로 재빌드.
