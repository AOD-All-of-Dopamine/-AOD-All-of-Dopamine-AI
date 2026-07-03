# M1 — Content Intelligence 파이프라인 (AOD 추천) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (권장) 또는 superpowers:executing-plans. 스텝은 `- [ ]` 체크박스.

**Goal:** M0 클라이언트 위에 6단계 배치 파이프라인(①대상선별 ②Vane 리뷰수집 ③Qwen 추출 ④임베딩 ⑤품질점수 ⑥upsert)과 `run.py` CLI를 TDD로 구현하고, 웹소설 100~200건 샘플에 풀가동해 추출된 fun_tag를 눈검수(핵심 가설 검증)한다.

**Architecture:** 콘텐츠 1건당 한 번 실행되는 오프라인 배치. `public.contents`+`webnovel_contents` 읽기 → Vane HTTP로 리뷰 evidence 수집 → LLM으로 fun_tag/profile_text 추출 → 임베딩 → 품질점수 → `aod_ai` 4테이블 upsert. 신규 태그는 `proposed`로 분리.

**Tech Stack:** Python 3.11+ (M0 산출물 재사용), pytest(단위=목 클라이언트, 통합=로컬 pg `aod_ai_test`).

**전제:** M0 완료(스캐폴드·마이그레이션·클라이언트·시드 사전). **공유 계약:** `2026-07-03-aod-rec-00-contracts.md` (특히 §4 파이프라인, §9 통합 보정 [정규화 태그 저장]·[funtag_dict --sync]) 를 그대로 인용.

---

## M1 — Content Intelligence 파이프라인 (웹소설 샘플 풀가동 + fun_tag 눈검수)

**Deliverable/Checkpoint:** M0 클라이언트 인터페이스(`LlmClient`/`EmbeddingClient`/`VaneClient`, CONTRACTS §4) 위에 6단계 배치 파이프라인 ①~⑥과 `run.py` CLI를 TDD로 구현하고, 웹소설 샘플에 대해 `python -m aod_ai.run --domain WEBNOVEL --limit 200`이 `aod_ai` 자산 4테이블을 채우며, 추출된 fun_tag+evidence를 사람이 읽을 파일로 덤프해 **핵심 가설(fun_tag 품질)을 눈검수**한다.

> 전제(M0 산출물, 재작성 금지): `aod_ai/models.py`(CONTRACTS §3), `aod_ai/config.py`의 `Settings`(§5), `aod_ai/db.py`의 `connect()`, `aod_ai/clients/*`(§4), `aod_ai/pipeline/__init__.py`, `aod_ai/funtag_dict.py`(시드 사전 로드/동기화, §1.1), `resources/seed_fun_tags.yaml`(시드 30~60), `migrations/001~004*.sql`(§2), 그리고 **기존 `tests/conftest.py`(§1.1: 임시 pg 픽스처 + 가짜 클라이언트)** 는 이미 존재.
> 모든 명령은 `-AOD-All-of-Dopamine-AI/`에서 실행. 테스트 DB는 로컬 Postgres(pgvector 설치 가능): `AOD_TEST_DSN`(기본 `postgresql://postgres:postgres@localhost:5432/aod_ai_test`).

---

### Task 1: 대상 선별 ① `select_targets` + 테스트 하네스(conftest)

**Files:**
- Create: `-AOD-All-of-Dopamine-AI/aod_ai/pipeline/select_targets.py`
- Test: `-AOD-All-of-Dopamine-AI/tests/test_select_targets.py`
- Modify: `-AOD-All-of-Dopamine-AI/tests/conftest.py`

- [ ] **먼저 기존 `tests/conftest.py`(M0 산출물)를 읽어** 이미 정의된 픽스처를 확인한다. M0가 제공하는 가짜 클라이언트 픽스처(`fake_vane`/`fake_llm`/`fake_emb` 또는 동등물)와 DB/스키마 픽스처가 있으면 **그대로 재사용**하고 아래에서 중복 정의하지 않는다(같은 파일 재정의는 조용히 shadow 되므로 금지). M0에 없거나 이름이 다를 때만 아래 M1 픽스처를 **추가**한다. env를 주입하는 픽스처는 **`autouse` 금지**(M0의 `test_config.py`/`test_clients_*.py` 환경을 오염시키지 않도록 opt-in), 이름은 M1 전용 `m1_env`로 둔다.
- [ ] 실패 테스트 작성 — 먼저 `tests/conftest.py`에 M1 픽스처(`m1_env`(opt-in), `_schema`, `db`)를 추가(M0에 동일 역할이 없을 때만):
```python
# tests/conftest.py  (M1 추가분 — M0 conftest에 없는 것만 추가, autouse 사용 안 함)
import os
import pathlib
import psycopg
import pytest
from pgvector.psycopg import register_vector

_MIGRATIONS = pathlib.Path(__file__).resolve().parents[1] / "migrations"
_DSN = os.environ.get(
    "AOD_TEST_DSN", "postgresql://postgres:postgres@localhost:5432/aod_ai_test"
)

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
def _schema():
    conn = psycopg.connect(_DSN, autocommit=True)
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


@pytest.fixture()
def db(_schema):
    conn = psycopg.connect(_DSN, autocommit=True)
    register_vector(conn)
    conn.execute(
        "TRUNCATE aod_ai.content_semantic_profile, aod_ai.content_fun_tag, "
        "aod_ai.content_embedding, aod_ai.content_quality_score, aod_ai.fun_tag_dict"
    )
    conn.execute("TRUNCATE public.contents, public.webnovel_contents, public.external_ranking")
    yield conn
    conn.close()
```
그리고 `tests/test_select_targets.py`:
```python
from aod_ai.models import SelectedTarget
from aod_ai.pipeline.select_targets import compute_content_hash, select_targets


def _insert_webnovel(db, content_id, title, synopsis, genres):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, original_title, synopsis) "
            "VALUES (%s, 'WEBNOVEL', %s, NULL, %s)",
            (content_id, title, synopsis),
        )
        cur.execute(
            "INSERT INTO public.webnovel_contents (content_id, genres, platforms) "
            "VALUES (%s, %s, %s)",
            (content_id, genres, []),
        )


def test_select_targets_includes_missing_and_changed_skips_uptodate(db):
    _insert_webnovel(db, 1, "전지적 독자 시점", "재난 웹소설", ["판타지"])
    _insert_webnovel(db, 2, "달빛조각사", "게임 판타지", ["게임판타지"])
    _insert_webnovel(db, 3, "나 혼자만 레벨업", "헌터물", ["액션"])
    h2 = compute_content_hash("달빛조각사", None, "게임 판타지", ["게임판타지"])
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO aod_ai.content_semantic_profile "
            "(content_id, domain, profile_text, content_hash) "
            "VALUES (2, 'WEBNOVEL', 'x', %s)",
            (h2,),
        )
        cur.execute(
            "INSERT INTO aod_ai.content_semantic_profile "
            "(content_id, domain, profile_text, content_hash) "
            "VALUES (3, 'WEBNOVEL', 'x', 'STALE')"
        )

    targets = select_targets(db, "WEBNOVEL", 200)

    assert [t.content_id for t in targets] == [1, 3]
    assert all(isinstance(t, SelectedTarget) for t in targets)
    assert targets[0].genres == ["판타지"]
    assert targets[0].content_hash == compute_content_hash(
        "전지적 독자 시점", None, "재난 웹소설", ["판타지"]
    )
```
- [ ] 실행 → 실패 확인: `python -m pytest tests/test_select_targets.py -q` → `ModuleNotFoundError: No module named 'aod_ai.pipeline.select_targets'` (수집 에러, `1 error`).
- [ ] 최소 구현 — `aod_ai/pipeline/select_targets.py`:
```python
import hashlib

from aod_ai.models import SelectedTarget

_CANDIDATE_SQL = """
SELECT c.content_id, c.domain, c.master_title, c.original_title,
       c.synopsis, COALESCE(w.genres, '{}') AS genres
FROM public.contents c
JOIN public.webnovel_contents w ON w.content_id = c.content_id
WHERE c.domain = %s
ORDER BY c.content_id
"""
_EXISTING_SQL = "SELECT content_id, content_hash FROM aod_ai.content_semantic_profile"


def compute_content_hash(master_title, original_title, synopsis, genres):
    payload = f"{master_title}|{original_title}|{synopsis}|{sorted(genres)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_targets(conn, domain: str, limit: int) -> list[SelectedTarget]:
    with conn.cursor() as cur:
        cur.execute(_EXISTING_SQL)
        existing = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute(_CANDIDATE_SQL, (domain,))
        rows = cur.fetchall()

    out: list[SelectedTarget] = []
    for content_id, dom, master_title, original_title, synopsis, genres in rows:
        genres = list(genres or [])
        content_hash = compute_content_hash(master_title, original_title, synopsis, genres)
        if existing.get(content_id) == content_hash:
            continue
        out.append(
            SelectedTarget(
                content_id=content_id,
                domain=dom,
                master_title=master_title,
                original_title=original_title,
                synopsis=synopsis,
                genres=genres,
                content_hash=content_hash,
            )
        )
        if len(out) >= limit:
            break
    return out
```
- [ ] 실행 → 통과 확인: `python -m pytest tests/test_select_targets.py -q` → `1 passed`.
- [ ] 커밋:
```
git add aod_ai/pipeline/select_targets.py tests/test_select_targets.py tests/conftest.py && git commit -m "feat: select_targets stage with content_hash change detection" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: 리뷰 수집 ② `collect_reviews`

**Files:**
- Create: `-AOD-All-of-Dopamine-AI/aod_ai/pipeline/collect_reviews.py`
- Test: `-AOD-All-of-Dopamine-AI/tests/test_collect_reviews.py`
- Modify: `-AOD-All-of-Dopamine-AI/tests/conftest.py` (`fake_vane` 픽스처 — M0에 없을 때만 추가)

- [ ] 실패 테스트 작성 — `tests/conftest.py`에 `fake_vane`가 M0에 **없을 때만** 추가(있으면 재사용):
```python
# tests/conftest.py  (Task 2 추가분 — M0가 fake_vane 미제공 시에만)
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
```
`tests/test_collect_reviews.py` (Settings()를 쓰므로 `m1_env`를 명시적으로 요청):
```python
from aod_ai.models import ReviewSource, SelectedTarget
from aod_ai.pipeline.collect_reviews import VANE_SYSTEM_INSTRUCTIONS, collect_reviews


def _target():
    return SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="전지적 독자 시점",
        original_title=None, synopsis="재난", genres=["판타지"], content_hash="h",
    )


def test_collect_reviews_builds_query_dedupes_and_caps(monkeypatch, fake_vane, m1_env):
    monkeypatch.setenv("VANE_MAX_SOURCES", "3")
    raw = [ReviewSource(content=f"리뷰{i}", url=f"http://u/{i}") for i in range(5)]
    raw.append(ReviewSource(content="중복", url="http://u/0"))
    vane = fake_vane(raw)

    out = collect_reviews(vane, _target())

    assert len(out) == 3
    assert [s.url for s in out] == ["http://u/0", "http://u/1", "http://u/2"]
    call = vane.calls[0]
    # WEBNOVEL → 웹소설 (도메인 라벨은 target.domain에서 파생)
    assert call["query"] == '"전지적 독자 시점" 웹소설 리뷰 후기 재미'
    assert call["sources"] == ["web", "discussions"]
    assert call["system_instructions"] == VANE_SYSTEM_INSTRUCTIONS
```
- [ ] 실행 → 실패 확인: `python -m pytest tests/test_collect_reviews.py -q` → `ModuleNotFoundError: No module named 'aod_ai.pipeline.collect_reviews'` (`1 error`).
- [ ] 최소 구현 — `aod_ai/pipeline/collect_reviews.py` (스펙 §4② 템플릿 `"{제목}" {도메인} 리뷰 후기 재미`의 `{도메인}`을 `target.domain`에서 매핑으로 파생 — 하드코딩 금지, OCP 준수):
```python
from aod_ai.config import Settings
from aod_ai.models import ReviewSource

VANE_SYSTEM_INSTRUCTIONS = (
    "독자 반응과 재미 요소 위주로 실제 리뷰·후기의 원본 표현을 최대한 보존해 수집하라. "
    "합성 요약을 만들지 말고, 독자들이 재미있다고 느낀 지점의 원문 스니펫을 그대로 전달하라."
)

# 스펙 §4②의 {도메인} 라벨. 새 도메인은 여기에 한 줄만 추가하면 확장(쿼리 문자열 수정 불필요).
_DOMAIN_LABEL = {
    "WEBNOVEL": "웹소설",
    "WEBTOON": "웹툰",
    "MOVIE": "영화",
    "TV": "드라마",
    "GAME": "게임",
}


def collect_reviews(vane, target) -> list[ReviewSource]:
    label = _DOMAIN_LABEL[target.domain]
    query = f'"{target.master_title}" {label} 리뷰 후기 재미'
    raw = vane.search(
        query=query,
        sources=["web", "discussions"],
        system_instructions=VANE_SYSTEM_INSTRUCTIONS,
    )
    max_sources = Settings().VANE_MAX_SOURCES
    seen: set[str] = set()
    out: list[ReviewSource] = []
    for src in raw:
        if src.url in seen:
            continue
        seen.add(src.url)
        out.append(src)
        if len(out) >= max_sources:
            break
    return out
```
- [ ] 실행 → 통과 확인: `python -m pytest tests/test_collect_reviews.py -q` → `1 passed`.
- [ ] 커밋:
```
git add aod_ai/pipeline/collect_reviews.py tests/test_collect_reviews.py tests/conftest.py && git commit -m "feat: collect_reviews stage with domain-label mapping, dedupe and VANE_MAX_SOURCES cap" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: LLM 추출 ③ `extract_profile`

**Files:**
- Create: `-AOD-All-of-Dopamine-AI/aod_ai/pipeline/extract.py`
- Test: `-AOD-All-of-Dopamine-AI/tests/test_extract.py`
- Modify: `-AOD-All-of-Dopamine-AI/tests/conftest.py` (`fake_llm` 픽스처 — M0에 없을 때만 추가)

- [ ] 실패 테스트 작성 — `tests/conftest.py`에 `fake_llm`가 M0에 **없을 때만** 추가(있으면 재사용):
```python
# tests/conftest.py  (Task 3 추가분 — M0가 fake_llm 미제공 시에만)
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
```
`tests/test_extract.py` (Settings() 미사용 → env 불필요):
```python
from aod_ai.models import Extraction, FunTagItem, ReviewSource, SelectedTarget
from aod_ai.pipeline.extract import extract_profile


def _target():
    return SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="T",
        original_title=None, synopsis="s", genres=["판타지"], content_hash="h",
    )


def test_extract_profile_recomputes_is_new_against_active_tags(fake_llm):
    model_out = Extraction(
        fun_tags=[
            FunTagItem(tag="사이다전개", tag_score=0.9, tag_confidence=0.8,
                       evidence="속 시원한 복수", is_new=True),
            FunTagItem(tag="회귀먼치킨", tag_score=0.7, tag_confidence=0.6,
                       evidence="회귀 후 무쌍", is_new=False),
        ],
        normalized_summary="요약", profile_text="임베딩용 텍스트", extraction_quality=0.75,
    )
    llm = fake_llm(model_out)

    result = extract_profile(
        llm, _target(), [ReviewSource(content="c", url="u")],
        active_tags=["사이다전개", "성장물"],
    )

    flags = {i.tag: i.is_new for i in result.fun_tags}
    assert flags == {"사이다전개": False, "회귀먼치킨": True}
    assert llm.calls[0]["active_tags"] == ["사이다전개", "성장물"]
    assert result.profile_text == "임베딩용 텍스트"
```
- [ ] 실행 → 실패 확인: `python -m pytest tests/test_extract.py -q` → `ModuleNotFoundError: No module named 'aod_ai.pipeline.extract'` (`1 error`).
- [ ] 최소 구현 — `aod_ai/pipeline/extract.py`:
```python
from aod_ai.models import Extraction


def extract_profile(llm, target, sources, active_tags: list[str]) -> Extraction:
    extraction = llm.extract(metadata=target, sources=sources, active_tags=active_tags)
    active = {t.strip().lower() for t in active_tags}
    corrected = [
        item.model_copy(update={"is_new": item.tag.strip().lower() not in active})
        for item in extraction.fun_tags
    ]
    return extraction.model_copy(update={"fun_tags": corrected})
```
- [ ] 실행 → 통과 확인: `python -m pytest tests/test_extract.py -q` → `1 passed`.
- [ ] 커밋:
```
git add aod_ai/pipeline/extract.py tests/test_extract.py tests/conftest.py && git commit -m "feat: extract_profile stage authoritatively recomputes is_new vs active_tags" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: 임베딩 ④ `build_profile_embedding`

**Files:**
- Create: `-AOD-All-of-Dopamine-AI/aod_ai/pipeline/embed.py`
- Test: `-AOD-All-of-Dopamine-AI/tests/test_embed.py`
- Modify: `-AOD-All-of-Dopamine-AI/tests/conftest.py` (`fake_emb` 픽스처 — M0에 없을 때만 추가)

- [ ] 실패 테스트 작성 — `tests/conftest.py`에 `fake_emb`가 M0에 **없을 때만** 추가(있으면 재사용):
```python
# tests/conftest.py  (Task 4 추가분 — M0가 fake_emb 미제공 시에만)
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
```
`tests/test_embed.py` (Settings() 미사용 → env 불필요):
```python
import pytest

from aod_ai.models import Extraction
from aod_ai.pipeline.embed import build_profile_embedding


def _extraction():
    return Extraction(
        fun_tags=[], normalized_summary="s",
        profile_text="임베딩 텍스트", extraction_quality=0.5,
    )


def test_build_profile_embedding_returns_1024_and_passes_profile_text(fake_emb):
    emb = fake_emb([0.0] * 1024)
    vec = build_profile_embedding(emb, _extraction())
    assert len(vec) == 1024
    assert emb.texts == ["임베딩 텍스트"]


def test_build_profile_embedding_rejects_wrong_dimension(fake_emb):
    emb = fake_emb([0.0] * 512)
    with pytest.raises(AssertionError):
        build_profile_embedding(emb, _extraction())
```
- [ ] 실행 → 실패 확인: `python -m pytest tests/test_embed.py -q` → `ModuleNotFoundError: No module named 'aod_ai.pipeline.embed'` (`1 error`).
- [ ] 최소 구현 — `aod_ai/pipeline/embed.py`:
```python
def build_profile_embedding(emb, extraction) -> list[float]:
    vector = emb.embed_text(extraction.profile_text)
    assert len(vector) == 1024, f"expected 1024-dim embedding, got {len(vector)}"
    return vector
```
- [ ] 실행 → 통과 확인: `python -m pytest tests/test_embed.py -q` → `2 passed`.
- [ ] 커밋:
```
git add aod_ai/pipeline/embed.py tests/test_embed.py tests/conftest.py && git commit -m "feat: build_profile_embedding stage with 1024-dim guard" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: 품질점수 ⑤ `compute_quality`

**Files:**
- Create: `-AOD-All-of-Dopamine-AI/aod_ai/pipeline/quality.py`
- Test: `-AOD-All-of-Dopamine-AI/tests/test_quality.py`

- [ ] 실패 테스트 작성 — `tests/test_quality.py` (Settings() 미사용 → env 불필요):
```python
from datetime import date

import pytest

from aod_ai.models import SelectedTarget
from aod_ai.pipeline.quality import compute_quality


def _seed(db):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, average_score, review_count, release_date) "
            "VALUES (1, 'WEBNOVEL', 'A', 4.0, 10, %s)",
            (date.today(),),
        )
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, average_score, review_count, release_date) "
            "VALUES (2, 'WEBNOVEL', 'B', 2.0, 0, %s)",
            (date.today(),),
        )
        cur.execute(
            "INSERT INTO public.external_ranking (platform, ranking, content_id, title) "
            "VALUES ('NaverSeries', 1, 1, 'A')"
        )


def test_compute_quality_bayesian_and_components(db):
    _seed(db)
    target = SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="A",
        original_title=None, synopsis=None, genres=[], content_hash="h",
    )

    q = compute_quality(db, target)

    assert q.bayesian_score == pytest.approx(3.5)         # 0.5*4 + 0.5*3(C=global mean)
    assert q.platform_rank_score == pytest.approx(0.5)    # 1/(1+1)
    assert q.review_count_score == pytest.approx(1.0)     # log1p(10)/log1p(10)
    assert q.recency_score == pytest.approx(1.0)          # released today
    assert q.quality_popularity_score == pytest.approx(0.73)  # .4*.7+.3*.5+.2*1+.1*1
```
- [ ] 실행 → 실패 확인: `python -m pytest tests/test_quality.py -q` → `ModuleNotFoundError: No module named 'aod_ai.pipeline.quality'` (`1 error`).
- [ ] 최소 구현 — `aod_ai/pipeline/quality.py`:
```python
import math
from datetime import date

from aod_ai.models import QualityScore

BAYESIAN_MIN_VOTES = 10.0
RECENCY_HALFLIFE_DAYS = 365.0

_TARGET_SQL = """
SELECT c.average_score, c.review_count, c.release_date,
       (SELECT MIN(er.ranking) FROM public.external_ranking er
        WHERE er.content_id = c.content_id) AS best_rank
FROM public.contents c
WHERE c.content_id = %s
"""
_GLOBAL_SQL = """
SELECT AVG(average_score), MAX(review_count)
FROM public.contents
WHERE average_score IS NOT NULL
"""


def compute_quality(conn, target) -> QualityScore:
    with conn.cursor() as cur:
        cur.execute(_TARGET_SQL, (target.content_id,))
        avg_score, review_count, release_date, best_rank = cur.fetchone()
        cur.execute(_GLOBAL_SQL)
        global_avg, max_reviews = cur.fetchone()

    C = float(global_avg or 0.0)
    R = float(avg_score) if avg_score is not None else C
    v = float(review_count or 0)
    m = BAYESIAN_MIN_VOTES
    bayesian = (v / (v + m)) * R + (m / (v + m)) * C

    platform_rank = 1.0 / (1.0 + float(best_rank)) if best_rank else 0.0

    max_reviews = float(max_reviews or 0)
    review_count_score = math.log1p(v) / math.log1p(max_reviews) if max_reviews > 0 else 0.0

    if release_date is not None:
        days = (date.today() - release_date).days
        recency = 0.5 ** (days / RECENCY_HALFLIFE_DAYS)
    else:
        recency = 0.0

    quality_popularity = (
        0.4 * (bayesian / 5.0)
        + 0.3 * platform_rank
        + 0.2 * review_count_score
        + 0.1 * recency
    )

    return QualityScore(
        bayesian_score=bayesian,
        platform_rank_score=platform_rank,
        review_count_score=review_count_score,
        recency_score=recency,
        quality_popularity_score=quality_popularity,
    )
```
- [ ] 실행 → 통과 확인: `python -m pytest tests/test_quality.py -q` → `1 passed`.
- [ ] 커밋:
```
git add aod_ai/pipeline/quality.py tests/test_quality.py && git commit -m "feat: compute_quality stage (bayesian + rank + review_count + recency)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: 자산 upsert ⑥ `upsert_assets`

**Files:**
- Create: `-AOD-All-of-Dopamine-AI/aod_ai/pipeline/upsert.py`
- Test: `-AOD-All-of-Dopamine-AI/tests/test_upsert.py`

- [ ] 실패 테스트 작성 — `tests/test_upsert.py` (Settings().EMBEDDING_MODEL 사용 → `m1_env` 요청):
```python
import pytest

from aod_ai.models import (
    Extraction, FunTagItem, QualityScore, ReviewSource, SelectedTarget,
)
from aod_ai.pipeline.upsert import upsert_assets


def _target():
    return SelectedTarget(
        content_id=7, domain="WEBNOVEL", master_title="업서트대상",
        original_title=None, synopsis="s", genres=["판타지"], content_hash="HASH7",
    )


def _extraction():
    return Extraction(
        fun_tags=[
            FunTagItem(tag="사이다전개", tag_score=0.9, tag_confidence=0.8,
                       evidence="속시원", is_new=False),
            FunTagItem(tag="신규태그", tag_score=0.6, tag_confidence=0.5,
                       evidence="새로움", is_new=True),
        ],
        normalized_summary="요약", profile_text="프로파일 텍스트", extraction_quality=0.7,
    )


def _quality():
    return QualityScore(
        bayesian_score=3.5, platform_rank_score=0.5, review_count_score=1.0,
        recency_score=1.0, quality_popularity_score=0.73,
    )


def test_upsert_assets_writes_tables_isolates_proposed_and_is_idempotent(db, m1_env):
    with db.cursor() as cur:
        cur.execute("INSERT INTO aod_ai.fun_tag_dict (name, status) VALUES ('사이다전개', 'active')")
    sources = [ReviewSource(content="리뷰원문", url="http://u/1")]

    upsert_assets(db, _target(), _extraction(), [0.1] * 1024, _quality(), sources)

    with db.cursor() as cur:
        cur.execute(
            "SELECT profile_text, source_count, content_hash "
            "FROM aod_ai.content_semantic_profile WHERE content_id = 7"
        )
        assert cur.fetchone() == ("프로파일 텍스트", 1, "HASH7")
        cur.execute("SELECT tag FROM aod_ai.content_fun_tag WHERE content_id = 7 ORDER BY tag")
        assert [r[0] for r in cur.fetchall()] == ["사이다전개"]  # proposed 태그는 매칭 테이블 제외
        cur.execute("SELECT status FROM aod_ai.fun_tag_dict WHERE name = '신규태그'")
        assert cur.fetchone() == ("proposed",)
        cur.execute("SELECT dim FROM aod_ai.content_embedding WHERE content_id = 7")
        assert cur.fetchone() == (1024,)
        cur.execute(
            "SELECT quality_popularity_score FROM aod_ai.content_quality_score WHERE content_id = 7"
        )
        assert cur.fetchone()[0] == pytest.approx(0.73)

    upsert_assets(db, _target(), _extraction(), [0.2] * 1024, _quality(), sources)  # 재실행 멱등
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM aod_ai.content_fun_tag WHERE content_id = 7")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM aod_ai.content_semantic_profile WHERE content_id = 7")
        assert cur.fetchone() == (1,)
```
- [ ] 실행 → 실패 확인: `python -m pytest tests/test_upsert.py -q` → `ModuleNotFoundError: No module named 'aod_ai.pipeline.upsert'` (`1 error`).
- [ ] 최소 구현 — `aod_ai/pipeline/upsert.py`. `vector`(bare `list[float]`)를 `vector(1024)` 컬럼에 바인딩할 때는 psycopg3 기본 list→`float8[]` 어댑터로 새어나가 `column embedding is of type vector but expression is of type double precision[]` 오류가 날 수 있으므로 **`pgvector.Vector`로 명시 래핑**(conftest의 `register_vector`가 등록한 Vector 덤퍼가 확실히 적용됨):
```python
import json

from pgvector import Vector

from aod_ai.config import Settings

_PROFILE_SQL = """
INSERT INTO aod_ai.content_semantic_profile
  (content_id, domain, normalized_summary, profile_text, evidence,
   extraction_quality, source_count, content_hash, processed_at)
VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, now())
ON CONFLICT (content_id) DO UPDATE SET
  domain = EXCLUDED.domain,
  normalized_summary = EXCLUDED.normalized_summary,
  profile_text = EXCLUDED.profile_text,
  evidence = EXCLUDED.evidence,
  extraction_quality = EXCLUDED.extraction_quality,
  source_count = EXCLUDED.source_count,
  content_hash = EXCLUDED.content_hash,
  processed_at = now()
"""
_DICT_SQL = """
INSERT INTO aod_ai.fun_tag_dict (name, status)
VALUES (%s, 'proposed')
ON CONFLICT (name) DO NOTHING
"""
_FUN_TAG_SQL = """
INSERT INTO aod_ai.content_fun_tag (content_id, tag, tag_score, tag_confidence)
VALUES (%s, %s, %s, %s)
ON CONFLICT (content_id, tag) DO UPDATE SET
  tag_score = EXCLUDED.tag_score,
  tag_confidence = EXCLUDED.tag_confidence
"""
_EMB_SQL = """
INSERT INTO aod_ai.content_embedding (content_id, embedding, model, dim)
VALUES (%s, %s, %s, %s)
ON CONFLICT (content_id) DO UPDATE SET
  embedding = EXCLUDED.embedding, model = EXCLUDED.model, dim = EXCLUDED.dim
"""
_QUALITY_SQL = """
INSERT INTO aod_ai.content_quality_score
  (content_id, bayesian_score, platform_rank_score, review_count_score,
   recency_score, quality_popularity_score, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, now())
ON CONFLICT (content_id) DO UPDATE SET
  bayesian_score = EXCLUDED.bayesian_score,
  platform_rank_score = EXCLUDED.platform_rank_score,
  review_count_score = EXCLUDED.review_count_score,
  recency_score = EXCLUDED.recency_score,
  quality_popularity_score = EXCLUDED.quality_popularity_score,
  computed_at = now()
"""


def upsert_assets(conn, target, extraction, vector, quality, sources) -> None:
    evidence = json.dumps(
        [{"content": s.content, "url": s.url} for s in sources], ensure_ascii=False
    )
    model = Settings().EMBEDDING_MODEL
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                _PROFILE_SQL,
                (
                    target.content_id, target.domain, extraction.normalized_summary,
                    extraction.profile_text, evidence, extraction.extraction_quality,
                    len(sources), target.content_hash,
                ),
            )
            cur.execute(
                "DELETE FROM aod_ai.content_fun_tag WHERE content_id = %s",
                (target.content_id,),
            )
            for item in extraction.fun_tags:
                if item.is_new:
                    cur.execute(_DICT_SQL, (item.tag,))
                else:
                    cur.execute(
                        _FUN_TAG_SQL,
                        (target.content_id, item.tag, item.tag_score, item.tag_confidence),
                    )
            cur.execute(
                _EMB_SQL,
                (target.content_id, Vector(vector), model, len(vector)),  # 명시 Vector 래핑
            )
            cur.execute(
                _QUALITY_SQL,
                (
                    target.content_id, quality.bayesian_score, quality.platform_rank_score,
                    quality.review_count_score, quality.recency_score,
                    quality.quality_popularity_score,
                ),
            )
```
- [ ] 실행 → 통과 확인: `python -m pytest tests/test_upsert.py -q` → `1 passed`.
- [ ] 커밋:
```
git add aod_ai/pipeline/upsert.py tests/test_upsert.py && git commit -m "feat: idempotent upsert_assets writing profile/fun_tag/embedding(Vector-wrapped)/quality" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: 눈검수 덤프 `write_eyeball_dump`

**Files:**
- Create: `-AOD-All-of-Dopamine-AI/aod_ai/eyeball.py`
- Test: `-AOD-All-of-Dopamine-AI/tests/test_eyeball.py`

- [ ] 실패 테스트 작성 — `tests/test_eyeball.py` (Settings() 미사용 → env 불필요):
```python
from aod_ai.eyeball import write_eyeball_dump
from aod_ai.models import Extraction, FunTagItem, SelectedTarget


def test_write_eyeball_dump_contains_titles_tags_and_evidence(tmp_path):
    target = SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="전지적 독자 시점",
        original_title=None, synopsis="s", genres=["판타지", "현대"], content_hash="h",
    )
    extraction = Extraction(
        fun_tags=[
            FunTagItem(tag="사이다전개", tag_score=0.91, tag_confidence=0.82,
                       evidence="복수 장면이 통쾌하다", is_new=False),
            FunTagItem(tag="신규태그", tag_score=0.5, tag_confidence=0.4,
                       evidence="독특한 설정", is_new=True),
        ],
        normalized_summary="재난 생존 웹소설", profile_text="pt", extraction_quality=0.77,
    )
    out = tmp_path / "fun_tags_review.md"

    write_eyeball_dump([(target, extraction)], str(out))

    text = out.read_text(encoding="utf-8")
    assert "전지적 독자 시점" in text
    assert "사이다전개" in text
    assert "복수 장면이 통쾌하다" in text
    assert "(proposed/new)" in text
    assert "재난 생존 웹소설" in text
```
- [ ] 실행 → 실패 확인: `python -m pytest tests/test_eyeball.py -q` → `ModuleNotFoundError: No module named 'aod_ai.eyeball'` (`1 error`).
- [ ] 최소 구현 — `aod_ai/eyeball.py`:
```python
def write_eyeball_dump(records, out_path: str) -> None:
    lines: list[str] = ["# M1 fun_tag 눈검수 덤프", ""]
    for target, extraction in records:
        lines.append(f"## [{target.content_id}] {target.master_title}")
        lines.append(f"- 장르: {', '.join(target.genres)}")
        lines.append(f"- normalized_summary: {extraction.normalized_summary}")
        lines.append(f"- extraction_quality: {extraction.extraction_quality:.3f}")
        lines.append("- fun_tags:")
        for item in extraction.fun_tags:
            flag = " (proposed/new)" if item.is_new else ""
            lines.append(
                f"  - {item.tag}{flag} | score={item.tag_score:.2f} "
                f"conf={item.tag_confidence:.2f} | 근거: {item.evidence}"
            )
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
```
- [ ] 실행 → 통과 확인: `python -m pytest tests/test_eyeball.py -q` → `1 passed`.
- [ ] 커밋:
```
git add aod_ai/eyeball.py tests/test_eyeball.py && git commit -m "feat: eyeball fun_tag+evidence dump for M1 checkpoint" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: CLI 배선 `run.py` + 엔드투엔드 통합 테스트

**Files:**
- Create: `-AOD-All-of-Dopamine-AI/aod_ai/run.py`
- Test: `-AOD-All-of-Dopamine-AI/tests/test_run.py`

- [ ] 실패 테스트 작성 — `tests/test_run.py` (end-to-end는 `collect_reviews`·`upsert_assets`가 `Settings()`를 부르므로 `m1_env` 요청; `parse_args`는 불필요):
```python
from datetime import date

import aod_ai.run as run
from aod_ai.models import Extraction, FunTagItem, ReviewSource


def test_parse_args_defaults_and_values():
    args = run._parse_args(["--domain", "WEBNOVEL", "--limit", "200"])
    assert args.domain == "WEBNOVEL"
    assert args.limit == 200


def test_run_pipeline_end_to_end(db, fake_vane, fake_llm, fake_emb, m1_env):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, average_score, review_count, release_date) "
            "VALUES (1, 'WEBNOVEL', '전독시', 4.0, 10, %s)",
            (date.today(),),
        )
        cur.execute(
            "INSERT INTO public.webnovel_contents (content_id, genres, platforms) "
            "VALUES (1, %s, %s)",
            (["판타지"], []),
        )
        cur.execute("INSERT INTO aod_ai.fun_tag_dict (name, status) VALUES ('사이다전개', 'active')")

    vane = fake_vane([ReviewSource(content="리뷰원문", url="http://u/1")])
    extraction = Extraction(
        fun_tags=[
            FunTagItem(tag="사이다전개", tag_score=0.9, tag_confidence=0.8,
                       evidence="속시원", is_new=True),          # 실제로는 active → False 로 교정돼야
            FunTagItem(tag="신규발명태그", tag_score=0.5, tag_confidence=0.4,
                       evidence="신규", is_new=False),            # 실제로는 신규 → proposed 로 교정돼야
        ],
        normalized_summary="요약", profile_text="프로파일", extraction_quality=0.7,
    )
    llm = fake_llm(extraction)
    emb = fake_emb([0.05] * 1024)

    records = run.run_pipeline(db, "WEBNOVEL", 200, vane=vane, llm=llm, emb=emb)

    assert len(records) == 1
    with db.cursor() as cur:
        cur.execute("SELECT content_hash FROM aod_ai.content_semantic_profile WHERE content_id = 1")
        assert cur.fetchone()[0]
        cur.execute("SELECT tag FROM aod_ai.content_fun_tag WHERE content_id = 1 ORDER BY tag")
        assert [r[0] for r in cur.fetchall()] == ["사이다전개"]
        cur.execute("SELECT status FROM aod_ai.fun_tag_dict WHERE name = '신규발명태그'")
        assert cur.fetchone() == ("proposed",)
        cur.execute("SELECT dim FROM aod_ai.content_embedding WHERE content_id = 1")
        assert cur.fetchone() == (1024,)
        cur.execute(
            "SELECT quality_popularity_score FROM aod_ai.content_quality_score WHERE content_id = 1"
        )
        assert cur.fetchone()[0] is not None
```
- [ ] 실행 → 실패 확인: `python -m pytest tests/test_run.py -q` → `ModuleNotFoundError: No module named 'aod_ai.run'` (`1 error`).
- [ ] 최소 구현 — `aod_ai/run.py`:
```python
import argparse

from aod_ai import db
from aod_ai.clients.embedding import EmbeddingClient
from aod_ai.clients.llm import LlmClient
from aod_ai.clients.vane import VaneClient
from aod_ai.config import Settings
from aod_ai.eyeball import write_eyeball_dump
from aod_ai.pipeline.collect_reviews import collect_reviews
from aod_ai.pipeline.embed import build_profile_embedding
from aod_ai.pipeline.extract import extract_profile
from aod_ai.pipeline.quality import compute_quality
from aod_ai.pipeline.select_targets import select_targets
from aod_ai.pipeline.upsert import upsert_assets


def _load_active_tags(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM aod_ai.fun_tag_dict WHERE status = 'active'")
        return [r[0] for r in cur.fetchall()]


def run_pipeline(conn, domain, limit, *, vane, llm, emb):
    active_tags = _load_active_tags(conn)
    records = []
    for target in select_targets(conn, domain, limit):
        sources = collect_reviews(vane, target)
        extraction = extract_profile(llm, target, sources, active_tags)
        vector = build_profile_embedding(emb, extraction)
        quality = compute_quality(conn, target)
        upsert_assets(conn, target, extraction, vector, quality, sources)
        records.append((target, extraction))
    return records


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="aod_ai.run")
    p.add_argument("--domain", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--dump", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    s = Settings()
    conn = db.connect()
    vane = VaneClient(base_url=s.VANE_BASE_URL)
    llm = LlmClient(base_url=s.OPENAI_BASE_URL, api_key=s.OPENAI_API_KEY, model=s.LLM_MODEL)
    emb = EmbeddingClient(
        base_url=s.OPENAI_BASE_URL, api_key=s.OPENAI_API_KEY,
        model=s.EMBEDDING_MODEL, dim=s.EMBEDDING_DIM,
    )
    records = run_pipeline(conn, args.domain, args.limit, vane=vane, llm=llm, emb=emb)
    if args.dump:
        write_eyeball_dump(records, args.dump)
    print(f"processed {len(records)} contents")


if __name__ == "__main__":
    main()
```
- [ ] 실행 → 통과 확인: `python -m pytest tests/test_run.py -q` → `2 passed`.
- [ ] **M1 회귀 확인(스코프 지정)** — `pytest`를 인자 없이 돌리면 CONTRACTS §1.1의 M0 테스트(`test_config`/`test_migrate`/`test_models`/`test_clients_llm`/`test_clients_embedding`/`test_clients_vane`)까지 전부 수집되어 총계가 `10 + (M0 테스트 수)`가 되고, M0 테스트 수는 M0 산출물에 종속되어 이 plan에서 고정할 수 없다. 따라서 회귀 판정은 **M1 8개 파일로 스코프**해 결정한다(M1 테스트 함수 총 10개: T1=1, T2=1, T3=1, T4=2, T5=1, T6=1, T7=1, T8=2 = 10):
```
python -m pytest tests/test_select_targets.py tests/test_collect_reviews.py tests/test_extract.py tests/test_embed.py tests/test_quality.py tests/test_upsert.py tests/test_eyeball.py tests/test_run.py -q
```
→ `10 passed`. (선택) 전체 스위트 `python -m pytest -q`는 `10 + M0` passed가 나오며 M0가 깨지지 않았는지 함께 확인하되, 숫자 판정 기준은 위 M1 스코프의 `10 passed`로 한다.
- [ ] 커밋:
```
git add aod_ai/run.py tests/test_run.py && git commit -m "feat: run.py CLI wiring 6-stage pipeline + end-to-end integration test" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: 웹소설 샘플 풀가동 + fun_tag 눈검수 체크포인트

**Files:**
- (실행 전용, 소스 변경 없음)

- [ ] **선행 필수 — 시드 fun_tag 사전을 DB에 active로 동기화**: `run.py`의 `_load_active_tags`는 `status='active'` 행만 읽고, `upsert_assets`는 active 태그로 매칭된(=is_new=False) 태그만 `content_fun_tag`에 적재한다. `fun_tag_dict`에 active 행이 없으면 `active_tags=[]` → 모든 태그가 proposed로 빠져 `content_fun_tag`가 비고 핵심 가설(fun_tag 매칭)을 검증할 수 없다. 그러므로 배치 실행 전에 M0 산출물 `resources/seed_fun_tags.yaml`을 `aod_ai.fun_tag_dict`(status='active')로 로드/동기화한다:
```
python -m aod_ai.funtag_dict --sync
```
active 태그 수 검증:
```
python -c "import aod_ai.db as d; c=d.connect(); print(c.execute(\"SELECT count(*) FROM aod_ai.fun_tag_dict WHERE status='active'\").fetchone()[0])"
```
기대: `resources/seed_fun_tags.yaml`의 시드 항목 수와 동일한 **30~60 사이의 양수**(active > 0). 0이면 이후 단계 진행 금지(사전 동기화부터 재확인).
- [ ] 마이그레이션 적용 상태 확인 후 샘플 배치 실행 (실제 RDS 읽기 + 관리형 API + 로컬 Vane 기동 상태):
```
python -m aod_ai.run --domain WEBNOVEL --limit 200 --dump out/m1_fun_tags_review.md
```
기대 출력(마지막 줄): `processed <N> contents` (0 < N ≤ 200, 미처리·해시변경 콘텐츠 수).
- [ ] 자산 적재 검증:
```
python -c "import aod_ai.db as d; c=d.connect(); print([c.execute(q).fetchone()[0] for q in ['SELECT count(*) FROM aod_ai.content_semantic_profile','SELECT count(*) FROM aod_ai.content_fun_tag','SELECT count(*) FROM aod_ai.content_embedding','SELECT count(*) FROM aod_ai.content_quality_score','SELECT count(*) FROM aod_ai.fun_tag_dict WHERE status=%s' % chr(39)+'proposed'+chr(39)]])"
```
기대: 5개 카운트 모두 출력, `content_semantic_profile` == processed N, `content_embedding`·`content_quality_score` == N. `content_fun_tag > 0`은 **선행 시드 동기화(active > 0) 전제 하에서만 유효**하며, 시드 태그와 겹치는 추출이 있었음을 의미한다. `fun_tag_dict(proposed)`는 신규 제안 태그 수(≥0).
- [ ] **눈검수(핵심 가설 검증)**: `out/m1_fun_tags_review.md`를 열어 콘텐츠별 fun_tag가 evidence(원본 리뷰 표현)로 뒷받침되는지, `(proposed/new)` 태그가 시드 사전 확장 후보로 타당한지, `extraction_quality` 낮은 항목이 source 빈약 콘텐츠와 일치하는지 사람이 직접 확인. 부적합 태그 패턴 발견 시 `resources/seed_fun_tags.yaml`·프롬프트 조정은 후속 반복으로 회부.
- [ ] 체크포인트 통과 기준: 시드 동기화(active > 0)가 완료되고, 샘플이 오류 없이 완주(`processed N`, N>0)하며, `content_fun_tag > 0` 및 덤프의 fun_tag 다수가 evidence로 납득 가능하면 M1 완료로 판정.
