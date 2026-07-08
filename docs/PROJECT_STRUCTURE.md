# AOD 추천 시스템 — 파일 구조 안내

> 최종 갱신: 2026-07-08 (M1.1 재수집 완료 시점)
> 무엇이 어디에 있는지, fun_tag 품질을 좌우하는 파일이 무엇인지 정리한 문서.

## 저장소·브랜치 지도

| 저장소 | 브랜치 | 내용 | 상태 |
|---|---|---|---|
| `-AOD-All-of-Dopamine-AI` | `feature/m0-infra` | M0 인프라 | 리뷰 APPROVE, 푸시됨 |
| `-AOD-All-of-Dopamine-AI` | `feature/m1-content-intelligence` | M1 파이프라인 + 재수집 + 롤백 | 리뷰 APPROVE, 푸시됨 (m0에서 분기) |
| `-AOD-All-of-Dopamine-back` | `feature/m2-recommend-serving` | M2 홈 추천 서빙 (Java) | 리뷰 APPROVE, 푸시됨 |
| `-AOD-All-of-Dopamine-back` | `feature/transform-rule-registry` | 크롤러 Transform 하드코딩 제거 | 리뷰 APPROVE, 푸시됨 |

---

## 1. AI 레포 (`-AOD-All-of-Dopamine-AI`) — Python 배치 (Content Intelligence)

```
aod_ai/                        ★ 핵심 패키지
├── run.py                     CLI 진입점 — 6단계 파이프라인 배선
│                              (--domain --limit --recollect --delay --dump)
├── config.py                  설정 (.env를 읽는 pydantic Settings)
├── db.py                      RDS 연결 (psycopg3 + pgvector, search_path=aod_ai,public)
├── migrate.py                 aod_ai 스키마 마이그레이션 러너 (python -m aod_ai.migrate)
├── models.py                  데이터 모델 (Extraction, FunTagItem, SelectedTarget,
│                              QualityScore, ReviewSource, content_hash)
├── funtag_dict.py             시드 사전 DB 적재 (python -m aod_ai.funtag_dict --sync)
├── eyeball.py                 눈검수 md 덤프 생성기
│
├── clients/                   외부 서비스 클라이언트
│   ├── llm.py                 🔍 [품질레버① 프롬프트] fun_tag 추출 프롬프트+규칙 전부
│   │                          (_build_prompt: 한국어 강제·장르누출 금지·중복 금지·
│   │                           근거 인용·품질 정직 규칙 5개)
│   ├── vane.py                🔍 [품질레버② 수집] Vane(검색엔진) 호출.
│   │                          v1.12.2 대응: 프로바이더 자동등록, chatModel/embeddingModel
│   │                          필수 필드, dedupe(url), 소스 상한, 길이예산 2000자
│   └── embedding.py           OpenAI 호환 임베딩 (dimensions=1024 고정)
│
└── pipeline/                  6단계 — 콘텐츠 1건이 이 순서로 흐른다
    ├── select_targets.py      ① 대상 선별: 신규/해시변경분 + 재수집 모드
    │                          (select_recollect_targets: source_count=0 재처리)
    ├── collect_reviews.py     🔍 [품질레버②] 검색 쿼리 템플릿
    │                          '"{제목}" {도메인라벨} 리뷰 후기 재미' + 도메인라벨 맵
    ├── extract.py             ③ LLM 추출 (is_new을 active 사전 기준으로 권위 재계산)
    ├── embed.py               ④ profile_text 임베딩 (1024차원 가드)
    ├── quality.py             ⑤ 품질점수 (bayesian + platform_rank + review_count
    │                          + recency; 도메인 스코프 전역통계는 배치당 1회)
    └── upsert.py              ⑥ aod_ai 4테이블 멱등 upsert
                               (§9 canonical 태그 정규화, proposed 태그 격리)
```

```
resources/seed_fun_tags.yaml   🔍 시드 fun_tag 사전 45개 (name/aliases/description)
                               — 태그 정의가 곧 추출 정확도. 검수 후 확장/조정 대상
migrations/                    aod_ai 스키마 DDL (마이그레이션 러너가 순번 적용)
├── 001_schema_pgvector.sql    스키마 + vector 확장 + 이력 테이블
├── 002_asset_tables.sql       fun_tag_dict, content_semantic_profile,
│                              content_fun_tag, content_embedding, content_quality_score
├── 003_user_and_log_tables.sql  user_profile_cache, rec_impression, rec_event
└── 004_hnsw_indexes.sql       벡터 HNSW 인덱스 (cosine)

sql/
├── roles.sql                  권한격리 (DBA 1회 실행: public 읽기전용 + aod_ai 읽기쓰기)
└── rollback_aod_ai.sql        ★ 전체 롤백 — RDS에 추가된 것 전부 제거
                               (스키마·테이블·데이터·계정. public 무손상. 왕복 검증됨)

docker/vane/docker-compose.yml Vane v1.12.2 (SearXNG 내장 단일 이미지, VANE_REF 고정 빌드)
                               ※ 네이버 엔진은 컨테이너 내 /etc/searxng/settings.yml에서 활성화됨

out/                           눈검수 덤프 (git 미추적)
├── m1_review_rich_sources.md  ★ 우선 검수 대상 — 리뷰 소스 4건 이상 우량 32건
├── m1_recollect_review.md     재수집 배치 전체 190건
├── m1_fun_tags_review.md      1차 배치 전체 200건 (소스 부족 시절)
└── m1_sample5.md / _v2.md     파일럿 5건 — 프롬프트 개선 전/후 비교용

tests/                         18파일 · 37테스트 (단위=목, DB=testcontainers pgvector)
docs/superpowers/specs/        설계 스펙 (2026-06-28)
docs/superpowers/plans/        공유계약(00-contracts) + M0/M1/M2 구현 계획서
aod_reccomendation.md          원본 기획서 (fun_tag cross-domain 추천)
.env                           🔍 [품질레버③ 모델] LLM_MODEL=gpt-4o-mini 등 (git 미추적)
.env.example                   env 키 템플릿
```

### 실행 치트시트

```bash
python -m aod_ai.migrate                     # 스키마 마이그레이션
python -m aod_ai.funtag_dict --sync          # 시드 사전 적재
python -m aod_ai.run --domain WEBNOVEL --limit 200 --dump out/review.md          # 본 배치
python -m aod_ai.run --domain WEBNOVEL --limit 200 --recollect --delay 20 ...    # 소스없는 것 재수집
python -m pytest -q                          # 테스트 (Docker 필요; 첫 실행 flake 시 재실행)
```

---

## 2. 백엔드 레포 (`-AOD-All-of-Dopamine-back`) — 이번에 추가된 부분

### M2 추천 서빙 (`feature/m2-recommend-serving`, api 모듈)

```
-AOD-All-of-Dopamine-api/src/main/java/com/example/AOD/recommend/
├── RecommendController.java   GET /api/recommendations?location=home&page&size
├── RecommendService.java      오케스트레이션: 후보→피처→랭킹→후처리→로깅
│                              @Cacheable(homeRecommendations). M2 한계: 무조건 콜드스타트,
│                              page 미지원(M3), 캐시 evict 없음(M4)
├── candidate/
│   ├── AiAssetRepository.java aod_ai 네이티브 쿼리 (pgvector ANN <=>, fun_tag ANY,
│   │                          quality fallback, 하이드레이션)
│   ├── CandidateGenerator.java 소스 union→dedupe (콜드스타트는 quality만)
│   └── Candidate.java         서빙 내부 작업 타입
├── feature/FeatureCalculator.java  5피처 (funtag overlap·cosine·장르Jaccard·quality·recency)
├── rank/Ranker.java           home_score = 0.45/0.25/0.15/0.10/0.05 가중합
├── postprocess/PostProcessor.java  hard filter → soft 캡(20%/8%) → diversity(60%/50%)
├── log/RecLogWriter.java      rec_impression insert (score_breakdown jsonb)
└── dto/                       RecommendationItem, RecRequest, ScoreBreakdown, FunTag, QualityScore

src/main/resources/db/migration/V4__grant_aod_ai_access.sql
                               Flyway(신규 도입, baseline=3) — 서빙 계정에 aod_ai 읽기+로그쓰기 grant
                               ※ 백엔드 재배포 시 부팅에서 자동 실행 (아직 실 RDS 미적용)
src/test/java/...recommend/    단위·계약 테스트 + AiAssetIntegrationTest(Testcontainers pgvector)
```

### 크롤러 Transform 리팩토링 (`feature/transform-rule-registry`, crawler 모듈)

```
crawler/src/main/java/com/example/crawler/
├── service/RuleRegistry.java  ★ 신규 — rules/**/*.yml 자동발견·인덱싱.
│                              "새 플랫폼 추가 = yml 1개" 성립 (자바 switch 3중 중복 제거)
├── service/TransformEngine.java  yml 기반 raw→(master/domain/platform) 분리 엔진
│                              defaults(명시 기본값)·platformsFrom(platforms 병합) 지원
├── service/GenericDomainUpserter.java  리플렉션 필드 주입 + valueMap(값 치환표) 지원
├── util/FlexibleDateParser.java  ★ 신규 — 날짜 파서 통합 (영어 날짜 잠재버그 수정)
└── resources/rules/**.yml     플랫폼별 매핑 룰 — 선언 가능 섹션은
                               docs/3_TRANSFORM_ENGINE.md §4 표 참고
                               ⚠️ game/epic.yml은 미완성(휴면) — Epic 수집 전 완성 필요
```

---

## 3. fun_tag 품질 검토 — 세 레버 요약

| 레버 | 파일 | 검토 포인트 |
|---|---|---|
| ① 프롬프트 | `aod_ai/clients/llm.py` `_build_prompt` | 태그 "이름"만 넘기는 중 — 사전 description 포함 시 정확도 개선 여지 |
| ② 리뷰 수집 | `collect_reviews.py`(쿼리) + `vane.py` + 컨테이너 SearXNG 설정 | 스니펫이 진짜 리뷰인지 점검, 쿼리 키워드 변형 실험 |
| ③ 모델 | `.env` `LLM_MODEL` | gpt-4o-mini ↔ gpt-4o A/B로 모델 한계 vs 프롬프트 문제 분리 |

## 4. RDS에 있는 것 (참고)

- `public` 스키마: 기존 백엔드 테이블 — **이번 작업에서 수정된 적 없음 (읽기만)**
- `aod_ai` 스키마: 신규 테이블 8개 + 데이터 (프로파일 205 / fun_tag 281 / 임베딩 205 / 품질 205 / proposed 태그 20)
- 계정: `aod_ai`(배치용, public 읽기전용 + aod_ai 읽기쓰기), `aod_public_ro`(롤)
- 전부 제거하려면: `psql "<dba-url>" -f sql/rollback_aod_ai.sql`
