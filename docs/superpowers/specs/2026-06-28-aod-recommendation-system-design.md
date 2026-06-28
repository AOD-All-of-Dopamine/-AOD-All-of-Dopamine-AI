# AOD 개인화 추천 시스템 — 구현 설계 (Design Spec)

- 작성일: 2026-06-28
- 상태: 승인 대기 (브레인스토밍 산출물)
- 원본 기획: `aod_reccomendation.md` (fun_tag 기반 cross-domain 추천)
- 범위: 기획서를 **현재 당장 서비스 가능한 형태**로 구현하기 위한 첫 슬라이스 설계

---

## 0. 이 문서의 목적

`aod_reccomendation.md`는 잘 짜인 기획이지만 7개의 독립 서브시스템(리뷰수집·LLM추출·임베딩/색인·품질점수·서빙·유저프로파일·로깅)을 포함한 큰 범위다. 이 문서는 그 기획을 **실제로 구현 가능한 아키텍처와 결정으로 확정**하고, **수직 증분 마일스톤**으로 쪼갠 결과다.

기획서가 답을 비워둔 부분(모델 호스팅, Vane 정체, 데이터 접근, 벡터 저장소, 사전 구성)을 이 문서에서 확정한다.

---

## 1. 확정된 핵심 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 첫 슬라이스 범위 | **fun_tag 풀 파이프라인** (기획대로) | 핵심 가설(fun_tag)을 바로 검증 |
| 모델 호스팅 | **하이브리드** — 배치는 관리형 API(Qwen LLM + Qwen Embedding), 서빙은 CPU Python 서비스 | 무거운 모델은 전부 오프라인 배치에서만 쓰임. 서빙은 모델 호출 0 |
| 리뷰 수집(Vane) | Perplexica 자가호스팅을 **리뷰 retrieval 레이어**로 사용 | `sources[]` 원본 스니펫을 evidence로 확보 |
| 데이터 접근 | **같은 RDS, 단 스키마+권한으로 격리** — `aod_ai` 스키마(읽기쓰기) + `public` 읽기전용 role | 두번째 DB 없이 스키마/마이그레이션 충돌 회피 |
| 벡터 저장소 | **pgvector** (`aod_ai`) + 서빙은 **인프로세스 ANN** | 수만 건 규모엔 충분, 인덱스 파일 lifecycle 제거 |
| fun_tag 사전 | **하이브리드** — 시드 30~60개 + Qwen 제안 → 검수 후 확장 | 매칭 일관성과 확장성의 균형 |

### 1.1 핵심 통찰 — 무거운 모델은 배치에서만

| 작업 | 모델 | 실행 시점 | 빈도 |
|---|---|---|---|
| fun_tag 추출 | Qwen LLM | 배치(콘텐츠 등록/변경 시) | 콘텐츠당 1회 |
| profile_text 임베딩 | Qwen Embedding | 배치 | 콘텐츠당 1회 |
| **추천 서빙** | **모델 호출 없음** | 요청 시 | 매 요청 |

서빙 경로는 인프로세스 ANN 검색 + 점수 산술뿐이라 **GPU가 전혀 필요 없다.** 유저 프로파일 벡터도 미리 만들어둔 콘텐츠 임베딩의 가중평균이라 요청 때 모델을 부르지 않는다.

---

## 2. 시스템 아키텍처

3개 구성요소 + 관리형 API + 공유 RDS.

```
                    ┌─────────────── 관리형 API (DashScope 등) ───────────────┐
                    │           Qwen LLM  ·  Qwen Embedding                    │
                    └────────▲──────────────────▲─────────────────▲───────────┘
                             │                  │                 │
[배치/오프라인]              │                  │                 │
  ┌──────────┐   ┌───────────┴───┐   ┌──────────┴────┐   ┌────────┴────────┐
  │ Vane +   │──▶│ ReviewCollect │──▶│ Qwen Extract  │──▶│ Embed + Quality │
  │ SearXNG  │   │ (sources[])   │   │ (fun_tags..)  │   │ Score           │
  └──────────┘   └───────────────┘   └───────────────┘   └────────┬────────┘
   (배치박스 docker)                                               │ upsert
                                                                   ▼
                              ┌──────────────── RDS PostgreSQL ─────────────┐
   공유 읽기(read-only) ◀─────│ public: contents, *_contents, platform_data,│
   Content·좋아요·            │         bookmarks, content_likes, reviews    │
   북마크·리뷰               │ aod_ai: fun_tag_dict, content_semantic_     │
                              │         profile, content_fun_tag,           │
                              │         content_embedding(pgvector),        │
                              │         content_quality_score,              │
                              │         user_profile_cache,                 │
                              │         rec_impression, rec_event           │
                              └───────────────▲─────────────────────────────┘
[온라인/서빙]                                  │ read assets + write logs
  ┌─────────────────────────────────────────┐ │
  │ AI Serving API (FastAPI, CPU)            │─┘
  │  candidate gen(인프로세스 ANN+fun_tag SQL)│◀──── 프론트(home-page / work-detail)
  │  → feature → ranking(home/related)       │       또는 백엔드 프록시
  │  → post-processing → Top-N + 로깅        │
  └─────────────────────────────────────────┘
```

### 2.1 구성요소

| 구성요소 | 기술 | 역할 |
|---|---|---|
| **배치 워커** | Python | Vane 호출→Qwen 추출→임베딩→품질점수→`aod_ai` upsert. 신규/변경 콘텐츠 또는 주기 실행 |
| **서빙 API** | FastAPI (CPU only) | 요청 시 후보생성→스코어링→랭킹→후처리→Top-N. 모델 호출 없음 |
| **Vane + SearXNG** | Docker (Next.js + SearXNG) | 배치에서만 쓰는 리뷰 retrieval. 내부 LLM/임베딩은 관리형 API에 연결 |

배치 워커와 서빙은 **한 코드베이스, 두 실행모드**로 둔다(공유 모델/스키마/스코어링 로직 재사용).

### 2.2 데이터 접근 원칙

> **읽기는 공유, 쓰기는 스키마 격리, 서빙은 인프로세스 ANN으로 DB 부하 최소화.**

- AI 자산(임베딩·프로파일·점수·로그)은 **전부 `aod_ai` 스키마에만** 기록 → 백엔드 `public`/Flyway와 충돌 0.
- AI 계정: `public`에 **읽기전용 role**, `aod_ai`에 읽기쓰기 → 백엔드 테이블 실수로 못 건드림.
- pgvector는 **`aod_ai`에만** 설치.
- 배치 읽기는 **off-peak·throttle** (AOD는 크롤/변환을 새벽에 돌리므로 그 시간대 활용).
- 유저 상호작용(bookmarks·content_likes·reviews)은 **복제하지 않고** `public`에서 읽어 `user_profile_cache`만 갱신.

---

## 3. 데이터 모델 (`aod_ai` 스키마)

| 테이블 | 역할 | 핵심 컬럼 |
|---|---|---|
| `fun_tag_dict` | 통제어휘 사전(하이브리드) | id, name, aliases[], description, status(active/proposed/rejected), created_at |
| `content_semantic_profile` | 콘텐츠별 추출 결과 | content_id(PK), domain, normalized_summary, profile_text, evidence(jsonb), extraction_quality, source_count, content_hash, processed_at |
| `content_fun_tag` | 콘텐츠×태그 (정규화) | content_id, tag, tag_score, tag_confidence — 인덱스: (tag), (content_id) |
| `content_embedding` | profile_text 벡터 | content_id(PK), embedding `vector(dim)`, model, dim — HNSW 인덱스 |
| `content_quality_score` | 품질·인기도 | content_id(PK), bayesian_score, platform_rank_score, review_count_score, recency_score, quality_popularity_score, computed_at |
| `user_profile_cache` | 유저 취향 캐시 | user_id(PK), fun_tag_profile(jsonb), negative_fun_tag(jsonb), profile_vector `vector(dim)`, positive_count, updated_at |
| `rec_impression` | 노출된 추천 아이템 | request_id, user_id, location, selected_content_id?, content_id, candidate_source, rank_position, score_breakdown(jsonb), served_at |
| `rec_event` | 유저 행동 | id, request_id?, user_id, content_id, event_type(click/long_view/hide…), value, created_at |

> `content_id`는 `public.contents.content_id`를 논리적으로 참조하지만 **물리 FK는 걸지 않는다**(스키마 격리 유지, 백엔드 마이그레이션과 결합 회피).

---

## 4. 배치 파이프라인 (Content Intelligence)

콘텐츠 1건당 한 번(또는 변경 시) 실행. **여기서만 LLM/임베딩 비용 발생.**

```
① 대상 선별 ─▶ ② Vane 리뷰수집 ─▶ ③ Qwen 추출 ─▶ ④ 임베딩 ─▶ ⑤ 품질점수 ─▶ ⑥ aod_ai upsert
```

| 단계 | 처리 | 핵심 포인트 |
|---|---|---|
| **① 대상 선별** | `public.contents` 읽어 프로파일 없음 / `content_hash` 변경분만 | 재처리 최소화 = 비용 통제. hash = title+synopsis+genre 등 |
| **② Vane 리뷰수집** | `POST /api/search`, `sources:["web","discussions"]`, query=`"{제목}" {도메인} 리뷰 후기 재미`, `systemInstructions`로 "독자 반응·재미요소 위주" 유도, `stream:false` | `sources[].content`(원본 스니펫)+url을 **evidence corpus**로 수집(상한 N개·dedupe·길이예산). 결과 빈약 시 source_count 낮게 |
| **③ Qwen 추출** | 입력=(메타데이터 + evidence 스니펫 + **활성 fun_tag 사전**) → **JSON 강제 출력(schema)** | fun_tags는 사전 내 선택 + 신규는 `proposed`로 분리. tag_score/tag_confidence/evidence/normalized_summary/profile_text/extraction_quality 산출 |
| **④ 임베딩** | `profile_text` → Qwen-Embedding API → `content_embedding`(pgvector) | HNSW 인덱스 유지 |
| **⑤ 품질점수** | bayesian(R,v=리뷰평점·수; C,m=전역) + platform_rank(백엔드 `ExternalRanking` 테이블) + review_count + recency | `content_quality_score` 저장. 실제 테이블/컬럼명은 구현 plan에서 확정 |
| **⑥ upsert** | 위 결과 전부 `aod_ai`에 기록, content_hash·processed_at 세팅 | 멱등·재시도·throttle |

### 4.1 Vane 사용 원칙

- Vane 응답 `message`(LLM 합성 답변)에 의존하지 않고 **`sources[].content`(원본 스니펫)을 evidence로** 사용 → 이중 요약으로 실제 리뷰 표현이 뭉개지는 것 방지.
- Vane 내부 `chatModel`/`embeddingModel`은 관리형 API provider로 설정 → "Vane 내부 LLM(검색요약) + Qwen(구조화 추출)"의 2단 구조.
- SearXNG는 JSON 포맷 활성화 필요(`search.formats`에 `json`).

### 4.2 신규 태그 흐름 (하이브리드 사전)

Qwen이 사전에 없는 태그를 제안하면 `fun_tag_dict(status=proposed)`로 **적재만** 하고, **검수 통과(status=active) 전엔 매칭/후보생성에 사용하지 않는다.**

### 4.3 tag_score / tag_confidence 사용 원칙 (기획 §4.5 계승)

- `tag_score`는 모델 추정값 → **절대값 신뢰 금지, 후보 간 상대 비교에만 사용.**
- `tag_confidence` 낮으면 ranking 반영 강도 낮추거나 제외.
- evidence 부족한 태그는 candidate source로 사용하지 않음.
- source 부족 콘텐츠는 `extraction_quality` 낮게 → 상위 노출 보수적.

---

## 5. 서빙 파이프라인 (온라인, 모델호출 0)

요청: `{user_id, location: home|related, selected_content_id?}`

```
컨텍스트 구성 ─▶ 후보생성(500~1000) ─▶ 피처계산 ─▶ 랭킹 ─▶ 후처리 ─▶ Top-N + 로깅
```

### 5.1 후보 생성 (union → dedupe)

| 소스 | home | related | 방식 | 후보수 |
|---|---|---|---|---|
| user profile vector ANN | ✓ | ✓ | 인프로세스 ANN | ~300 |
| user fun_tag match | ✓ | ✓ | `content_fun_tag` SQL, 유저 top태그 | ~300 |
| selected content ANN | | ✓ | 클릭 콘텐츠 유사 | ~300 |
| selected fun_tag match | | ✓ | 클릭 콘텐츠 태그 매칭 | ~300 |
| quality/popularity | ✓ | ✓ | fallback | ~100 |
| metadata similarity | ✓ | ✓ | 장르·creator·platform | ~100 |

중복 제거 후 약 500~1000개 pool.

### 5.2 피처 (5종)

| 피처 | 계산 |
|---|---|
| `fun_tag_match_score` | 유저(또는 선택콘텐츠) fun_tag 프로파일과 후보 fun_tags의 가중 overlap (tag_score×tag_confidence 가중, 저신뢰 감쇠) |
| `profile_similarity_score` | 유저/선택콘텐츠 벡터와 후보 벡터의 cosine |
| `metadata_match_score` | genre overlap, creator, platform, domain |
| `quality_popularity_score` | `content_quality_score` 조회 |
| `recency_score` | 신규/최근 업데이트 가산 |

### 5.3 랭킹 (가중치는 설정값, 초기 가설)

```
home_score = 0.45·funtag + 0.25·profile_sim + 0.15·quality + 0.10·metadata + 0.05·recency

related_score = 0.4·home_score + 0.6·selected_similarity
  selected_similarity = b1·sel_funtag + b2·sel_profile_sim + b3·metadata + b4·quality
```

가중치는 확정값이 아니라 초기 가설 — 검수와 로그 분석으로 조정.

### 5.4 후처리 (순서대로)

1. **Hard filter** — policy/age_rating, 명시적 hide, 접근불가 제외
2. **Soft penalty/boost** (상한 둠):
   - `negative_preference_penalty` ≤ final score의 20% (negative fun_tag overlap + negative profile sim + negative genre overlap)
   - `already_seen_penalty` — home 강하게(또는 제외), related 약하게
   - `recency_boost` ≤ 8%
3. **Diversity re-ranking** — Top20 내 한 domain ≤60%, 한 platform ≤50%, 승격 후보는 Top100 평균 이상 또는 1위의 70% 이상, 조정 대상은 Top100 내에서만
4. **Top-N serving** — 초기 20 + 페이지네이션

```
candidate pool 500~1000 → ranking Top 100 → post-processing Top 50 → 초기 노출 Top 20 → 스크롤/더보기
```

각 노출은 `rec_impression`에 score_breakdown·candidate_source·rank_position과 함께 기록.

---

## 6. 유저 프로파일 & 콜드스타트

### 6.1 프로파일 구축

**`user_fun_tag_profile`**: 긍정 반응 콘텐츠들의 fun_tag를 **행동강도 × 최근성 감쇠 × (tag_score·tag_confidence)** 로 가중합산 후 정규화.

| 행동 | 가중 |
|---|---|
| impression | ~0 |
| click | 낮음 |
| long_view | 중간 |
| bookmark | 높음 |
| like / 높은 평점 | 매우 높음 |
| hide / dislike | negative 셋으로 분리 |

**`user_profile_vector`**: 긍정 콘텐츠 임베딩의 가중평균(행동·최근성). negative 셋의 태그/벡터는 `negative_preference_penalty`에 사용.

**갱신**: `user_profile_cache`에 저장, 신규 상호작용 시 증분 또는 야간 배치. `positive_count`로 콜드스타트 분기.

### 6.2 콜드스타트

| 기록 수 | 전략 |
|---|---|
| 0 | quality/popularity + platform ranking + **온보딩 취향**(`onboarding-page.tsx` 초기 선호 → fun_tag 프로파일 시드) |
| 1~2 | 최근 클릭 콘텐츠와 유사 fun_tag/profile |
| 3+ | full 유저 프로파일 + 벡터 |
| 충분 | 개인화 weight 강화 |

---

## 7. 로깅

| 테이블 | 1행 단위 | 핵심 필드 |
|---|---|---|
| `rec_impression` | 노출된 추천 아이템 | request_id, user_id, location, selected_content_id?, content_id, candidate_source, rank_position, score_breakdown(jsonb), served_at |
| `rec_event` | 유저 행동 | request_id?, user_id, content_id, event_type(click/long_view/hide…), value, created_at |

- **중복 방지**: bookmark·like·review는 `public`에 이미 있으므로 `rec_event`엔 **click/long_view/hide만** 적재하고, bookmark/like는 백엔드 테이블 조인으로 분석.
- **출처 귀속**: 추천면에서 발생한 행동에 프론트가 `request_id`를 실어 보내야 source/score에 귀속 가능.
- **용도**: 오프라인 지표(CTR, long_view_rate, bookmark_rate, hide_rate, cross_domain_click_rate, candidate_source_ctr, score_bucket_performance) + weight 튜닝.

---

## 8. 평가 (기획 §8 계승)

- **Offline 검수**(서비스 전): fun_tag 적합성, cross-domain 납득성, 홈/상세 품질, 품질 보정, 다양성, 비선호 반영, cold-start 안정성을 샘플로 직접 검수.
- **Online 지표**(서비스 후): §7 로그 기반 CTR·long_view_rate·hide_rate·candidate_source_ctr·score_bucket_performance.

---

## 9. 구현 순서 (마일스톤)

풀 파이프라인이라도 **수직 증분 + 체크포인트**로 쪼갠다. 각 마일스톤은 별도 plan/구현 사이클.

| | 내용 | 체크포인트 |
|---|---|---|
| **M0** | Python 서비스 골격(batch/serve 2모드), `aod_ai` 마이그레이션 + 읽기전용 role, 시크릿/관리형API 설정, Vane+SearXNG 도커, **시드 fun_tag 사전(30~60) 작성** | 인프라 기동 |
| **M1** | 샘플(웹소설 100~200건) Content Intelligence 풀가동 | **fun_tag 품질 눈검수 ← 핵심 가설 검증(가장 싸게)** |
| **M2** | 홈 서빙(후보→피처→랭킹→기본 후처리), 테스트 유저 프로파일 | 추천결과 오프라인 검수 |
| **M3** | 유저 프로파일 구축 + 콜드스타트 + 상세(related) 추천 | |
| **M4** | 로깅 + 오프라인 평가, 프론트(home·work-detail) 연동 | 엔드투엔드 |
| **M5** | 전체 카탈로그 배치 확장 + 로그 기반 weight 튜닝 | 운영 |

---

## 10. 열린 이슈 / 사전 확인 (구현 전 해소)

| # | 이슈 | 영향 | 비고 |
|---|---|---|---|
| 1 | RDS PostgreSQL **버전 15+** 및 pgvector 설치 가능 여부 | 벡터 저장소 전체 | 매니지드 정책 확인 |
| 2 | Vane+SearXNG **배치 박스 호스팅 위치/비용** | 리뷰수집 인프라 | 별도 박스 또는 배치 때만 기동 |
| 3 | 관리형 API **제공자 확정(DashScope 등)·단가·rate limit** | 배치 비용·속도 | M0 전 |
| 4 | **임베딩 차원**(8B는 ~3.5~4k dim) | pgvector 인덱스·메모리·인프로세스 ANN | 차원 클수록 메모리↑ — 필요시 경량 임베딩 검토 |
| 5 | **age_rating 데이터** 백엔드 Content 보유 여부 | hard filter | 없으면 해당 필터 보류 |
| 6 | 시드 fun_tag 사전 **초기 30~60개 확정 주체** | 추출 품질 | M0 산출물 |
| 7 | 프론트가 추천 행동에 **request_id 전달** 가능 여부 | 로그 출처 귀속 | M4 프론트 연동 |

---

## 11. 비범위 (이번 슬라이스 제외)

- 학습 기반 ranking 모델(LTR 등) — 로그 충분히 쌓인 후 고도화 단계
- 실시간 임베딩/온라인 학습
- CDC/논리복제 기반 완전 DB 분리 — 규모 커지면 검토
- A/B 테스트 인프라
