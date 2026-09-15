# 추천 탭 설계 — 화면 · 로그 · 백엔드 · 프론트 · 추천 서비스

- 상태: **v2** — v1 초안을 코드 대조 리뷰(23건)로 고친 판. 리뷰 기록은 부록 B
- 작성: 2026-09-15
- 관련 문서: `INDUSTRY_COMPARISON.md`(유튜브·당근 사례) · `TRANSITION_ROADMAP.md`(로그 기반 전환 단계) · `crossdomain/DESIGN.md`(플랫폼 혼합)
- 대상 시스템: 프론트 `allofdophamin.com`(Vite + React SPA) · 백엔드 `-AOD-All-of-Dopamine-back`(Spring Boot 3.4, Java 17, PostgreSQL) · 추천기 이 리포(Python, 플랫폼 4개)
- **이 문서가 로그 테이블·이벤트 이름·지표의 기준이다.** `TRANSITION_ROADMAP.md` §8 은 근거 조사이고, 이름은 이 문서를 따른다.

---

## 1. 목표와 범위

### 1-1. 목표
1. 로그인 사용자에게 **"추천" 탭**에서 취향 기반 추천을 보여준다.
2. 좋아요·싫어요가 바뀌면 **다음 요청에 바로** 반영한다.
3. 추천 탭보다 **로그를 먼저** 켜서, 나중에 A/B·인터리빙·학습에 쓸 수 있는 데이터를 쌓는다.

### 1-2. 범위
- **포함**: 화면 · API · 로그 · 백엔드/프론트/추천 서비스 변경 · 출시 순서 · 테스트 · 운영
- **알고리즘은 바꾸지 않는다.** 단, 두 가지는 허용한다
  - **순서를 바꾸지 않는 계측**: 점수 인자를 응답에 싣는 것. 평가 목록과 같은지 동일성 테스트로 확인한다(§10)
  - **입력 검증**: 코퍼스 밖·빈 시드를 서비스 어댑터에서 거르는 것
- **포함하지 않음**: 웹툰을 넣은 플랫폼 혼합(M6 확장 — 별건 사전등록), 학습 모델, A/B 운영, 상세 페이지 "비슷한 작품"

### 1-3. 현재 상태 (코드로 확인한 사실)
| 영역 | 사실 |
|---|---|
| 프론트 라우트 | `home` · `explore` · `ranking` · `new` · `collections`(+`new`·`:id`·`:id/edit`) · `profile`(+`likes`·`bookmarks`·`reviews`) · `work/:id` · `review/:id` · `search` · `login` · `signup` · `onboarding` · `internal/ranking` |
| 메뉴 | 홈 · 탐색 · 랭킹 · 신작 · 컬렉션 · 프로필 — 이미 6개 |
| 레이아웃 | 홈은 `max-w-[1280px]`(데스크톱 폭), 온보딩·리뷰·컬렉션 폼만 `max-w-[720px]` |
| 온보딩 | 장르 18개 중 최대 5개 선택 → **`console.log` 만 하고 저장하지 않음**. `/onboarding` 으로 가는 링크 0곳. 백엔드 `SignUpRequest.preferredGenres` 도 어디서도 쓰지 않음 |
| 상세 페이지 | "볼 수 있는 곳" = `platformInfo.url` 외부 링크(`target="_blank"`) (`WorkApiService.java:427-441`) |
| 프론트 계측 | `IntersectionObserver` · `sendBeacon` · 분석 SDK 없음. `localStorage` 키는 `token` 하나. API 는 다른 오리진 `https://api.allofdophamin.com` |
| 좋아요·싫어요·북마크 | `POST /api/works/{id}/like·dislike·bookmark` **토글**. 같은 타입 재호출 = 행 삭제, 좋아요↔싫어요 = 같은 행 덮어씀 (`LikeService.java:57-68`, `BookmarkService.java:50`). 예외는 전부 400 |
| 카드 DTO | `WorkSummaryDTO` (id · domain · title · thumbnail · score · genres · platforms · creator …) |
| 추천 API | main 에 없음. `feature/m2-recommend-serving` 브랜치에 뼈대 — 익명 콜드스타트만, 로그는 요청 스레드에서 작품마다 동기 INSERT, `V4` 번호가 main 과 충돌 |
| 인증·보안 | JWT(subject = username), 컨트롤러가 헤더 직접 파싱. `SecurityConfig` 전부 `permitAll`. 회원 탈퇴 API 없음 |
| HTTP 클라이언트 | `RestTemplate` 빈에 타임아웃 없음 (`RestTemplateConfig.java:15-16`) |
| DB 마이그레이션 | api 모듈 Flyway V1·V3~V6 (V7 은 `perf/works-review-count-index` 가 선점). 크롤러 모듈에도 V2·V3 파일이 있으나 크롤러는 Flyway 를 쓰지 않는다 |
| 추천기 — 공통 | 네 플랫폼 모두 `next_page` 와 주도 시드(`dominant_seed`) 계산이 있고 후처리에 난수가 없어 **같은 입력이면 같은 결과** |
| 추천기 — 프로세스 | 네 플랫폼 모두 최상위 패키지가 `src` 이고 `os.chdir`·`AOD_ARTIFACTS` 환경변수를 쓴다 → **한 프로세스에 둘 이상 적재할 수 없다** (시험대가 플랫폼마다 프로세스를 따로 띄우는 이유) |
| 추천기 — 시드 검증 | 코퍼스 밖 시드: Steam·TMDB·웹소설 **예외**, 웹툰만 조용히 제외. 빈 시드: Steam·TMDB `ValueError` |
| 추천기 — 싫어요 입력 | Steam `disliked_appids`(유사 감점) · 웹툰 `disliked_ids`(제외 + 같은 시리즈 제외, 감점 0) · **TMDB·웹소설은 인자 없음** |
| 추천기 — 점수 | TMDB 는 보정을 곱셈으로 누적해 `final_score` 만 남긴다. 웹툰·웹소설은 `seed_similarity`·`final_score` 뿐. Steam 만 일부 중간 컬럼 |
| 플랫폼 혼합 M6 | `mix.py` `ORDER=("steam","tmdb","wn")` — **웹툰 없음**. 웹소설 화수는 카드 문자열을 정규식으로 읽음 |
| ID | `platform_data` TMDB 는 숫자 ID + `TMDB_MOVIE/TV`, 코퍼스 키는 `movie_{id}`, TMDB `next_page` 입력은 코퍼스 **행 번호**. 웹소설 코퍼스는 NaverSeries 만 |
| 측정 | 지연 중앙 91~584ms 는 시험대에서 `recommend(k=20)` 을 한 번에 하나씩 잰 값(Steam 첫 요청 804ms). 긴 seen · 동시 부하는 **미측정** |

---

## 2. 화면 설계 — 추천 탭

### 2-1. 진입
- 경로 `/for-you`. **메뉴가 이미 6개**라 7번째로 넣지 않는다:
  - **모바일**: 홈 화면 상단에 `홈 | 추천` 세그먼트
  - **데스크톱**: 상단 메뉴에 "추천" 추가
- **비로그인**: "로그인하면 취향 추천을 볼 수 있어요" 배너 + 탭별 랭킹 대체 목록(§2-7). 추천 요청은 하지 않는다.
- **로그인 · 시드 0개**: 온보딩(§2-5) 안내 카드.

### 2-2. 레이아웃
```
┌───────────────────────────────────────────────────────┐
│ 추천                                                   │
│ [전체] [영화] [시리즈] [게임] [웹툰] [웹소설]               │  ← 칩 (가로 스크롤)
├───────────────────────────────────────────────────────┤
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐              │  ← 데스크톱 4열 · 모바일 1열
│ │ 카드   │ │ 카드   │ │ 카드   │ │ 카드   │              │
│ └────────┘ └────────┘ └────────┘ └────────┘              │
│   카드 = 기존 카드 + "코코를 좋아해서" + [♡] [⋯]            │
│                    [ 더 보기 ]                          │
└───────────────────────────────────────────────────────┘
```
- **칩**
  - 전체 = Steam·TMDB·웹소설 M6 혼합 (평가된 규칙 그대로. **웹툰은 전체 탭에 넣지 않는다** — M6 확장은 별건)
  - 영화·시리즈 = TMDB `media` 분리
  - 게임·웹툰·웹소설 = 플랫폼 단독
- **읽기 순서는 행 우선**(왼쪽→오른쪽, 위→아래). 추천기 순위 1위가 왼쪽 위 — 넷플릭스는 가장 강한 추천을 행의 왼쪽에 둔다.
- 넷플릭스식 가로 행("○○를 봤기 때문에" 행 여러 개)은 행 선택·행 간 중복 제거가 따로 필요해 후속으로 둔다.
- 카드 = 기존 목록 카드 + **이유 한 줄** + **♡** + **더보기 메뉴**.

### 2-3. 추천 이유
- 추천기가 계산한 주도 시드로 서버가 문구를 만든다. 시드 출처별 문구:
  | 시드 출처 | 문구 |
  |---|---|
  | 좋아요 | `{제목}을(를) 좋아해서` |
  | 북마크 | `{제목}을(를) 담아둬서` |
  | 평점 4 이상 리뷰 | `{제목}을(를) 높게 평가해서` |
- 조사: 제목 마지막 글자가 한글이면 받침으로 `을`/`를`, 그 밖(영문·숫자·기호)은 `을(를)`.
- 탐색 칸(`TRANSITION_ROADMAP.md` §3, 출시 뒤 1단계)은 `새로운 발견`. 랭킹 대체 목록은 이유를 표시하지 않는다.

### 2-4. 피드백 조작
| 조작 | 위치 | 즉시 화면 | 서버 |
|---|---|---|---|
| 좋아요 | 카드 ♡ | 채워진 ♡ | `PUT /api/works/{id}/reaction {state: LIKE}` |
| 싫어요 | 더보기 메뉴 | **카드 제거 + "되돌리기" 토스트(5초)** | `PUT … {state: DISLIKE}` → 응답의 `previousState` 보관 |
| 되돌리기 | 토스트 | 카드 복원 | `PUT … {state: previousState}` — 좋아요였던 작품은 좋아요로 돌아간다 |
| 관심 없음 | 더보기 메뉴 | 카드 제거 + 되돌리기 | `PUT /api/recommendations/not-interested/{id}` · 되돌리기 `DELETE` |
| 북마크 | 더보기 메뉴 | 표시 | 기존 토글 API |

- **토글이 아니라 상태 지정 API** 를 새로 쓴다: 토글은 멱등이 아니라 재시도·연타에 뒤집히고, 좋아요 → 싫어요 → 되돌리기(=싫어요 토글)가 좋아요를 지워 버린다. 기존 토글 API 는 다른 화면용으로 남긴다.
- **싫어요와 관심 없음은 다른 신호다**
  - 싫어요 = 작품 평가. 모든 추천면에서 빠지고 플랫폼별 규칙(§8-3)으로 제외·감점. 검색·탐색에서는 계속 보인다(넷플릭스: 엄지 내림 작품은 홈에서 빠지지만 검색으로 찾을 수 있다)
  - 관심 없음 = 노출 조절. **추천 탭(전 칩)에서 90일 동안 제외만** 하고 감점·시리즈 제외는 하지 않는다. 작품 평가·좋아요 목록에는 영향 없음
  - 유튜브는 싫어요 · 관심 없음("이미 봤어요 / 마음에 안 들어요") · 채널 추천 안 함을 따로 받는다
- 가벼운 버튼 — 넷플릭스는 별점을 엄지로 바꾼 시험에서 평가 활동이 200% 늘었다.
- 피드백 직후 화면 반응(즉시 제거·되돌리기)은 세 회사 모두 공개 문서에 없다 — **우리 설계 선택**이다.
- 당근은 숨기기를 "해당 글만"에서 "비슷한 글까지 줄이기"로 고도화한 뒤 숨기기 사용이 약 40% 줄었다. 우리 싫어요 유사 감점(웹툰 T-11 기각, 후속 후보)과 같은 방향이다.

### 2-5. 콜드스타트 — 온보딩을 "작품 고르기"로
- 현재 온보딩(장르, 저장 안 함)을 **"좋아하는 작품을 골라주세요"** 로 바꾼다. 추천기는 장르가 아니라 작품을 시드로 받는다.
  - 플랫폼 칩 → 그 플랫폼 랭킹 상위 그리드 + 검색
  - **최소 3개, 플랫폼마다 2개 이상 권장** 안내 — 플랫폼당 시드 1개면 M6 할당이 절반이 되고 TMDB 는 시드 1~2개에서 깊은 페이지가 무너진다(EXPERIMENT_LOG §103)
  - 저장 = `PUT /api/works/{id}/reaction {state: LIKE, source: onboarding}`. 고른 작품은 진짜 좋아요로 보고 **좋아요 목록에도 보인다**
- 진입: **가입 완료 후 `/onboarding` 으로 이동**(가입 흐름 변경 필요) · 추천 탭 시드 0개 카드.
- **건너뛰기 허용** → 탭별 랭킹. 넷플릭스는 프로필을 만들 때 좋아하는 작품 몇 개를 고르게 하고, 건너뛰면 다양한 인기작을 보여준다.
- 시드 1~2개: 추천 상단에 "좋아하는 작품을 더 담으면 추천이 넓어져요".

### 2-6. 더 보기 · 체인 · 새로고침
- **"더 보기" 버튼**(무한 스크롤 아님), 한 번에 20개.
- **체인** = 한 칩에서 이어 보는 추천 흐름. 서버가 `chainId` 를 발급하고, 체인에서 이미 보여준 작품을 **서빙 상태 테이블 `rec_chain`** 에 동기로 저장한다(로그 테이블에서 읽지 않는다 — §5-8).
- **체인 수명 = 브라우저 세션 × 칩.** 프론트는 `sessionStorage` 에 칩별 `chainId` 를 둔다.
  - 상세에서 **뒤로 오면 같은 체인·같은 목록·스크롤 위치 복원**(react-query 캐시 + 스크롤 복원). 새 요청을 만들지 않는다
  - 새 세션·탭 재진입·30분 무활동 뒤에는 새 체인
- **새로고침 버튼은 두지 않는다.** 추천기는 같은 입력에 같은 첫 페이지를 내므로 "새로고침"은 같은 목록을 보여줘 고장처럼 보인다.
  넷플릭스도 "최근에 추천받은 것을 다시 찾을 수 있어야 한다"며 안정성을 명시했다.
  목록은 좋아요·싫어요·관심 없음이 바뀌면 **다음 "더 보기"부터**, 새 체인은 첫 페이지부터 달라진다.
- 칩을 바꾸면 칩마다 체인이 따로다.

### 2-7. 빈 상태 · 오류 · 대체
| 상황 | 화면 | `fallback_reason` |
|---|---|---|
| 비로그인 | 로그인 배너 + 랭킹 대체 | `anonymous` |
| 시드 0 | 온보딩 안내 카드 | `no_seed` |
| 칩 플랫폼에 시드 없음 | "이 플랫폼에서 좋아하는 작품을 담아보세요" + 그 플랫폼 랭킹 | `no_seed_platform` |
| 추천 서비스 실패·시간 초과 | 랭킹 대체를 조용히 표시 | `service_error` / `timeout` |
| 서킷 열림 | 랭킹 대체 | `circuit_open` |
| 더 볼 작품 없음 | "더 보기" 숨김 | — |

- **랭킹 대체 규칙**: `external_ranking` 에서 `content_id` 가 있는 행만(null 제외), 성인 작품 제외. 전체 칩 = 게임·영화·시리즈·웹소설 랭킹을 번갈아 20개. 이유 표시 없음.
- 당근 동네생활은 추천 엔진이 늦거나 실패하면 후보의 기본 순서로 대체한다.

---

## 3. 시스템 구조

```
[브라우저]
  추천 탭 ──GET /api/recommendations──────────────▶ [Spring API]
  트래커 ──POST /api/rec-events (묶음, keepalive)───▶   ① 사용자 식별 (JWT)
  반응   ──PUT /api/works/{id}/reaction────────────▶   ② SeedResolver: 시드·싫어요·관심없음 (content_id)
                                                       ③ CorpusMap: content_id → 플랫폼 코퍼스 키
                                                       ④ rec_chain 에서 seen 읽기
                                                       ⑤ POST /v1/recommend ──▶ [추천 라우터 (Python, 가벼움)]
                                                                                   │  M6 혼합 · 탭 분기
                                                                                   ├─▶ [steam 엔진 프로세스]   ~2.6GB
                                                                                   ├─▶ [tmdb 엔진 프로세스]    ~1.1GB
                                                                                   ├─▶ [webtoon 엔진 프로세스] ~0.2GB
                                                                                   └─▶ [webnovel 엔진 프로세스]~0.5GB
                                                       ⑥ 코퍼스 키 → content_id → WorkSummaryDTO (DB 없음·성인 제외 후 채움)
                                                       ⑦ rec_chain 갱신 (동기) · 로그 큐 적재 (비동기)
                                                       ▼
                                                 [PostgreSQL]
                                                  public.*       기존
                                                  aod_rec.*      서빙 상태 (rec_chain, not_interested, corpus_map)
                                                  aod_log.*      로그 (버려도 되는 경로)
```
- 엔진은 **플랫폼당 프로세스 1개**(§1-3 패키지 충돌). 메모리 총량은 약 4.3GB 로 같다. API 서버(t3.small)와 **다른 호스트**, 내부망 전용.
- 라우터는 엔진을 import 하지 않는다 — HTTP 로 부르고 M6 만 수행한다.
- **서빙 상태(`aod_rec`)와 로그(`aod_log`)를 분리한다.** 서빙 상태는 추천 정확성에 필요해 동기로 쓰고, 로그는 유실돼도 추천이 틀리지 않는다.

---

## 4. API 계약

### 4-1. 프론트 ↔ 백엔드

**추천 조회**
```
GET /api/recommendations?tab=all|movie|tv|game|webtoon|webnovel&chainId=&size=20
Authorization: Bearer <token>
X-Anon-Id: <uuid>   X-Session-Id: <uuid>

200 {
  "requestId": "uuid", "chainId": "uuid", "pageDepth": 0,
  "fallback": false, "fallbackReason": null,
  "items": [
    { "impressionId": "uuid", "rank": 0,
      "work": { WorkSummaryDTO },
      "reason": { "type": "like|bookmark|review|exploration", "seedContentId": 123, "text": "코코를 좋아해서" } }
  ],
  "hasMore": true
}
```
| 상태 | 의미 |
|---|---|
| 200 `fallback=true` | 비로그인 · 시드 없음 · 추천 서비스 실패 (화면은 정상) |
| 400 | `tab`·`size` 형식 오류 |
| 401 | 토큰이 **있는데** 만료·위조 — 프론트는 기존 로그인 만료 흐름으로. 토큰이 없으면 401 이 아니라 `anonymous` 대체 |
| 404 | `chainId` 가 없거나 만료 → 프론트는 `chainId` 없이 다시 요청 |

**반응 상태 지정 (신규)**
```
PUT /api/works/{id}/reaction
{ "state": "LIKE|DISLIKE|NONE", "source": "rec_tab|detail|onboarding|…",
  "requestId": "…", "impressionId": "…" }
200 { "state": "DISLIKE", "previousState": "LIKE", "likeCount": 10, "dislikeCount": 2 }
```
- 멱등: 같은 상태를 다시 보내면 변화 없음 · 이벤트도 남기지 않는다.
- 기존 `POST …/like·dislike` 토글은 **응답 형태 그대로 유지**(프론트 `userLikeType` 사용처 호환). 내부는 새 서비스 메서드를 호출한다.

**관심 없음 (신규)**
```
PUT    /api/recommendations/not-interested/{contentId}   { "requestId": "…", "impressionId": "…" }
DELETE /api/recommendations/not-interested/{contentId}
```

**이벤트 묶음**
```
POST /api/rec-events          Content-Type: text/plain (본문은 JSON) — sendBeacon/keepalive 호환
{ "anonId": "…", "sessionId": "…", "appVersion": "…", "device": "mobile|desktop",
  "events": [ { "eventId": "uuid", "type": "impression_viewed", "clientTs": "…Z",
                "requestId": "…", "impressionId": "…", "contentId": 123, "payload": { … } } ] }
202 { "accepted": 18, "rejected": 2 }      429 속도 제한
```
- 식별자는 **헤더가 아니라 본문**에 담는다 — `sendBeacon` 은 헤더를 실을 수 없고, `text/plain` 은 교차 오리진 사전 요청이 필요 없다.
- 인증 헤더가 있으면 user_id 를 붙인다. 없으면(비콘) `session_id` 로 같은 세션의 인증 요청에서 user_id 를 뒤에 채운다.
- 한 번에 최대 50개. 클라이언트가 보낼 수 있는 타입은 화이트리스트(§5-2 "기록 위치 = 클라이언트")만.

### 4-2. 백엔드 ↔ 추천 라우터
```
POST /v1/recommend
{ "tab": "all|movie|tv|game|webtoon|webnovel", "k": 20, "buffer": 10,
  "seeds":    { "steam": [730], "tmdb": [4821], "webtoon": [], "webnovel": [] },   // 코퍼스 키 (TMDB 는 행 번호)
  "disliked": { … 같은 모양 … },
  "excluded": { … 관심 없음 … },
  "seen":     { … 같은 모양 … } }

200 { "items": [ { "platform": "tmdb", "key": 5120, "rank": 0,
                   "dominantSeed": 4821, "candidateSource": "content_sim",
                   "isExploration": false, "propensity": 1.0,
                   "score": { "final": 0.61, "sim": 0.55, "factors": { "rating": 1.03, "genre": 1.12, "media": 1.0 } },
                   "factorSchema": "tmdb.v1" } ],
      "exhausted": { "steam": false, "tmdb": false, "webnovel": true },
      "droppedSeeds": { "tmdb": [99999] },
      "partial": [],
      "versions": { "router": "git sha", "engines": { "tmdb": { "sha": "…", "config": "PRODUCTION 해시", "corpus": "tmdb_v1" } } } }
```
- `k + buffer` 개를 받아, 백엔드가 DB 에 없는 작품·성인 작품을 뺀 뒤 20개를 채운다.
- `score.factors` 는 플랫폼별 곱셈 인자. 스키마에 버전(`factorSchema`)을 붙인다.
- `excluded` 는 제외만(감점·시리즈 제외 없음). `disliked` 는 플랫폼 규칙(§8-3).

---

## 5. 로그 설계

### 5-1. 원칙 (사례 근거)
| 원칙 | 근거 |
|---|---|
| 응답한 것과 화면에 보인 것을 나눈다 | 유튜브 노출 = 썸네일 50% 이상·1초 초과 · LinkedIn 50%·0.5초 · MRC 광고 표준 50%·1초 · 넷플릭스 Impressions("봤지만 반응 안 한 콘텐츠" 식별) |
| 서빙 시점 입력을 그대로 남긴다 | 구글 Rules of ML #29·#31 (유튜브 홈이 서빙 시점 로깅으로 품질 향상) · 넷플릭스 Axion |
| 부정 신호를 종류별로 따로 | 유튜브 싫어요·관심 없음·채널 추천 안 함 · 유튜브 2019 dismiss 를 만족 목표로 · 당근 숨기기 비율 지표 |
| 클릭보다 강한 전환을 목표로 | 유튜브 클릭 → 시청 시간(낚시성 문제) · 당근 클릭·채팅 전환 확률 |
| 편향 보정 칸을 처음부터 | 유튜브 REINFORCE(노출 확률을 로깅하지 못해 추정) · Bottou 2013 · Joachims 2017 |
| 로그 품질 장치 | 당근 이벤트센터(명명 규칙·스키마·DLQ·2단 중복 제거) · 넷플릭스 Impressions(스키마 없으면 누락 원인 판별 불가) · Microsoft ExP(SRM) |
| 서빙 상태와 로그를 섞지 않는다 | (v1 리뷰 결과) 비동기·유실 허용 로그에서 서빙 입력을 읽으면 추천이 틀린다 |

### 5-2. 이벤트
우리 서비스의 소비는 **외부 플랫폼**에서 일어난다. 유튜브의 시청 시간 자리는 **"볼 수 있는 곳" 링크 클릭**이다.

| 이벤트 | 기록 위치 | 정의 · payload | 신호 |
|---|---|---|---|
| (요청) | 서버 → `rec_request` | 추천 응답 1건 | — |
| (응답 작품) | 서버 → `rec_item_served` | 응답한 작품 1개 | — |
| `impression_viewed` | 클라이언트 | 카드가 **50%·1초 기준을 넘는 순간 1회** + 화면을 벗어나거나 페이지를 떠날 때 최종값 1회. `max_visible_ratio`·`visible_ms`·`final` | 매우 약함 |
| `card_clicked` | 클라이언트 | 카드 → 상세 | 약함 |
| `detail_viewed` | 클라이언트 | 상세 체류. `detail_open_id`·`visible_ms`·`max_scroll_ratio`·`idle_capped` | 짧으면 부정 · 길면 중간 |
| `outbound_clicked` | 클라이언트 | "볼 수 있는 곳" 링크. `platform`·`detail_open_id` | **강함 (주 지표)** |
| `rec_loaded_more` · `rec_tab_changed` | 클라이언트 | 탐색 행동 | 맥락 |
| `reaction_changed` | **서버** (반응 서비스) | `from`·`to`(LIKE/DISLIKE/NONE)·`source` — 기존 토글 API 경유도 포함 | 좋아요 매우 강함 · 싫어요 강한 부정 |
| `bookmark_changed` | **서버** | `on`/`off`·`source` | 강함 |
| `not_interested_changed` | **서버** | `on`/`off` | 부정 (노출 한정) |
| `review_saved` | **서버** | `rating`·신규/수정 | 강함 |

- **반응·북마크·리뷰는 서버 쓰기 경로에서** — 클라이언트 유실이 없고, 백엔드 테이블이 지우거나 덮어써도 `from→to` 이력이 남는다.
- 노출·클릭·체류·외부 링크는 클라이언트만 안다.
- 이름 규칙: `{대상}_{과거형 동사}`, 소문자 스네이크. 화면은 `surface` 필드로 뺀다(당근 `client_{action}_{service}_{screen}_{object}` 를 화면 수가 적은 우리 규모로 줄인 것).

### 5-3. 테이블

**서빙 상태 — 스키마 `aod_rec` (동기 · 파티션 없음)**
```sql
CREATE TABLE aod_rec.rec_chain (
  chain_id    uuid PRIMARY KEY,
  user_id     bigint NOT NULL,
  tab         text   NOT NULL,
  seen_ids    bigint[] NOT NULL DEFAULT '{}',   -- content_id, 최대 500
  page_depth  int    NOT NULL DEFAULT 0,
  updated_at  timestamptz NOT NULL
);                                               -- 24시간 지난 행은 매시간 삭제

CREATE TABLE aod_rec.not_interested (
  user_id bigint NOT NULL, content_id bigint NOT NULL, created_at timestamptz NOT NULL,
  PRIMARY KEY (user_id, content_id)
);                                               -- 90일 지난 행 매일 삭제

CREATE TABLE aod_rec.corpus_map (                -- 코퍼스 빌드가 산출 (§8-4)
  content_id bigint NOT NULL, platform text NOT NULL, corpus_key text NOT NULL,
  corpus_version text NOT NULL, PRIMARY KEY (platform, corpus_key, corpus_version)
);
CREATE INDEX ON aod_rec.corpus_map (content_id, corpus_version);
```

**로그 — 스키마 `aod_log` (비동기 · 월 파티션)**
PostgreSQL 파티션 테이블의 PK 는 파티션 키를 포함해야 한다.
```sql
CREATE TABLE aod_log.rec_request (
  request_id   uuid NOT NULL,
  served_at    timestamptz NOT NULL,             -- UTC
  chain_id     uuid NOT NULL,
  page_depth   int  NOT NULL,
  user_id      bigint,
  anon_id      uuid NOT NULL,
  session_id   uuid NOT NULL,
  surface      text NOT NULL,                    -- for_you
  tab          text NOT NULL,
  seed_ids     bigint[] NOT NULL,                -- content_id 스냅샷 (출처 순서대로)
  seed_sources text[]  NOT NULL,                 -- like/bookmark/review
  disliked_ids bigint[] NOT NULL,
  excluded_ids bigint[] NOT NULL,
  seen_ids     bigint[] NOT NULL,                -- 재현용 (최대 500)
  dropped_seed_ids bigint[] NOT NULL DEFAULT '{}',
  experiments  jsonb NOT NULL DEFAULT '{}',
  versions     jsonb NOT NULL,
  fallback     boolean NOT NULL,
  fallback_reason text,
  partial      text[] NOT NULL DEFAULT '{}',
  latency_ms   int,
  app_version  text, device text, user_agent text,
  PRIMARY KEY (request_id, served_at)
) PARTITION BY RANGE (served_at);

CREATE TABLE aod_log.rec_item_served (
  impression_id    uuid NOT NULL,
  served_at        timestamptz NOT NULL,
  request_id       uuid NOT NULL,
  content_id       bigint NOT NULL,
  platform         text NOT NULL,
  corpus_key       text NOT NULL,
  rank_position    int  NOT NULL,
  candidate_source text NOT NULL,
  reason_type      text,
  reason_seed_id   bigint,
  score            jsonb NOT NULL,
  factor_schema    text,
  is_exploration   boolean NOT NULL DEFAULT false,
  propensity       real NOT NULL DEFAULT 1.0,
  interleave_team  text,
  PRIMARY KEY (impression_id, served_at)
) PARTITION BY RANGE (served_at);

CREATE TABLE aod_log.event (
  event_id      uuid NOT NULL,
  server_ts     timestamptz NOT NULL,
  event_type    text NOT NULL,
  origin        text NOT NULL,                   -- client / server
  user_id       bigint, anon_id uuid, session_id uuid,
  content_id    bigint, request_id uuid, impression_id uuid,
  surface       text,
  payload       jsonb NOT NULL DEFAULT '{}',
  client_ts     timestamptz,
  app_version   text, device text, user_agent text,
  PRIMARY KEY (event_id, server_ts)
) PARTITION BY RANGE (server_ts);

-- 중복 제거: 파티션 PK 로는 event_id 전역 유일을 보장할 수 없다 → 짧은 보관 테이블로 거른다
CREATE TABLE aod_log.event_seen (event_id uuid PRIMARY KEY, server_ts timestamptz NOT NULL);  -- 7일 보관

CREATE TABLE aod_log.rejected_event (
  id bigserial PRIMARY KEY, raw jsonb NOT NULL, reason text NOT NULL, server_ts timestamptz NOT NULL
);                                               -- 1% 표본만 저장 · 30일 보관

-- 각 파티션 테이블: DEFAULT 파티션 + 월 파티션
CREATE TABLE aod_log.event_default PARTITION OF aod_log.event DEFAULT;
```
- 인덱스: `rec_request(user_id, served_at)` · `rec_request(chain_id)` · `rec_item_served(request_id)` · `event(impression_id)` · `event(user_id, server_ts)` · `event(session_id)`.
- **파티션 관리**: Flyway `V8` 은 스키마·DEFAULT·이번 달과 다음 달 파티션만 만든다. 이후는 백엔드 `@Scheduled` 작업이 매일 "다음 2개월 파티션이 없으면 생성", 보관 기간 지난 파티션 `DROP`.
  DEFAULT 파티션에 행이 쌓이면 경보(§11).
- 기존 m2 브랜치의 `aod_ai.rec_impression`·`rec_event` 는 위 테이블로 **대체**한다(main 에 없어 이행 비용 없음).

### 5-4. 측정 규칙
**노출 (`impression_viewed`)**
- `IntersectionObserver`(threshold 0 · 0.5 · 1.0)로 카드별 보인 비율·누적 보인 시간.
- `document.visibilityState !== 'visible'` 이면 시간을 멈춘다.
- **50%·1초를 넘는 순간 1회 전송**(모바일 탭 종료 유실 대비) + 화면을 벗어나거나 페이지를 떠날 때 **최종값 1회**(`final=true`). 체인 안 카드당 최대 2건.
- 원값을 함께 보내 기준은 분석에서 바꿀 수 있게 한다.

**상세 체류 (`detail_viewed`)**
- `detail_open_id` = 상세 페이지를 열 때 클라이언트가 만드는 UUID(뒤로가기로 다시 열면 새 값).
- 화면에 보인 시간만 누적. 60초 무입력이면 멈춤, 30분에서 자름(`idle_capped`).
- 전송: 15초마다 누적값 + `visibilitychange → hidden`·`pagehide` 에서 최종값. 서버·분석은 같은 `detail_open_id` 의 **최댓값**.

**식별자**
| ID | 발급 | 수명 |
|---|---|---|
| `anon_id` | 클라이언트 UUID, `localStorage` `aod_anon_id` | 브라우저 단위 (§5-9 삭제 규칙) |
| `session_id` | 클라이언트, `sessionStorage`, 30분 무활동 시 갱신 | 세션 |
| `chain_id` | 서버 | 세션 × 칩 |
| `request_id`·`impression_id` | 서버 | 응답 |
| `detail_open_id` | 클라이언트 | 상세 1회 열람 |
| `event_id` | 이벤트 생성자 | 1회 |

- 상세로 갈 때 URL 에 `?rid={requestId}&iid={impressionId}` → 체류·외부 링크·반응을 노출에 잇는다.
- 모든 시각 UTC. 클라이언트 시각과 서버 수신 시각을 둘 다 저장.

### 5-5. 실험 배정 (칸만 먼저)
- 추천은 로그인 사용자만 받으므로 **user_id 로만** 배정한다. `bucket = murmur3_32("{layer}:{salt}:{user_id}") % 10000`.
- 지금 레이어 `rec_ranker` 하나, 변형 `control` 하나. 응답을 **실제로 만든** 변형을 `rec_request.experiments` 에 남긴다(배정이 아니라 적용 기준 — Spotify). 대체 응답은 `experiments={}`.

### 5-6. 지표
| 지표 | 정의 |
|---|---|
| **외부 이동률 (주 지표 — 제안)** | 보인 노출(50%·1초) 중 같은 `impression_id` 로 `outbound_clicked` 가 난 비율 |
| 클릭률 | 보인 노출 대비 `card_clicked` |
| 좋아요·북마크율 | 보인 노출 대비 `reaction_changed(to=LIKE)`·`bookmark_changed(on)` |
| 짧은 체류율 | 클릭 중 `detail_viewed.visible_ms` < 5초 |
| 싫어요·관심 없음 비율 | 보인 노출 대비 |
| 대체 비율 · 부분 응답 비율 · 지연 p95 | `rec_request` |
| 시드 탈락률 | `dropped_seed_ids` / 시드 수 (코퍼스 커버리지) |
| 커버리지 | 기간 내 노출된 서로 다른 작품 / 코퍼스 |

### 5-7. 품질 점검
1. 일별 건수 추세·필드 누락률 경보
2. `impression_viewed` → `rec_item_served` 조인율 — **로그 큐 유실(`log_dropped_total`)이 0 인 날만** "보인 수 > 응답 수" 경보를 판정
3. `card_clicked` 대비 `detail_viewed` 유실률
4. `rejected_event` 비율 · DEFAULT 파티션 행 수
5. 봇: user_agent 규칙 + 초당 클릭 수 이상치 → 분석에서 제외(원본 보관)
6. 지표는 이틀 뒤 확정(늦게 온 이벤트) — 당근은 72시간
7. 실험을 켠 뒤: 그룹 비율 불일치(SRM) 매일 검정

### 5-8. 적재 경로와 부하
- **서빙 상태**(`rec_chain` upsert 1회/요청, `not_interested`)는 요청 트랜잭션에서 동기로.
- **로그**는 요청 스레드가 메모리 큐(최대 10,000건)에 넣기만 하고, `LogWriter` 스레드가 1초 또는 200건마다 **JDBC 배치 INSERT**.
  - `LogWriter` 는 **별도 Hikari 풀(연결 2개)** — 서비스 풀과 경쟁하지 않는다
  - 큐가 가득 차면 버리고 `log_dropped_total` 증가. 추천 응답을 로그 때문에 늦추지 않는다
  - 종료 시 큐를 비운다(graceful shutdown, 최대 10초)
  - 이벤트는 `event_seen` 에 `INSERT … ON CONFLICT DO NOTHING RETURNING` 으로 새 것만 `event` 에 넣는다
- 규모 어림: 일 사용자 1,000 × 요청 5 × 작품 20 = `rec_item_served` 10만 행/일. uuid·jsonb·인덱스 포함 행당 약 1KB → **약 100MB/일**, 이벤트 포함 약 300MB/일. 1년 보관 시 약 100GB — 월 파티션·보관 기간이 필요한 규모.
- **이전 조건**: 로그 쓰기가 DB CPU 20% 초과 또는 일 1,000만 행 초과 시 로그 전용 저장소로.

### 5-9. 개인정보
- **로그 수집을 켜기 전에** 개인정보 처리방침에 수집 항목(행동 이벤트·기기 정보·익명 식별자)·목적(추천 개선)·보관 기간을 반영한다(출시 순서 0단계). 동의가 필요한지는 법무 확인.
- IP 는 저장하지 않는다. `user_agent` 는 90일 뒤 지운다(파티션 갱신 작업이 컬럼을 null 로).
- 보관: 원시 요청·이벤트 로그 **1년**(파티션 DROP), 집계는 기간 제한 없음.
- 삭제: 탈퇴 API 가 아직 없다. 탈퇴 기능을 만들 때 `user_id` 행과 **그 사용자와 같이 쓰인 `anon_id`·`session_id` 행**을 함께 지우는 작업을 포함한다(anon_id 만 남으면 재연결 가능). 그 전까지는 요청 시 운영 스크립트로 삭제.

---

## 6. 백엔드 추가 사항 (Spring)

### 6-1. 브랜치 정리
- `feature/m2-recommend-serving` 을 main 기준으로 재작업. 마이그레이션 `V8__create_rec_schemas.sql`.
- 자바 랭커(`Ranker`·`CandidateGenerator`·`FeatureCalculator`)는 쓰지 않는다. 추천 탭은 캐시하지 않는다(좋아요 즉시 반영). 중복 Flyway 의존성 제거.

### 6-2. 새 구성 요소
| 구성 요소 | 역할 |
|---|---|
| `RecommendController` | `GET /api/recommendations` · 관심 없음 `PUT/DELETE` |
| `ReactionController`·`ReactionService` | `PUT /api/works/{id}/reaction` · 기존 토글 API 도 내부적으로 이 서비스를 호출 · `reaction_changed` 발행 |
| `RecEventController` | `POST /api/rec-events` (`text/plain` 수용) · 타입 화이트리스트 · 속도 제한 · 거절 표본 저장 |
| `SeedResolver` | 시드 규칙(§6-4) · 싫어요 · 관심 없음 |
| `CorpusMapRepository` | content_id ↔ 플랫폼 코퍼스 키 (현재 `corpus_version`) |
| `ChainService` | `rec_chain` 생성·조회·갱신 (seen 최대 500, 넘으면 `hasMore=false`) |
| `RecRouterClient` | 전용 `RestTemplate` 빈(연결 300ms · 읽기 2.0초) + 서킷 브레이커(Resilience4j, 30초 창에서 실패 50% → 30초 열림) + 동시 호출 제한(세마포어 20) |
| `CardAssembler` | 코퍼스 키 → content_id → `WorkSummaryDTO` 배치 조회 · DB 없음·성인 제외 · 20개 채우기 |
| `FallbackProvider` | §2-7 랭킹 대체 |
| `ReasonBuilder` | §2-3 문구·조사 |
| `ExperimentAssigner` | §5-5 |
| `LogQueue`·`LogWriter`·`PartitionMaintenanceJob` | §5-3·§5-8 |
| `RecFeatureFlag` | 추천 API·탭 노출 킬 스위치(설정값) · 허용 사용자 목록 |

### 6-3. 기존 코드 변경
- `LikeService`: 상태 지정 메서드(`setReaction(state)`) 추가 → 토글은 이를 호출. 이전 상태 반환. 예외를 상태 코드로 구분(없는 작품 404 · 비로그인 401).
- `BookmarkService`·`ReviewService`: 변경 시 이벤트 발행. `source`·`requestId`·`impressionId` 선택 파라미터.
- 가입 완료 응답 후 프론트가 `/onboarding` 으로 가도록 응답에 `needsOnboarding` 추가(시드 0).

### 6-4. 시드 규칙
1. 긍정 후보 = 좋아요 ∪ 북마크 ∪ 평점 4 이상 리뷰 (**열린 결정 1** — 초안 기본값)
2. **싫어요는 모든 긍정을 이긴다** — 싫어요한 작품은 시드에서 빠진다
3. 평점 2 이하 리뷰가 있는 작품은 시드에서 뺀다(싫어요로 쓰지는 않는다 — 싫어요 신호는 DISLIKE 만)
4. 같은 작품에 여러 출처가 있으면 문구 우선순위 좋아요 > 리뷰 > 북마크
5. **정렬 = 가장 최근 상호작용 순**(동률은 content_id). 추천기의 시드 순서가 결과에 영향을 주므로(버킷 순서) 이 규칙으로 고정한다
6. 플랫폼당 최대 50개(최근순 앞에서 자름)
7. `corpus_map` 에 없는 시드는 넘기지 않고 `dropped_seed_ids` 에 기록

---

## 7. 프론트 추가 사항 (React)

### 7-1. 화면·컴포넌트
| 추가 | 내용 |
|---|---|
| 라우트 `/for-you` · 홈 세그먼트(모바일) · 상단 메뉴(데스크톱) | §2-1 |
| `RecTabPage` | 칩 · 그리드 · 더 보기 · 빈 상태 · 스크롤 복원 |
| `RecCard` | 기존 카드 + 이유 + ♡ + 더보기 메뉴 |
| `UndoToast` | `previousState` 로 되돌리기 |
| `OnboardingPickWorks` | 장르 온보딩 교체 · 건너뛰기 · 플랫폼별 권장 안내 |
| 가입 흐름 | 가입 완료 → `needsOnboarding` 이면 `/onboarding` |
| 상세 페이지 | `rid`·`iid` 읽기 · `detail_open_id` · 체류 · "볼 수 있는 곳" 클릭 이벤트 |

### 7-2. 데이터
- `recApi.get({tab, chainId})` · `recApi.setReaction(id, state, ctx)` · `recApi.notInterested(id, on)`.
- react-query `useInfiniteQuery`:
  - 키 `['rec', tab, chainNonce]` — `chainNonce` 는 `sessionStorage` 에 칩별로 두고 새 체인일 때만 바꾼다
  - 첫 `pageParam` = 저장된 `chainId`(없으면 null), `getNextPageParam` = `hasMore ? chainId : undefined`
  - `staleTime: Infinity` · `gcTime` 30분 — 뒤로가기 복원 동안 다시 받지 않는다(전역 기본값과 무관하게 이 쿼리만)
  - 404(체인 만료) → nonce 교체 후 재요청
- 반응 성공 시 **보이는 목록은 유지**(카드 제거/표시만), 다음 페이지부터 서버가 반영.

### 7-3. 트래커 `tracker.ts`
- `anon_id`·`session_id` 발급(§5-4).
- `track(type, fields)` → 메모리 큐 → **5초 또는 20건마다** `fetch('/api/rec-events', {method:'POST', body, keepalive:true, headers:{'Content-Type':'text/plain'}})`. 로그인 상태면 `Authorization` 도 싣는다.
- `visibilitychange → hidden`·`pagehide` → `navigator.sendBeacon(url, new Blob([json], {type:'text/plain'}))`.
- 실패 시 3회 지수 재시도 후 버림(event_id 가 같아 재시도해도 중복 적재 안 됨).
- `useImpressionTracker(ref, impressionId)` · `useDwellTracker(contentId, rid, iid)` — §5-4.
- 개발 모드에서 콘솔에 이벤트 출력.

---

## 8. 추천 서비스 추가 사항 (Python)

### 8-1. 구성
- **엔진 프로세스 4개**: 시험대 `tryout/backend.py` 의 플랫폼 클래스를 바탕으로 플랫폼당 1개. `POST /engine/recommend` · `GET /health`.
- **라우터 1개**: 엔진을 import 하지 않는다. 탭 분기 · M6 혼합 · 제한 시간 · 부분 응답.
- 배포: 한 호스트(메모리 8GB 이상)에 컨테이너 5개, 내부망 전용.

### 8-2. 엔진 어댑터 (공통 검증)
- 코퍼스 밖 시드는 **예외 대신 제외**하고 `droppedSeeds` 로 반환(Steam·TMDB·웹소설은 지금 예외를 던진다).
- 남은 시드가 0이면 그 플랫폼은 **빈 결과 + `exhausted=true`**(오류 아님).
- 확정값(`PRODUCTION`)을 그대로 쓰고, `next_page(seeds, seen, page_size=k+buffer, …)` 를 호출.
- 점수 인자: 랭커를 **순서를 바꾸지 않게** 계측해 곱셈 인자를 따로 담는다(`factorSchema` 버전). §10 동일성 테스트로 확인.
- 웹소설 화수는 카드 문자열이 아니라 dataset 의 `episode_count` 필드로 넘긴다.

### 8-3. 싫어요·관심 없음 — 플랫폼별 동작
| 플랫폼 | 싫어요 (`disliked`) | 관심 없음 (`excluded`) |
|---|---|---|
| Steam | `disliked_appids` — 제외 + 유사 감점(w 2.0, 미채점) | `seen` 에 합쳐 제외 |
| TMDB | 인자 없음 → `seen_rows` 에 합쳐 **제외만** | 같음 |
| 웹툰 | `disliked_ids` — 제외 + 같은 시리즈 제외 (감점 0, T-11) | `exclude` 에 합쳐 제외 |
| 웹소설 | 인자 없음 → `seen_ids` 에 합쳐 제외. 이 경로는 `drop_excluded_series` 로 **같은 판본(작가·제목 키)도 빠진다** | `seen_ids` 에 합쳐 제외 (같은 판본 제외가 따라온다 — 알려진 차이) |

플랫폼 간 차이는 의도가 아니라 현재 엔진의 한계다. 싫어요 유사 감점을 플랫폼마다 맞추는 일은 T-11 후속 라운드에서 다룬다.

### 8-4. M6 혼합 (전체 탭)
- 대상: Steam · TMDB · 웹소설(평가된 그대로). 웹툰 포함은 사전등록 평가 후.
- 매 페이지: 플랫폼별 `next_page(k=50, seen=그 플랫폼 seen)` → M6 할당·라운드로빈으로 20개.
- **부족분은 채우지 않는다**(평가된 규칙 유지). 20개 미만이어도 응답한다.
- `hasMore` = 세 플랫폼 중 하나라도 `exhausted=false`.
- seen 은 플랫폼별로 나눠 `rec_chain` 에 content_id 로 둔다(백엔드가 `corpus_map` 으로 나눈다).

### 8-5. 지연 예산
| 계층 | 제한 |
|---|---|
| 프론트 요청 | 3.0초 → 넘으면 오류 화면 대신 재시도 버튼 |
| 백엔드 전체 | 2.5초 (DB 조회·카드 조립 포함) |
| 백엔드 → 라우터 읽기 | 2.0초 |
| 라우터 → 엔진 | 1.5초. 늦은 플랫폼은 빼고 `partial` 로 응답 |

- **출시 전 관문**: seen 0·200·500 × 동시 요청 1·5·20 에서 엔진별 `next_page` p95 를 잰다. 라우터 p95 가 1.5초를 넘으면 예산을 조정하거나 Steam 엔진을 복제한다.

### 8-5b. 코퍼스와 ID 매핑
- 코퍼스 빌드가 `corpus_map`(content_id ↔ 코퍼스 키)과 **커버리지 리포트**(플랫폼별 백엔드 작품 중 코퍼스에 있는 비율)를 함께 산출한다.
- TMDB 는 `platform_data(TMDB_MOVIE|TMDB_TV, 숫자 id)` → `movie_{id}`/`tv_{id}` → 코퍼스 행 번호.
- 웹소설은 NaverSeries 만 — KakaoPage 로만 좋아요한 작품은 시드가 되지 않는다(시드 탈락률 지표로 감시).
- 갱신 주기: **주 1회 제안**(크롤러 신규 작품 → 임베딩 → 새 `corpus_version` → 엔진 교체). 버전이 바뀌면 진행 중 체인은 끝까지 옛 매핑을 쓰지 않고 새로 매핑한다(seen 은 content_id 라 영향 없음).

---

## 9. 출시 순서

| 순서 | 작업 | 완료 조건 |
|---|---|---|
| 0 | **개인정보 처리방침 갱신** · 삭제 절차 | 법무 확인 · 게시 |
| 1 | `V8` 스키마 · `LogWriter` · 파티션 작업 · `ReactionService`(이벤트 발행) · `/api/rec-events` | 기존 좋아요 화면에서 `reaction_changed` 이력이 쌓이고 DEFAULT 파티션 0행 |
| 2 | 트래커 · 상세 체류 · 외부 링크 이벤트 | 기존 화면 이벤트 적재 · 품질 점검 1~4 통과 |
| 3 | 엔진 4개 + 라우터 배포(내부망) · 동일성 테스트 · **지연 관문** | §10 통과 · §8-5 관문 통과 |
| 4 | `corpus_map`·커버리지 리포트 · 추천 API(시드·체인·카드·대체·서킷) | 테스트 계정으로 탭별 응답 · 대체 경로 강제 시험 |
| 5 | 추천 탭 · 노출 추적 · 반응 상태 API · 관심 없음 — **기능 플래그로 팀만** | 팀 내부 1주 사용 · 지표 대시보드 확인 |
| 6 | 온보딩 작품 고르기 · 가입 흐름 | 시드 0 사용자가 추천까지 도달 |
| 7 | 단계 공개: 10% → 50% → 100% (플래그) | 각 단계 대체 비율 < 5% · 지연 p95 < 2.5초 · 오류율 < 1% |

---

## 10. 테스트 계획
| 종류 | 내용 |
|---|---|
| **동일성** | 엔진 서비스 `next_page` 결과 == 평가 경로 결과 (각 플랫폼 기준 목록: TMDB X-27 · 웹툰 T-10 n0 · 웹소설 W-6 · Steam X-20). 점수 인자 계측 전후 순서 동일 |
| 계약 | 라우터·엔진 요청/응답 JSON 스키마 · 코퍼스 밖·빈 시드 · `partial` |
| 백엔드 단위 | 시드 규칙 · 조사 · 반응 상태 전이(좋아요→싫어요→되돌리기=좋아요) · 토글 API 응답 형태 불변 |
| 백엔드 통합 | Testcontainers PostgreSQL 로 `V8` · 파티션 생성 · `event_seen` 중복 제거 · 체인 seen |
| 프론트 | 노출 추적(50%·1초 교차 1회 + 최종) · 비콘 페이로드 · 뒤로가기 복원 |
| 부하 | k6: 추천 API 동시 20 · 이벤트 초당 200 · 로그 큐 유실 0 |
| 장애 | 라우터 중단 → 대체 · 엔진 1개 중단 → `partial` · DB 로그 풀 고갈 → 추천 응답 정상 |

## 11. 운영 · 모니터링
| 경보 | 조건 |
|---|---|
| 대체 비율 | 10분 평균 > 10% |
| 부분 응답 | 10분 평균 > 5% |
| 지연 | 추천 API p95 > 2.5초 |
| 로그 유실 | `log_dropped_total` 증가 |
| DEFAULT 파티션 | 행 수 > 0 |
| 엔진 | `/health` 실패 · RSS > 기준의 120% · 코퍼스 버전 불일치 |
| 이벤트 | 일 건수 전주 대비 ±50% · 거절 비율 > 2% |

- 킬 스위치: `RecFeatureFlag` 끄면 추천 탭 숨김 · API 는 랭킹 대체만.
- 담당: 추천 서비스(AI 리포) · 백엔드 API · 프론트 트래커 — 담당자 지정은 열린 결정.

---

## 12. 열린 결정
1. 시드에 북마크·평점 4 이상 리뷰 포함 (초안: 포함)
2. 주 지표 = 외부 이동률 (초안: 채택)
3. 노출 보고 기준 50%·1초 (원값은 저장)
4. 로그 보관 1년 · 관심 없음 90일
5. 모바일 진입 = 홈 세그먼트 (메뉴 7개 회피)
6. 코퍼스 갱신 주 1회
7. 담당자

## 13. 후속
- 전체 탭에 웹툰 넣기 (M6 확장 사전등록)
- 싫어요 유사 감점 플랫폼 정렬 (T-11 후속)
- 상세 페이지 "비슷한 작품" · 홈 추천 섹션 · 넷플릭스식 가로 행
- 피드백 기록 확인·초기화 화면 (유튜브 My Activity) · 작가/개발사 단위 "추천 안 함"
- 탐색 칸 · 인터리빙 (`TRANSITION_ROADMAP.md` 1단계)
- Python 추천기 독스트링의 `aod_ai.rec_impression` 언급을 `aod_rec.rec_chain` 으로 정리

---

## 부록 A. 출처
로그·실험은 `TRANSITION_ROADMAP.md` §8-5, 구조는 `INDUSTRY_COMPARISON.md` 출처 목록. 화면:
- 넷플릭스 "Learning a Personalized Homepage" (2015) — https://netflixtechblog.com/learning-a-personalized-homepage-aa8ec670359a
- 넷플릭스 "Netflix Recommendations: Beyond the 5 stars" — https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429
- 넷플릭스 "Artwork Personalization" (2017) — https://netflixtechblog.com/artwork-personalization-c589f074ad76
- 넷플릭스 도움말 — https://help.netflix.com/en/node/100639
- 넷플릭스 "Goodbye Stars, Hello Thumbs" (2017) — http://about.netflix.com/en/news/goodbye-stars-hello-thumbs
- 넷플릭스 "Two Thumbs Up" (2022) — http://about.netflix.com/en/news/two-thumbs-up-even-better-recommendations
- 유튜브 도움말 "Manage your recommendations" — https://support.google.com/youtube/answer/6342839
- 유튜브 도움말 (시청 기록 꺼짐 시 홈 추천 제거) — https://support.google.com/youtube/answer/95725
- 유튜브 블로그 (2019, 홈 추천 조작) — https://blog.youtube/news-and-events/giving-you-more-control-over-homepage/
- 당근 피드실 채용 글 (숨기기 고도화 · −40%) — https://careers.daangn.com/blog/post/당근-피드실-채용-홈화면-개발자-pm/
- netflixtechblog.com 은 직접 접속이 막혀 프록시로 읽었다.

---

## 부록 B. 리뷰 기록 (v1 → v2)

v1 을 코드(백엔드·프론트 번들·Python 추천기)와 대조한 독립 리뷰에서 23건이 나왔다. 전부 반영했다.

| # | 등급 | 리뷰 지적 | v2 반영 |
|---|---|---|---|
| 1 | 심각 | "더 보기" seen 을 비동기·유실 허용 로그 테이블에서 읽음 | 서빙 상태 `aod_rec.rec_chain` 동기 저장, 로그와 분리 (§3·§5-3·§5-8) |
| 2 | 심각 | 파티션 테이블 PK 에 파티션 키 없음 · 전역 중복 제거 불가 · 파티션 생성·DEFAULT 없음 | PK `(id, ts)` · `event_seen` 중복 제거 · DEFAULT 파티션 · 파티션 유지 작업 (§5-3) |
| 3 | 심각 | 엔진 4개 한 프로세스 적재 불가(`src` 패키지·`chdir`·환경변수 충돌) | 엔진 프로세스 4개 + 라우터 (§3·§8-1) |
| 4 | 심각 | M6 에 웹툰 없음 · 혼합 페이지 규칙 없음 · 화수 문자열 파싱 | 전체 탭 = Steam·TMDB·웹소설, 페이지·부족분·`hasMore` 규칙, `episode_count` 필드 (§2-2·§8-4) |
| 5 | 심각 | 토글로 되돌리기 → 기존 좋아요 소실 · 비멱등 | 상태 지정 `PUT /reaction` + `previousState` (§2-4·§4-1·§6-3) |
| 6 | 심각 | 처리방침이 로그 수집보다 늦음 · 탈퇴 경로 없음 · anon_id 재연결 | 출시 0단계로 이동 · 삭제 범위에 anon/session 포함 (§5-9·§9) |
| 7 | 중요 | 점수 항목 원값을 엔진이 안 냄 | 순서 불변 계측 · 곱셈 인자 스키마 버전 · 동일성 테스트 (§1-2·§8-2·§10) |
| 8 | 중요 | 코퍼스 밖 시드 예외가 3개 플랫폼 · 빈 시드 예외 | 어댑터 공통 검증 · 빈 시드 = 빈 결과 (§8-2) |
| 9 | 중요 | TMDB·웹소설은 싫어요 인자 없음 | 플랫폼별 동작표 (§8-3) |
| 10 | 중요 | ID 키 불일치(TMDB 행 번호) · 커버리지 없음 · DB 없음/성인 시 20개 미달 | `corpus_map` · 커버리지 리포트 · `k+buffer` 채우기 · 시드 탈락률 (§4-2·§8-5b·§5-6) |
| 11 | 중요 | 지연 예산 모순 · 측정 조건 다름 · RestTemplate 타임아웃 없음 | 계층별 예산 · 출시 전 부하 관문 · 전용 빈·서킷·세마포어 (§8-5·§6-2) |
| 12 | 중요 | "관심 없음" 의미 모순 · 요청 필드 없음 | `excluded` 필드 · 범위(추천 탭 전 칩)·90일 · 제외만 (§2-4·§4-2·§8-3) |
| 13 | 중요 | 이탈 시에만 전송 → 유실 · 비콘 헤더 불가·교차 오리진 · `detail_open_id` 미정의 | 기준 교차 시 1회 + 최종 · 본문 식별자 · `text/plain`·`keepalive` · `detail_open_id` 정의 · 유실 보정 점검 (§4-1·§5-4·§5-7·§7-3) |
| 14 | 중요 | `/api/rec-events` 남용 방어 없음 · 이벤트 행에 기기 정보 없음 | 타입 화이트리스트 · 속도 제한 · 거절 1% 표본·30일 · `event` 에 기기 컬럼 (§4-1·§5-3·§6-2) |
| 15 | 중요 | 새로고침이 같은 목록 · 뒤로가기 새 체인 · 쿼리 키에 chainId | 새로고침 버튼 제거 · 체인 = 세션×칩 · 뒤로가기 복원 · 키 `chainNonce` (§2-6·§7-2) |
| 16 | 중요 | 온보딩 좋아요 부작용 · `source` 누락 · 가입 흐름 · 플랫폼당 시드 1개 | `source=onboarding` · 좋아요 목록 노출 결정 · `needsOnboarding` · 플랫폼별 2개 권장 (§2-5·§6-3) |
| 17 | 중요 | 시드 우선순위·충돌·순서 규칙 없음 · 북마크에도 "좋아해서" | 시드 규칙 7개 · 출처별 문구 (§2-3·§6-4) |
| 18 | 중요 | 대체 목록 규칙 빈칸 · `content_id` null · 전체 탭 대체 | null 제외 · 전체 탭 번갈아 · `fallback_reason` 코드 (§2-7) |
| 19 | 사소 | 라우트 누락 · 720px 오인 · 메뉴 6개 · 크롤러 V2/V3 | §1-3 사실 수정 · 반응형 그리드 · 모바일 홈 세그먼트 (§1-3·§2-1·§2-2) |
| 20 | 사소 | 로드맵과 테이블명·주 지표·seen 불일치 · 깨진 참조 | 이 문서를 기준으로 선언 · `seen_ids` 저장 · 로드맵 갱신 · 참조 수정 |
| 21 | 사소 | 행당 크기 과소 · 풀 공유 · 종료 flush · COPY | 1KB/행 재추정 · 별도 풀 2개 · graceful flush · JDBC 배치 (§5-8) |
| 22 | 사소 | anon 버킷 불필요 · 해시·salt · 조사 규칙 | user_id 만 · murmur3 + 레이어 salt · 받침 규칙 (§5-5·§2-3) |
| 23 | 사소 | 테스트·출시 플래그·모니터링·오류 코드·호환성·담당·코퍼스 갱신 없음 | §4-1 오류 코드 · §9 단계 공개 · §10 · §11 · §8-5b |

리뷰에서 코드로 **맞다고 확인된** v1 주장: 온보딩 `console.log`·링크 0곳 · 프론트 계측 없음 · "볼 수 있는 곳" 외부 링크 · 좋아요 삭제/덮어쓰기 · `WorkSummaryDTO` · JWT 파싱 · m2 브랜치 동기 로그·V4 충돌·V8 · 네 플랫폼 `next_page`·`dominant_seed`·결정성 · 지연·메모리 수치 · `reviews.rating` 0~5 · CORS 헤더 허용 · `external_ranking` 플랫폼 범위.
리뷰의 "전역 react-query `staleTime` 5분" 지적은 번들에서 확인되지 않아(기본값 0 확인) 전역값과 무관하게 이 쿼리만 설정하는 방식으로 반영했다.
