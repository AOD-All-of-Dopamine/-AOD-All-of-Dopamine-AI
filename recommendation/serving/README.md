# 추천 서빙 (aod_serving)

평가된 추천기 네 개(Steam·TMDB·웹툰·웹소설)를 **결과를 바꾸지 않고** HTTP 로 감싸 백엔드가 부를
`POST /v1/recommend` 를 제공한다. 이 문서는 ① 라우터를 호출할 백엔드 개발자, ② 이 서비스를
추천 호스트에 배포·운영할 사람 둘을 대상으로 한다.

## 목차
1. [무엇인가](#1-무엇인가)
2. [API](#2-api)
3. [백엔드 연동 메모](#3-백엔드-연동-메모-4번-하위-프로젝트)
4. [이 PC 에서 실행](#4-이-pc-에서-실행)
5. [추천 호스트 배포](#5-추천-호스트-배포)
6. [코퍼스 교체·설정·운영 모드](#6-코퍼스-교체되돌리기--configjson--운영-모드)
7. [검증 도구](#7-검증-도구)
8. [알아 둘 것 / 알려진 한계](#8-알아-둘-것--알려진-한계)

---

## 1. 무엇인가

**엔진 컨테이너 4개(플랫폼당 1개) + 라우터 컨테이너 1개.** 설계 문서: `recommendation/REC_TAB_DESIGN.md`
(§4-2 백엔드↔라우터 계약 · §8 추천 서비스 · §8-6 서빙 컨테이너 구성).

```
backend (Spring) ──POST /v1/recommend──▶ rec-router (FastAPI, :8080)
                                              │ httpx, 엔진별 1.5s 제한, 동시 호출
                        ┌──────────┬──────────┼──────────┐
                        ▼          ▼          ▼          ▼
                   rec-steam   rec-tmdb  rec-webtoon rec-webnovel   (FastAPI, :8000 각자)
                   (steam/src) (tmdb/src) (webtoon/src)(webnovel/src)
```

**플랫폼당 프로세스(컨테이너)가 하나인 이유**: 네 플랫폼의 추천기 코드가 전부 최상위 패키지 이름으로
`src` 를 쓴다. 한 프로세스에 두 플랫폼을 같이 올릴 수 없어, 엔진 프로세스는 기동 시 `chdir` +
`sys.path` 조작(`aod_serving/engine/bootstrap.py:enter_platform`)으로 플랫폼 하나에 전용된다.
라우터는 엔진 코드도 numpy·pandas 도 import 하지 않는다(httpx 로만 엔진을 부른다).

**"랭킹 결과를 바꾸지 않는다"는 원칙과 그것을 지키는 장치**
- 엔진 어댑터(`aod_serving/engine/adapters/*.py`)는 각 플랫폼의 평가된 확정값(`PRODUCTION`)과
  평가 경로 함수(`next_page`/`Engine.next_page`)를 **그대로** 호출한다 — 랭킹 로직을 재구현하지 않는다.
- 전체 탭 혼합은 평가된 `crossdomain/mix.py` 의 `M6` 을 파일로 그대로 불러 쓴다
  (`aod_serving/router/mixing.py`) — 재구현이 아니라 import.
- 지연 개선(§8, Steam/TMDB/웹소설)은 전부 "결과 불변"이 전제다 — 각 `tools/{steam,tmdb,webnovel}_baseline.py`
  로 고친 코드 전/후 기준 목록을 만들어 id·점수가 완전히 같은지 비교했다(`--check`).
- 동일성 도구(`tools/identity.py`)가 L1(어댑터 결과 == 같은 프로세스에서 평가 경로 직접 호출) ·
  L2(어댑터 결과 == 커밋된 평가 기준 목록: TMDB `x27_tmdb_pages.json` · 웹툰 `t10_pages.json` ·
  웹소설 `w6_pages.json` · Steam `s3_pages.json`)를 검증한다. 자세한 내용은 §7.

---

## 2. API

키는 **전부 문자열**이다(백엔드 `aod_rec.corpus_map.corpus_key` 가 `text`, TMDB 자연 키가 문자열이기
때문). **TMDB 의 item_id 는 코퍼스 행 번호가 아니라 `movie_603`/`tv_1399` 꼴의 문자열이다** — 행 번호는
코퍼스 버전이 바뀌면 의미가 달라져 `corpus_map` 에 저장하기 위험해서, 엔진 어댑터가 안에서
`item_id ↔ 행 번호`를 변환한다(REC_TAB_DESIGN §4-2 와 다른 점, `aod_serving/common/models.py` 머리말).

### 2-1. 라우터 `POST /v1/recommend` (:8080)

```jsonc
// 요청
{ "tab": "all", "k": 20, "buffer": 10,
  "seeds":    { "steam": ["730"], "tmdb": ["movie_603"], "webtoon": [], "webnovel": [] },
  "disliked": {}, "excluded": {}, "seen": {} }

// 200
{ "items": [
    { "platform": "tmdb", "key": "movie_603", "rank": 0, "dominantSeed": "movie_603",
      "candidateSource": "content_sim", "isExploration": false, "propensity": 1.0,
      "score": { "final": 0.61, "sim": 0.55, "factors": {} }, "factorSchema": "tmdb.v1" } ],
  "exhausted": { "steam": false, "tmdb": false, "webnovel": true },
  "droppedSeeds": {}, "partial": [],
  "versions": { "router": "c8ec317", "engines": {
    "steam": { "sha": "c8ec317", "config": "a10cb55f0cc0", "corpus": "tags_full" } } } }
```

- 응답 필드 이름은 전부 pydantic `alias_generator=to_camel` 로 camelCase 로 나간다
  (`dominant_seed`→`dominantSeed`, `factor_schema`→`factorSchema`, `dropped_seeds`→`droppedSeeds` 등,
  `aod_serving/common/models.py`). **`score.factors` 의 안쪽 키(`rec_pct`·`tag_fit`·`has_mc`·
  `interest_pct`)는 예외다** — `factors` 는 `dict[str, float]` 라 필드 이름이 아니라 값이고, 이 값은
  어댑터가 담은 원래 표기(snake_case) 그대로 나간다(변환 안 됨).
- `exhausted`/`droppedSeeds` 는 **탭에서 실제로 부르는 플랫폼만** 키로 갖는다. `all` 탭은 Steam·TMDB·
  웹소설만 대상이다(§8-4, 웹툰은 포함 안 됨 — 사전등록 평가 후로 미뤘다).

**탭 → 엔진 매핑** (`aod_serving/router/service.py:TAB_PLAN`)

| 탭 | 호출 엔진 | 엔진 `k` | 결과 순서 |
|---|---|---|---|
| `all` | steam · tmdb · webnovel (시드가 있는 플랫폼만) | 50(고정) | `mix.M6(lists, seeds, k=k+buffer, episodes)` |
| `game` | steam | `min(k+buffer, 100)` | 엔진이 준 순서 그대로 |
| `movie` / `tv` | tmdb (`media=movie`/`tv`) | `min(k+buffer, 100)` | 엔진이 준 순서 그대로 |
| `webtoon` | webtoon | `min(k+buffer, 100)` | 엔진이 준 순서 그대로 |
| `webnovel` | webnovel | `min(k+buffer, 100)` | 엔진이 준 순서 그대로 |

- 엔진 호출은 **동시에**, 엔진별 제한 시간 `ENGINE_TIMEOUT_MS`(기본 1500ms, 연결 0.3초 고정)만큼
  기다린다. 시간 초과·연결 실패·엔진 쪽 200 이외 응답은 그 플랫폼을 `partial` 에 넣는다 —
  **`partial` 에 들어간(실패한) 플랫폼은 `exhausted` 에도 나타나지 않는다**(성공한 플랫폼만 채운다).
  **시드가 처음부터 없는 플랫폼은 아예 호출하지 않고 `exhausted: true`** 로 둔다(실패와 구별).
- `all` 탭에서 `partial` 에 들어간 플랫폼은 **M6 의 `seeds` 에서도 빠진다** — 그 몫이 남은 플랫폼에
  재분배된다. 결과가 바닥난(`exhausted: true`) 플랫폼은 평가된 규칙대로 빈 자리를 채우지 않는다.
- **필요한 엔진이 전부 실패**하면 `503 { "error": "engines_unavailable", "partial": [...] }` — 백엔드는
  대체 목록으로 간다.
- `versions.router` = 이미지 빌드 인자 `GIT_SHA`(§5), `versions.engines[platform]` = 그 엔진 응답의
  `version`(엔진 `GIT_SHA` · `config_hash` · `corpus_version`).
- `candidateSource: "content_sim"` · `isExploration: false` · `propensity: 1.0` 은 지금은 상수다(§8-e).
- **오류**: 요청 스키마 위반(키 타입이 문자열이 아님·허용 안 된 필드 등) → `422`(FastAPI 자동 검증,
  `RouterRequest` 는 `extra="forbid"`). 필요한 엔진이 전부 실패 → 위 `503`. 그 밖의 처리되지 않은
  예외는 전역 핸들러가 잡아 `500 { "error": "internal" }` 로 돌려준다(엔진 쪽 `app.py` 와 같은 모양).
- **입력 한도**: `k` 1~50(기본 20) · `buffer` 0~50(기본 10) · 플랫폼별 `seeds` ≤ 50 ·
  `disliked`+`excluded`+`seen` 합계 ≤ 5,000 — **전부 `RouterRequest` 자체에 걸려 있어** 어겼을 때
  깔끔한 `422` 다(`aod_serving/common/models.py:RouterRequest`, `seeds`/`disliked`/`excluded`/`seen`
  한도는 `model_validator` 가 플랫폼별로 검사한다). 라우터가 그 플랫폼을 부를 때 내부에서 만드는
  `EngineRequest` 에도 같은 한도가 다시 걸려 있는데, 이건 방어적 이중화일 뿐이다 — `router/service.py:one()`
  에서 `EngineRequest(...)` 생성을 그 플랫폼의 실패를 `partial` 로 돌리는 `try` **안**에서 하므로,
  설령 여기서 막혀도(예: 잔여 버그) 요청 전체가 아니라 그 플랫폼만 `partial` 이 된다.

`GET /health` — 라우터 자신은 **항상 200**. 매 호출마다 알고 있는 각 엔진의 `/health` 를 0.5초
제한으로 실시간 조회해 본문에 담는다(캐시 아님): `{ "ready": true, "router_sha": "…", "engines": { "steam": {…엔진 /health 그대로…} } }`.
**`/health` 는 `/v1/recommend`·`/engine/recommend` 와 달리 pydantic `response_model` 을 안 쓰고
일반 dict 를 그대로 돌려준다 — 그래서 이 안의 키는 camelCase 로 바뀌지 않고 코드에 적힌
snake_case 그대로 나간다**(`router_sha`·`corpus_version`·`engine_sha` 등, `router/app.py`·`engine/app.py`).

### 2-2. 엔진 `POST /engine/recommend` (:8000, 내부망 전용)

```jsonc
// 요청
{ "k": 50, "seeds": ["730", "570"], "disliked": [], "excluded": [], "seen": [], "media": null }

// 200
{ "platform": "steam",
  "items": [ { "key": "240", "rank": 0, "dominantSeed": "730",
               "score": { "final": 1.1936, "sim": 0.5880,
                          "factors": { "rec_pct": 0.9993, "quality": 1.0, "tag_fit": 0.75, "has_mc": 1.0 } },
               "episodeCount": null } ],
  "exhausted": false, "droppedSeeds": ["999999999"],
  "factorSchema": "steam.v1",
  "version": { "sha": "…", "config": "a10cb55f0cc0", "corpus": "tags_full" },
  "tookMs": 412 }

// 422 입력 한도 초과·형식 오류(FastAPI 자동 검증) 또는 { "detail": "<platform> 은 media 를 받지 않는다" }
// 503 { "error": "not_ready", "reason": "…" }  또는  { "error": "busy" }
```

- 한도: `k` 1~100 · `seeds` ≤ 50 · `disliked`+`excluded`+`seen` 합계 ≤ 5,000. `media` 는 TMDB 만
  받는다(`movie`|`tv`|`null`) — 다른 플랫폼에 주면 422.
- `exhausted` = 돌려준 개수 < `k`. 코퍼스 밖·형식이 틀린 시드는 예외 대신 제외되어 `droppedSeeds` 에 담긴다.
- **한 프로세스는 한 번에 한 요청만 계산한다**(전용 계산 스레드 1개, 이유는 §8-a). 대기 중인 요청이
  `MAX_QUEUE`(기본 8)를 넘으면 즉시 `503 busy`. 대기가 `QUEUE_DEADLINE_MS`(기본 1500ms)를 넘기면
  계산을 시작하지 않고 바로 `503 busy` 로 빠진다(라우터가 이미 포기한 요청을 뒤에서 계속 계산하지 않는다).
  처리량이 부족하면 `WORKERS`(uvicorn 프로세스 수, §8-b 의 대가 참고)로 늘린다.

```jsonc
// GET /health
// 200
{ "ready": true, "platform": "steam", "corpus_version": "tags_full", "engine_sha": "…",
  "config_hash": "a10cb55f0cc0", "arrow_pool": "system", "rss_mb": 1893.0, "rss_file_mb": 740.0,
  "rss_anon_mb": …, "cgroup": { "current_mb": …, "peak_mb": …, "anon_mb": …, "file_mb": … } }
// 503 { "ready": false, "reason": "loading" | "artifact_invalid: …" | "config_invalid: …" | "load_failed: …" }
```

기동 순서: 아티팩트 검증 → 설정(`config.json`) 적용 → 적재 → **예열 1회**(dataset 첫 키로
`next_page`) → `ready: true`. 검증·설정 오류가 나도 프로세스는 살아 있고 `/health` 가 503 과 구체적
사유를 계속 보여준다(재시작 루프로 사유가 묻히지 않는다, §6).

---

## 3. 백엔드 연동 메모 (4번 하위 프로젝트)

- **`aod_rec.corpus_map.corpus_key` 에 넣을 값 = 위 API 의 문자열 키 그대로다** — Steam 은 `steam_appid`
  문자열, TMDB 는 `movie_{id}`/`tv_{id}`, 웹툰·웹소설은 `item_id` 문자열.
- 라우터가 아는 `platform` 이름은 `steam`/`tmdb`/`webtoon`/`webnovel` 넷 뿐이다(`mix.py` 내부의
  `wn` 표기는 라우터가 변환해 숨긴다 — 백엔드는 신경 쓸 필요 없음).
- 요청 시 `k + buffer` 개를 받는다. **백엔드가 DB 에 없는 작품·성인 작품 등을 거른 뒤 `k`(기본 20)개를
  채운다** — 버퍼에서 걸러진 것도 왜 빠졌는지 로그에 남기는 편이 커버리지 추적에 유리하다
  (REC_TAB_DESIGN §4-2, 구글 Rules of ML #6).
- `seen`/`disliked`/`excluded` 는 전부 **플랫폼별 키 목록**(딕셔너리 — `{"steam": [...], "tmdb": [...]}`).
  `disliked` 의 실제 동작은 플랫폼마다 다르다(REC_TAB_DESIGN §8-3, `engine/adapters/*.py`) — Steam 만
  제외 + 유사 감점(랭커 안에서, 이 API 로는 안 드러남)이 있고, 웹툰은 같은 시리즈까지 제외한다.
  TMDB 는 `disliked` 인자가 없어 그냥 `seen`/`excluded` 와 합쳐져 **제외만** 된다. 웹소설도 인자가
  없어 마찬가지로 합쳐지는데(`seen`·`disliked`·`excluded` 전부 `seen_ids` 하나로), 이 경로 자체가
  `drop_excluded_series` 로 **같은 판본(작가·제목 키)까지 함께 제외**한다 — 의도가 아니라 지금 엔진의
  한계다(알려진 차이). 시드이면서 동시에 `disliked` 에 있으면 시드에서 빠진다(넷 다 공통).
- 지연 예산(REC_TAB_DESIGN §8-5): **백엔드 → 라우터 읽기 제한 2.0초**, **라우터 → 엔진 1.5초**
  (`ENGINE_TIMEOUT_MS`, 컨테이너 환경변수로 조정 가능). 엔진 쪽 대기열 마감은 `QUEUE_DEADLINE_MS`
  (엔진 기본 1500ms) — 라우터 예산과 맞춰 둔다.
- 응답의 `versions`(라우터 sha · 엔진별 sha/config_hash/corpus) 를 `rec_request` 로그에 남길 것 —
  결과가 바뀐 원인(코드 배포·코퍼스 교체·설정 변경)을 나중에 구분하는 유일한 단서다.
- **전체(`all`) 탭은 Steam·TMDB·웹소설만 섞는다 — 웹툰은 자기 탭(`webtoon`)에서만 나온다**(§8-4,
  전체 탭 웹툰 포함은 사전등록 평가 후로 미뤄져 있다).

---

## 4. 이 PC 에서 실행

프로젝트 코드는 **Docker 안에서만** 돈다(이 PC 파이썬은 버전이 다르고 numpy 등이 없다). 전부
`recommendation/serving/scripts/dev.sh` 를 거친다:

| 명령 | 동작 |
|---|---|
| `dev.sh build` | `dev` 타깃 이미지 빌드(`Dockerfile`, 레포는 이미지에 안 담고 마운트로 쓴다) |
| `dev.sh test [pytest 인자…]` | 레포를 **읽기 전용**으로 마운트하고 pytest(운영 컨테이너도 `read_only: true` 라 이 방식이 실제와 더 가깝다) |
| `dev.sh run <명령…>` | 레포를 읽기·쓰기로 마운트하고 임의 명령 실행 — 기준 목록·`manifest.json` 생성용 |
| `dev.sh up` | compose 5개 서비스(엔진 4 + 라우터) `--build` 로 빌드 후 기동(`compose.yaml` + `compose.local.yaml`) |
| `dev.sh down` | compose 스택 내림 |
| `dev.sh net <명령…>` | compose 망(`aod-rec_aod-rec`) 안에서 dev 이미지로 명령 실행 — `tools/e2e.py`·`tools/loadgate.py` 용 |

아티팩트(임베딩·parquet)는 `recommendation/{platform}/artifacts/{corpus_version}/` 에 있어야 한다.
**`corpus_embeddings.npy` 는 네 플랫폼 모두 `.gitignore` 대상**이다(Steam 만 764MB) — 평가를 돌린
개발 머신이나 평가 서버에서 복사해 와야 하고, 리포를 새로 받은 것만으로는 서비스가 뜨지 않는다.
`corpus_index.parquet`·`dataset.parquet` 와, Steam·웹툰·웹소설의 `manifest.json` 은 git 에 커밋돼
있다(TMDB 는 폴더 전체가 gitignore 라 `manifest.json` 도 없다 — §5).

**테스트 종류**
- **hermetic**(마커 없음): `tests/conftest.py` 가 만드는 작은 가짜 코퍼스(6행)만으로 돈다 — 아티팩트가
  없어도 통과한다. `dev.sh test -q -m "not artifacts"` → **130 passed, 4 deselected, 약 8초**
  (이 PC, dev 컨테이너 안에서 실측).
- **`-m artifacts`**: 실제 코퍼스(위 경로)가 로컬에 있어야 돈다(없으면 `skip`). 전체(`dev.sh test -q`,
  마커 없이) 를 이 PC 에서 돌리면 **134 passed, 약 3분 20초**(Steam 아티팩트 적재·다회 `next_page` 호출이
  대부분을 차지한다) — CI(GitHub Actions, `.github/workflows/serving-tests.yml`)에는 아티팩트가 없으므로
  이 4개는 CI 에서 돌지 않고(`-m "not artifacts"`), 나머지 130개만 PR 마다 돈다.

---

## 5. 추천 호스트 배포

### 5-1. 아티팩트 배치

`/srv/aod-artifacts/{platform}/{corpus_version}/` 에 `manifest.json`·`corpus_embeddings.npy`·
`corpus_index.parquet`·`dataset.parquet`(+ 플랫폼별 부가 파일)를 둔다(REC_TAB_DESIGN §8-6).

- **Steam·웹툰·웹소설**: `corpus_index.parquet`·`dataset.parquet`·`manifest.json` 이 이미 **git 에
  추적**돼 있다(gitignore 는 `corpus_embeddings.npy` 만 막는다) — 호스트에 파일을 복사한 뒤
  `python -m aod_serving.tools.make_manifest --platform <p> --dir /srv/aod-artifacts/<p>/<corpus>`
  를 돌리면 **새로 만들지 않고 커밋된 manifest 의 sha256 으로 지금 폴더를 검증만 한다** — 즉 "복사한
  `corpus_embeddings.npy` 가 평가 때와 바이트 단위로 같은가"를 확인하는 무결성 점검이다. 통과하면
  `기존 manifest 와 일치 — 검증 통과 · …행 × …` 를 출력하고 끝난다(`tools/make_manifest.py`).
- **TMDB**: `recommendation/tmdb/artifacts/tmdb_v1/` 폴더 전체가 gitignore 라 커밋된 manifest 가
  없다. 파일을 호스트에 올린 뒤 같은 명령으로 **처음부터 만든다**(신규 생성 경로).
- **`--force`**: manifest.json 이 이미 있어도 **다시 만든다**(sha256 을 지금 폴더 기준으로 재계산). 무결성
  기준 자체를 새로 세우는 것이므로, 정말로 새 코퍼스를 게시할 때만 쓴다 — 복사가 깨졌는지 확인하는
  용도로는 쓰면 안 된다(깨진 파일에 기준을 맞춰버린다).

### 5-2. 이미지 빌드·태그

```bash
docker build -f serving/Dockerfile --target engine --build-arg PLATFORM=steam \
  --build-arg GIT_SHA="$(git rev-parse --short HEAD)" -t <REGISTRY>/aod-rec-engine:steam-<TAG> recommendation/
docker build -f serving/Dockerfile --target router \
  --build-arg GIT_SHA="$(git rev-parse --short HEAD)" -t <REGISTRY>/aod-rec-router:<TAG> recommendation/
```

`tmdb`·`webtoon`·`webnovel` 도 `PLATFORM` 만 바꿔 동일하게 4장 + 라우터 1장, 총 5장. 빌드 컨텍스트는
`recommendation/`(Dockerfile 은 `serving/Dockerfile`). `GIT_SHA` 가 `/v1/recommend`·`/engine/recommend`
응답의 `versions.*.sha` 와, `/health` 의 `engine_sha`/`router_sha` 로 그대로 나간다(안 주면 `dev`).

### 5-3. `compose.yaml` 환경변수

`compose.yaml` 은 추천 호스트용, 이 PC 에서는 `compose.local.yaml` 로 이미지 대신 로컬 빌드·아티팩트
바인드 마운트·127.0.0.1 포트로 덮어쓴다. 실제 `mem_limit`/`cpus` 값과 그 근거는 부하 관문 결과이므로
**`compose.yaml` 과 `LOADGATE_RESULTS.md` 를 직접 참고**(이 문서에는 옮겨 적지 않는다 — 관문을 다시
돌릴 때마다 값이 바뀔 수 있다). 환경변수 **이름**은 안정적이다:

| 변수 | 기본값 | 의미 |
|---|---|---|
| `REGISTRY` | `local` | 이미지 레지스트리 접두사 |
| `ENGINE_TAG` / `ROUTER_TAG` | `dev` | 이미지 태그 |
| `BIND_IP` | `127.0.0.1` | 라우터 포트를 바인딩할 호스트 IP — **운영에서는 사설 IP** |
| `ARTIFACTS_BASE` | `/srv/aod-artifacts` | 아티팩트 루트 |
| `STEAM_CORPUS`/`TMDB_CORPUS`/`WEBTOON_CORPUS`/`WEBNOVEL_CORPUS` | `tags_full`/`tmdb_v1`/`wt_v1`/`wn_v6` | 플랫폼별 `CORPUS_VERSION` |
| `SERVING_MODE` | `prod` | `dev`\|`prod` — §6 |
| 엔진 `WORKERS` | `1`(uvicorn 기본) | 워커 프로세스 수 — §8-b 의 대가 참고 |
| 엔진 `MAX_QUEUE` | `8` | 대기열 최대 길이(코드 기본값, `EngineState`) |
| 엔진 `QUEUE_DEADLINE_MS` | `1500` | 대기열 마감(ms) |
| 라우터 `ENGINE_TIMEOUT_MS` | `1500`(compose.yaml 이 명시) | 엔진 호출 제한 시간(ms), 연결 0.3s 는 코드 고정값 |

### 5-4. 기동 순서·헬스체크

엔진이 각자 적재 → 예열 → `/health` 200 이 될 때까지 컨테이너 헬스체크가 폴링한다(`compose.yaml`
`x-engine.healthcheck` — 파이썬으로 `/health` 를 직접 호출, 3회 재시도). 라우터는 `depends_on` 으로
**엔진 4개가 전부 `service_healthy` 여야** 시작한다(초안 그대로 — 엔진 하나가 안 뜬 채로 라우터를 먼저
띄울지는 REC_TAB_DESIGN §8-6 이 "출시 전 결정" 으로 남겨 뒀다). 라우터 자신도 `/health` 헬스체크를 갖는다.

### 5-5. 보안

라우터 포트(`8080`)는 `BIND_IP`(사설 IP)에만 바인딩한다 — 백엔드 API 서버의 보안 그룹만 그 포트를
열어 준다. **엔진 포트(`8000`)는 호스트에 노출하지 않는다** — `compose.yaml` 의 엔진 서비스에는
`ports:` 가 아예 없고 `aod-rec` 내부 네트워크로만 라우터와 통신한다.

---

## 6. 코퍼스 교체·되돌리기 · `config.json` · 운영 모드

**교체**: 새 코퍼스 폴더를 `ARTIFACTS_BASE` 밑에 두고(§5-1 검증 통과 후) 그 플랫폼의 `CORPUS_VERSION`
환경변수만 바꿔 해당 엔진 컨테이너를 재시작한다. **되돌리기**: 이전 코퍼스 버전 이름으로 다시 바꾸고
재시작(이전 폴더를 지우지 않았다면). 코퍼스 갱신·설정 변경은 이 두 값만 바뀌므로 **이미지 재빌드가
필요 없다**(REC_TAB_DESIGN §8-6 "재빌드가 필요한 경우" 표 — 엔진 코드·계약 스키마가 바뀔 때만 재빌드).

**`config.json`**(코퍼스 폴더 안, 선택) — 코드 기본값(확정값 `PRODUCTION`)의 **허용된 키만** 덮어쓴다.
알 수 없는 키·타입 오류는 기동 실패(`/health` 가 `config_invalid: …` 로 보여준다). 허용 키는
`aod_serving/engine/overrides.py:ALLOWED` 에 정의돼 있다:

| 플랫폼 | `production` 허용 키 | `postprocess` 허용 키 |
|---|---|---|
| steam | `strategy`(str) · `rec_boost`·`quality_w`·`quality_cap`(수) · `quality_src`(str) · `tag_w`·`mc_w`·`trend_weight`(수) | (없음) |
| tmdb | `strategy`(str) · `hub_lambda`·`rating_boost`·`media_w`·`genre_w`·`align_w`·`director_w`(수) | `franchise_max`·`seed_franchise_max`(정수) · `interleave`(불리언) · `drop_seed_iter`(str) |
| webtoon | `strategy`(str) · `pop_boost`·`hub_lambda`·`star_boost`·`tag_w`·`creator_w`(수) · `tag_drop_genre`(불리언) · `dislike_w`·`dislike_floor`(수) | `series_max`·`artist_max`(정수) · `drop_adult`(불리언) |
| webnovel | `strategy`(str) · `pop_boost`(수) · `min_interest_count`(정수 또는 null) · `drop_excluded_series`(불리언) | (없음) |

top-level 키는 `production`/`postprocess`/`verdict`/`approved` 넷만 허용된다. 예:
```json
{ "production": { "tag_w": 0.0 }, "postprocess": {}, "approved": true, "verdict": { "id": "D-xx" } }
```

**운영 모드 승인 규칙**: `SERVING_MODE=prod` 이고 그 코퍼스 버전이 평가 기준 코퍼스
(`tags_full`/`tmdb_v1`/`wt_v1`/`wn_v6`, `BASELINE_CORPORA`)가 **아니면**, `config.json` 에
`"approved": true` 와 `"verdict": {"id": …}` 가 없으면 기동을 거부한다(`config_invalid: 운영 모드: 새
코퍼스 … 에 approved: true 가 없다`). `SERVING_MODE=dev` 에서는 이 검사를 건너뛴다.

**계약 위반 시 `/health` 가 보여 주는 사유** (문자열 그대로):
- `"loading"` — 아직 적재 중
- `"artifact_invalid: <상세>"` — manifest 없음/불일치, sha256 불일치, 필수 컬럼 없음/비었는데 관련
  보정 항이 켜져 있음, 임베딩 모양·노름 이상, 행 수 불일치, 키 중복 등(`engine/contract.py`)
- `"config_invalid: <상세>"` — `config.json` 파싱 실패·허용 안 된 키·타입 오류·운영 모드 승인 누락
- `"load_failed: <예외타입>: <메시지>"` — 그 밖의 적재 중 예외

---

## 7. 검증 도구

모두 `recommendation/serving/aod_serving/tools/` 아래, `dev.sh run`/`dev.sh net` 으로 돌린다.

| 도구 | 무엇을 확인하나 | 실행 |
|---|---|---|
| `identity` | **L1**: 어댑터 결과 == 같은 프로세스에서 평가 경로를 직접 부른 결과(id·순서·점수 완전 일치, 싫어요/제외/seen 조합 포함). **L2**: 어댑터 결과 == 커밋된 평가 기준 목록 — TMDB/웹툰/Steam 은 id 완전 일치 + 점수 오차 1e-6, **웹소설은 (제목, 작가) 키로 비교**(동점 판본 때문, §8-c) | `dev.sh run python -m aod_serving.tools.identity --platform steam [--level l1\|l2\|all] [--allow-ties] [--limit N]` |
| `e2e` | compose 로 띄운 상태에서 **L3**(라우터 전체 탭 결과 == 엔진 직접 호출 + `mix.M6` 재현) + 계약(스키마·422·코퍼스 밖 시드→`droppedSeeds`·빈 시드→`exhausted`) + 장애(엔진 중단 시 `partial`/503) | `dev.sh net python -m aod_serving.tools.e2e [--expect-partial steam] [--bundles N]` |
| `loadgate` | 부하·메모리 관문(시드 1·10·50 × seen 0·200·500 × 동시 1·5·20) — **결과·판정은 `LOADGATE_RESULTS.md` 참고**(이 README 에는 수치를 옮기지 않는다 — 갱신 진행 중일 수 있다) | `dev.sh net python -m aod_serving.tools.loadgate --out <경로> [--platforms steam,tmdb,…] [--per-cell N] [--skip-router\|--skip-engines]` |
| `{steam,webnovel,tmdb}_baseline` | 코드 변경(주로 지연 개선)이 **결과를 바꾸지 않았는지** — `--out` 으로 기준 목록 생성, `--check` 로 비교(같은 머신·같은 스레드 설정에서 **비트 단위** 완전 일치가 기준; 다른 머신 비교는 `--allow-ties` 로 동점 자리의 순서 차이만 허용), `--bench` 로 지연만 측정, `--cases <부분문자열>` 로 사례 필터 | `dev.sh run python -m aod_serving.tools.steam_baseline --check <기존 파일>` |
| `scripts/stress_engine.sh` | 엔진 세그폴트 회귀(§8-a) — 컨테이너를 반복 기동하며 과거 크래시가 났던 세 지점(기동 직후 첫 요청 · 동시 요청 묶음 · 유휴 10초 초과 후)을 다시 때린다 | `scripts/stress_engine.sh <platform> <runs> <reqs>` |
| `scripts/repro_segv.py` | 소켓 없이 스레드 친화도 가설만 빠르게 재현(모드: `main`/`fresh`/`keepalive`/`pool`/`dedicated`/`anyio`) | `dev.sh run env PLATFORM=webtoon python scripts/repro_segv.py --mode fresh --iters 20` |

---

## 8. 알아 둘 것 / 알려진 한계

**(a) PyArrow mimalloc 세그폴트와 두 겹의 방어.** pyarrow 25 의 기본 Arrow 메모리 풀(mimalloc)은
스레드마다 힙을 두는데, Arrow 를 쓰던 스레드가 **종료**하면 그 TLS 블록을 재활용한 새 스레드의 첫
Arrow 할당이 `mi_thread_init()` 에서 SIGSEGV 를 낸다(웹툰은 예전 구조에서 첫 요청 100% 재현). 방어
① 엔진의 모든 계산(적재·예열·모든 요청)은 **전용 계산 스레드 하나**에서만 돈다(`ThreadPoolExecutor
(max_workers=1)`, `engine/app.py`) — Arrow 를 만지는 스레드가 프로세스 수명 내내 하나뿐이라 전제 자체가
없어진다. ② `ARROW_DEFAULT_MEMORY_POOL=system`(glibc malloc, 스레드 지역 힙 없음) 을 `aod_serving/native.py` 의
코드(패키지 import 시점)와 Dockerfile `ENV` 둘 다에 박아 둔다 — 할당자만 바뀌므로 수치 결과는 그대로다.

**(b) 한 프로세스는 한 번에 한 요청.** 위 (a) 의 결과로 계산 스레드가 하나뿐이라 동시 요청은 줄을 선다
(선입선출 대기열, `MAX_QUEUE`). 대기가 `QUEUE_DEADLINE_MS` 를 넘으면 계산 없이 바로 `503 busy`. 처리량은
`WORKERS`(uvicorn 프로세스 수)로 늘릴 수 있지만 **워커마다 코퍼스 전체를 따로 적재**한다(Steam 이면
워커당 상주 메모리 약 1.9GB 추가) — `mem_limit` 을 워커 수에 맞춰 올려야 하고, `/health` 는 무작위
워커 하나가 답하므로 다른 워커가 준비됐어도 그 워커가 적재 중이면 503 일 수 있다.

**(c) 웹소설 동점 판본.** 같은 작품의 본편과 맛보기판(10화)은 임베딩·점수가 완전히 같은 동점이라, 그
순서는 pandas 기본 정렬(불안정)이 정하고 CPU 의 SIMD 경로에 따라 달라질 수 있다. 서빙에서는 이 순서를
고치지 않는다(평가된 순서를 바꾸는 일이라서) — 그래서 동일성 비교는 id 가 아니라 **(제목, 작가) 키**로
한다. 후속 제안(범위 밖, 사전등록 필요): 동점이면 관심 수가 더 큰 판본을 결정적으로 고르게 — M6 의
"20화 미만 제외"와 겹치면 맛보기판이 뽑힌 작품이 전체 탭에서 통째로 사라질 수 있다.

**(d) 점수 인자(`factors`)는 플랫폼마다 다르다.** 어댑터는 **랭커가 이미 프레임에 남기는 컬럼만**
싣는다(랭커를 계측해 새 인자를 뽑는 일은 하지 않는다). Steam(`rec_pct`·`quality`·`tag_fit`·`has_mc`)과
웹소설(`interest_pct`)만 채워지고, **TMDB·웹툰은 항상 빈 객체 `{}`** 다(REC_TAB_DESIGN §8-2 대비 축소—
TMDB·웹툰 랭커 계측은 후속 범위).

**(e) 탐색 슬롯이 없다.** `candidateSource` 는 항상 `"content_sim"`, `isExploration` 은 항상 `false`,
`propensity` 는 항상 `1.0` — 지금은 상수다.

**(f) BLAS 스레드 수에 따라 Steam 점수 끝자리가 달라질 수 있다.** 스레드 수가 바뀌면 행렬 곱의 누적
순서가 바뀌어 부동소수점 결과가 미세하게 달라진다(id 목록·순서는 그대로). 그래서 `OMP_NUM_THREADS`
등은 서빙 코드가 일부러 건드리지 않고 `compose.yaml` 환경변수로 고정한다 — 같은 머신에서의 완전 일치
확인은 `*_baseline --check`, 다른 머신 비교는 `identity --allow-ties`.

**(g) 문서(REC_TAB_DESIGN) 대비 달라진 점** (구현 중 확인된 것들)
1. TMDB 키는 `item_id` 문자열이고, API 의 모든 키가 문자열이다(§4-2, §2 참고).
2. 점수 인자는 기존 프레임 컬럼만 싣는다 — TMDB·웹툰 랭커 계측은 후속(§8-2, 위 (d)).
3. **Steam·웹소설·TMDB 랭커/후처리의 결과 불변 지연 개선을 이번 범위에 포함했다** — §8-5 부하 관문을
   통과시키기 위해서였고, 각각 `*_baseline --check` 로 기존 결과와 완전 일치함을 확인했다.
4. 부하 관문의 합격 판정은 동시 1·5 만 보고 동시 20 은 참고치, 호스트는 8GB 기준(실행 설계에서 이미
   결정된 값 — §7 `loadgate`, 실측 수치는 `LOADGATE_RESULTS.md`).
5. 웹소설 동일성은 id 가 아니라 (제목, 작가) 키 기준이다(위 (c)).
6. ECR 푸시·배포 자동화는 추천 호스트가 생긴 뒤의 범위다. 대신 아티팩트 없이 도는 테스트와 이미지
   빌드 가능성을 GitHub Actions(`.github/workflows/serving-tests.yml`)로 PR마다 확인한다.

**(h) 라우터 입력 한도는 라우터 자신이 검증한다(수정 완료).** 플랫폼별 `seeds` ≤ 50·
`disliked+excluded+seen` ≤ 5,000 한도는 `RouterRequest` 의 `model_validator` 가 플랫폼별로 검사해
어겼을 때 `422` 로 막는다 — 엔진까지 가지 않는다(`tests/test_models.py`, `tests/test_router_app.py`).
`router/service.py:one()` 에서 만드는 `EngineRequest(...)` 는 같은 한도를 다시 검사하지만, 이제는
그 플랫폼의 실패를 `partial` 로 돌리는 `try` **안**에서 만들어 방어적 이중화로만 남아 있다 — 설령
여기서 막혀도 요청 전체가 아니라 그 플랫폼만 `partial` 이 된다(`tests/test_router_service.py`).
라우터에는 그 밖의 처리되지 않은 예외를 잡는 전역 핸들러도 생겼다(엔진 쪽 `app.py` 와 같은 모양,
`500 { "error": "internal" }`). §2-1 에도 같은 내용을 적어 뒀다.
