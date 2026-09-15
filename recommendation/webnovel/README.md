# 웹소설 추천 (webnovel)

네이버 시리즈 웹소설 대상 오프라인 배치 추천 파이프라인.
`recommendation/steam/` 의 검증된 구조를 이식하고 도메인 술어만 교체했다.

```
크롤 → 정제 → 표현 → 임베딩 → [표현 진단 게이트] → 검색·랭킹 → 후처리 → 판정
```

---

## 1. Steam 판과 무엇이 같고 무엇이 다른가

**같은 것** — 구조와 계약. `metrics.py`(NDCG/P@k), `personalization/*`(시드 로딩 →
전체 코퍼스 유사도 → 집계 → 랭킹), `next_page()` 의 새로고침·제외 집합 계약,
후처리 순서(hard filter → 시리즈 상한 → 다양성 → Top-N), 단계적 임베딩(`--append` 시
`embedding_row` 연속성 보장), 표현 실험 분리(`--out artifacts/rep_vN`).

**다른 것** — 컬럼과 술어.

| | Steam | 웹소설 |
|---|---|---|
| 아이템 키 | `steam_appid` | **`item_id`** (도메인 중립) |
| 인기도/품질 신호 | `recommendations_total` (결측 = 이진 신호) | **`interest_count` (관심 수, 연속값)** |
| 품질 하한 | `require_known_reviews` (있/없) | **`min_interest_count`** (임계값) |
| 연령 필터 | `ADULT_GENRES` 장르 태그 | **`age_limit`** (19세 이용가 제외) |
| 시리즈 마커 | 로마숫자 · edition · GOTY | **`n부` · 외전 · 시즌n · [단행본] · [독점] · 개정판** |
| 다양성 축 | 시드 인터리빙 + 시리즈 상한 | + **작가 상한**(`cap_author`) |
| 임베딩 텍스트 | 설명 + 장르 + 모드 (**이름 제외**) | **제목** + 장르 + 줄거리 |

제목을 넣는 이유: 웹소설 제목은 `"퇴사 후 아포칼립스로 출근합니다"` 처럼 그 자체가
로그라인이다. 줄거리가 카피 문구뿐인 꼬리 작품에서는 사실상 유일한 의미 신호가 된다.

작가 상한을 추가한 이유: 웹소설 작가는 제목이 전혀 다른 작품을 여러 편 내는데
문체·소재가 임베딩상 매우 가까워서 시리즈 상한만으로는 한 작가가 페이지를 점령한다.
게임은 개발사 다작이 이만큼 유사하지 않아 Steam 에는 없던 축이다.

---

## 2. 데이터 소스 실측 (2026-07-29)

백엔드(`-AOD-All-of-Dopamine-back`)의 Java `NaverSeriesCrawler` 를 Python 으로 포팅했다.
포팅하며 확인한 것들 — **일부는 백엔드 크롤러의 현재 동작과 다르다.**

### 목록에 10,000개 하드 캡
`categoryProductList.series` 는 `page=400` 까지 25개씩 주고 401부터 0개다.
단일 목록으로는 1만 개가 최대이므로 **장르 × 완결여부로 슬라이스**해서 열거하고
`productNo` 로 dedup 한다. 장르 단독 열거 실측(캡 = 그 장르에 1만 개 이상 존재):

| 장르 | 로맨스 | 로판 | 판타지 | 현판 | BL | 무협 | 라이트노벨 | 미스터리 |
|---|---|---|---|---|---|---|---|---|
| 열거 가능 | 10,000 `CAP` | 10,000 `CAP` | 10,000 `CAP` | 10,000 `CAP` | 10,000 `CAP` | 6,275 | 1,125 | 700 |

합계 약 58,000건(중복 포함).

### 19금은 로그인 리다이렉트로 막힌다 — 열거의 절반이 빠진다
Java 는 `#adult_msg` / `enctp=19` 마커로 판정하는데, 실제 차단은 **네이버 로그인 페이지로의
302 리다이렉트**(`nidlogin.login?svctype=128...`)다. `requests` 가 리다이렉트를 따라가므로
status 는 200 이고, 그대로 파싱하면 **제목이 `SERIES`(사이트명)인 빈 레코드**가 코퍼스에
들어간다. `is_redirected_away()` 로 막고 `data_loader` 에 방어를 하나 더 뒀다.

파일럿 200건 중 107건(53%)이 이 경우였다. → **열거 5.8만 ≈ 실제 코퍼스 2.7만.**

### `firstDate` 는 수집하지 않는다 (백엔드 버그)
Java 는 `volumeList.series?sortOrder=ASC` 의 `lastVolumeUpdateDate` 첫 등장값을
1화 등록일로 저장한다. 실측하면 그 필드는 **회차별 값이 아니라 응답 시각**이다 —
104화짜리 작품의 전 회차가 동일한 값을 갖고, 같은 작품을 ASC/DESC 로 부르면 값이
달라진다(`00:00:25` vs `23:00:09`). 즉 백엔드 `webnovel` 의 `release_date` 는 계속
"오늘"이 들어가 왔다. 이 엔드포인트에 회차별 발행일은 존재하지 않아 호출을 뺐고
(작품당 요청 절반), 신작/트렌드 신호는 보류했다.

### 인기도 셀렉터가 드리프트됐다
Java 의 1차 셀렉터 `a.btn_download > span`(관심 수)은 **현재 페이지에 없다.**
`div.end_head` 텍스트(`"평점 10.0 관심 2 공유"`)가 유일한 경로다 — Java 의 폴백이
현재의 정답이라 그것을 1차 경로로 삼았다.

**평점은 랭킹에 쓰지 않는다.** 참여자 수가 노출되지 않아 `관심 2 / 평점 10.0` 이 흔하다.
이 도메인의 인기도·품질 신호는 평점이 아니라 **관심 수**다.

### 줄거리는 걱정보다 두껍다
49건 실측: **중앙값 482자**, p25 346자, min 76자, 30자 미만 0건 —
Steam 의 `short_description`(300자 내외)보다 오히려 길다.
다만 카피 문구만 있는 꼬리 작품은 존재한다(실측 22자 사례).

> 초기 조사에서 "중앙값 53자"로 본 것은 **측정 오류**였다. 정규식이 첫 `</div>` 에서
> 끊겨 중첩 마크업 안의 본문을 잘라먹었다. bs4 로 뽑으면 위 수치가 나온다.
> 얇은 것은 줄거리가 아니라 **장르**다 — 작품당 1개 coarse 라벨뿐이다.

---

## 3. 실행

```bash
cd recommendation/webnovel
pip install -r requirements.txt      # + beautifulsoup4, lxml

# 1) 크롤 (이어받기 기본, 목록은 data/product_ids.json 에 캐시)
python -m src.crawl_naverseries --out data/webnovels.jsonl --rps 0.5 [--limit N]

# 2) 정제 → artifacts/wn_v1/{dataset.parquet, dataset_profile.json}
python -m src.data_loader

# 3) 표현 (rep_v2 는 --with-meta 로 작가/출판사 편입, 반드시 --out 분리)
python -m src.text_builder [--out artifacts/rep_v2] [--with-meta]

# 4) 임베딩 (단계적)
python -m src.embed_qwen --only-with-interest   # 1단계: 관심 수 있는 작품
python -m src.embed_qwen --append               # 2단계: 나머지

# 5) 표현 진단 ← 판정 전 게이트
python -m src.diagnose_representation

# 6) 추천
python -m src.personalized_retrieve --liked-ids 14504924 ... --top-n 20
```

환경변수는 `.env` (`.env.example` 참고). `AOD_WEBNOVEL_DATA` 가 `webnovels.jsonl` 의
디렉터리를 가리킨다. `AOD_ARTIFACTS` 로 아티팩트 디렉터리를 갈아끼울 수 있다.

---

## 4. 표현 진단 게이트 (`src/diagnose_representation.py`)

이 도메인은 장르가 1개 라벨뿐이라 임베딩이 "전부 비슷비슷한" 공간을 만들 위험이 있다.
그러면 랭킹·후처리를 아무리 손봐도 소용이 없다. **사람 판정에 들어가기 전에** 여기를 통과시킨다.

| 지표 | 합격선 | 왜 |
|---|---|---|
| 무작위 쌍 평균 유사도 | `< 0.45` | Steam 기준선: 무관 텍스트 0.229 / 노이즈 오염 0.519 |
| 같은 장르 − 다른 장르 유사도 | `> 0.05` | 장르조차 못 나누면 취향은 당연히 못 잡는다 |
| Top-10 이웃 장르 일치율 | `> 0.5` | 실제 추천이 하는 일에 가장 가까운 측정 |

불합격이면 표현을 바꾼다 (아래 순서대로, 매번 `--out` 분리 후 재측정):

1. **rep_v2 — 메타데이터**: 작가/출판사/완결여부/분량대 편입 (`--with-meta`). 비용 0.
2. **rep_v3 — fun_tag 규칙 매칭**: `origin/feature/m1-content-intelligence` 의
   `resources/seed_fun_tags.yaml`(회귀/환생/빙의/먼치킨/사이다/고구마/피폐/혐관/집착/성좌물
   … 45개)을 제목+줄거리에 규칙 매칭. 웹소설 제목은 클리셰를 대놓고 노출해서 규칙만으로도
   상당 부분 잡힌다.
3. **rep_v4 — LLM 추출**: m1 트랙처럼 LLM 으로 태그 추출. 비용이 가장 커서 마지막.

---

## 5. 테스트

```bash
python -m pytest tests/ -q
```

- `test_crawl_naverseries.py` — 한국어 수량 파서(`2억 5,006만`/`139.3만`/`2.5천`),
  `end_head` 관심 수 추출, 19금 거부, 작가/출판사가 장르로 새지 않을 것, 줄거리 최장 노드 선택
- `test_data_loader.py` — 합산 길이 게이트, `SERIES` 제목 거부, 결측/0 구분, 중복 제거
- `test_postprocess.py` — 한국어 시리즈 판정, 판본 통합, 연령·관심 수 필터, 작가 상한, 라운드로빈
- `test_refresh.py` — **새로고침 계약** (아티팩트 있을 때만 실행)

`test_refresh.py` 는 Steam 에서 실제로 터졌던 문제를 성질로 고정한다:
제외 집합이 없으면 같은 목록이 나온다(버그가 아니라 성질 — 호출자가 `seen_ids` 를 누적할
책임), 새로고침 간 중복 없음, 한 시드가 페이지의 70% 초과 금지, 3페이지까지 품질 하한 유지.

---

## 6. 아직 안 한 것

- **판정** — 20개 프로필 생성 → blind xlsx → NDCG/P@10. Steam 과 같은 지표로 비교해야
  이 도메인의 실제 성능을 안다. `REFRESH_POP_BOOST=0.15` / `REFRESH_MIN_INTEREST=100` 은
  **Steam 에서 가져온 출발점이지 이 도메인에서 검증된 값이 아니다.**
- **전체 크롤** — 파일럿 이후 5.8만 열거 → 2.7만 코퍼스
- 카카오페이지 — `keywords[]` 를 주는 유일한 소스라 장르가 얇은 문제를 직접 보완한다
  (GraphQL 경로, 스케줄러 없이 수동 트리거만 존재)
- 트렌드/신작 신호 — 발행일 경로를 못 찾아 보류
