# AOD Steam 추천 시스템 — 현재 아키텍처

> 이 문서는 **지금 코드에 실제로 구현되어 있는 것**을 기술한다.
> 목표 아키텍처(pgvector + Spring 서빙 + fun_tag 크로스도메인)는
> [`docs/superpowers/specs/2026-06-28-aod-recommendation-system-design.md`](../../docs/superpowers/specs/2026-06-28-aod-recommendation-system-design.md) 참고.
> 둘 사이의 갭은 아래 [§8 피드백이 필요한 부분](#8-피드백이-필요한-부분)에 정리했다.

관련 스펙: [P1 multi-seed 설계](../../docs/superpowers/specs/2026-07-23-steam-p1-multiseed-personalization-design.md)

---

## 1. 한눈에 보기

현재 시스템은 **오프라인 배치 실험 파이프라인**이다. 온라인 서빙 API는 아직 없다.
하나의 임베딩 자산(19,476 × 1024) 위에 세 개의 실험 트랙이 얹혀 있다.

| 트랙 | 질문 | 상태 |
|---|---|---|
| **S1** | 게임 A를 좋아하는 사람에게 무엇을 보여줄까? (item-to-item) | 베이스라인 확정 (R1 rec 3% + R2 meta 2%) |
| **P1** | 좋아하는 게임 3~5개를 주면? (multi-seed 개인화) | MAX 동결(불안정 경고 첨부), TOP2_MEAN 챌린저 보존 |
| **T1** | 신작/뜨는 게임을 밀어줄 수 있나? | **DROP 권고** — 신호가 사실상 0 (19,476개 중 1개만 non-zero) |

---

## 2. 전체 아키텍처

```
┌─ 오프라인 (한 번만 / 코퍼스 변경 시) ─────────────────────────────────────┐
│                                                                            │
│  steam_games.jsonl                                                         │
│  (외부 백엔드 리포, 절대경로 하드코딩 ⚠)                                     │
│        │                                                                   │
│        ▼  data_loader.py                                                   │
│  ┌───────────────────────────────────────────────┐                         │
│  │ 정제 + 필터                                    │                         │
│  │  · type == "game"                              │                         │
│  │  · HTML 태그 제거 / 엔티티 언이스케이프         │                         │
│  │  · short_description ≥ 30자                    │                         │
│  │  · appid 중복 제거 (긴 description 우선)        │                         │
│  │  · steam_ranking.parquet 조인 (steam_rank)     │                         │
│  └───────────────────────────────────────────────┘                         │
│        │  dataset.parquet  (19,476행)                                       │
│        ▼  text_builder.py                                                  │
│  ┌───────────────────────────────────────────────┐                         │
│  │ semantic_text 조립                             │                         │
│  │   "Description: ...                            │                         │
│  │    Genres: 액션, 인디                           │                         │
│  │    Categories: 싱글 플레이어 ...               │                         │
│  │    Steam Top Sellers: #12"   ← 코드는 지원하나  │                         │
│  │                                실제 산출물엔 없음 ⚠│                       │
│  └───────────────────────────────────────────────┘                         │
│        │                                                                   │
│        ▼  embed_qwen.py   (Qwen3-Embedding-0.6B, CPU fp32, ~2.5시간)        │
│  ┌───────────────────────────────────────────────┐                         │
│  │ corpus_embeddings.npy   (19476 × 1024, f32)   │  ← L2 정규화됨           │
│  │ corpus_index.parquet    (row ↔ appid 매핑)     │                         │
│  │ anchor_embeddings.npy   (40 × 1024)           │  ← query instruct 프리픽스│
│  └───────────────────────────────────────────────┘                         │
└────────────────────────────────────────────────────────────────────────────┘
         │                                            │
         │  (S1 트랙)                                  │  (P1 / T1 트랙)
         ▼                                            ▼
┌─ S1: item-to-item ───────────────┐   ┌─ P1: multi-seed 개인화 ──────────────┐
│                                  │   │                                      │
│ anchor_builder.py                │   │ liked_appids [3~5]                   │
│  8장르 × 4인기버킷 층화 → 80후보   │   │      │                               │
│  → (사람이 엑셀에서 40개 선별)     │   │      ▼ seed_loader.py                │
│      │                           │   │  appid → embedding lookup            │
│      ▼ retrieve.py               │   │      │                               │
│  queries @ corpus.T              │   │      ▼ candidate_retriever.py        │
│  (40×1024 @ 1024×19476, exact)   │   │  seed_vecs @ corpus.T                │
│  자기 자신 제외 → Top-100         │   │  → [n_seeds × 19,476] 전량 유사도     │
│      │                           │   │      │                               │
│      ▼ rank.py                   │   │      ▼ score_aggregator.py           │
│  R1: × (1 + rec_pct × 0.03)      │   │  MAX / MEAN / TOP2_MEAN              │
│  R2: + meta_signal × 0.02        │   │  (+ MAX는 dominant_seed 기록)         │
│      │                           │   │      │                               │
│      ▼                           │   │      ▼ personalized_ranker.py        │
│  ranked_top100_*.parquet         │   │  × (1 + rec_pct × 0.03)              │
│                                  │   │  seed 제외 → Top-N                    │
│                                  │   │      │                               │
│                                  │   │      ├─▶ (T1) trend_ranker.py        │
│                                  │   │      │   + trend_signal × w  ← 사실상 │
│                                  │   │      │                        무효 ⚠  │
│                                  │   │      ▼                               │
│                                  │   │  ranked_{strategy}.parquet           │
└──────────────────────────────────┘   └──────────────────────────────────────┘
         │                                            │
         └──────────────────┬─────────────────────────┘
                            ▼  export_*.py
              ┌──────────────────────────────────┐
              │ 블라인드 평가 엑셀                 │
              │  (전략 정보 제거, mapping 별도 저장)│
              └──────────────────────────────────┘
                            │  ← 판정자: LLM proxy (GPT-5.6 Thinking)
                            ▼     relevance 0~3 / confidence
              ┌──────────────────────────────────┐
              │ evaluate.py / eval_ranking.py /  │
              │ eval_personalization.py          │
              │  NDCG@10 · P@10 · Conf@10        │
              │  seed dominance · error 분포      │
              └──────────────────────────────────┘
```

---

## 3. 스코어링 공식

모든 임베딩이 L2 정규화되어 있으므로 **내적 = 코사인 유사도**다.

**S1 — R1 (인기도 부스트)**
```
final_score = similarity × (1 + rec_percentile × 0.03)
rec_percentile = recommendations_total 의 코퍼스 백분위 (결측은 0.0)
```

**S1 — R2 (+ 메타크리틱)**
```
final_score = similarity × (1 + rec_percentile × 0.03 + meta_signal × 0.02)
meta_signal = clip( (metacritic 백분위 − 0.5) × 2, 0, 1 )   # 중앙값 미만은 0
```

**P1 — multi-seed 집계** (`sim_matrix`: `[n_seeds × 19,476]`)
```
MAX        : score(c) = max_i  sim(c, s_i)          ← 현재 동결된 베이스라인
MEAN       : score(c) = mean_i sim(c, s_i)
TOP2_MEAN  : score(c) = mean( top2_i sim(c, s_i) )  ← 챌린저
final_score = score(c) × (1 + rec_percentile × 0.03),  seed 제외
```

**T1 — 트렌드 (권고: DROP)**
```
trend_excess = max(0, 연령코호트_rec_백분위 − 전역_rec_백분위)
trend_signal = trend_excess × freshness_weight   # 1.0 / 0.75 / 0.35 / 0.0
final_score  = score × (1 + rec_pct × 0.03 + trend_signal × w)
```

---

## 4. 아티팩트 계약

| 경로 | 내용 | 생산자 |
|---|---|---|
| `artifacts/s1_v2/dataset.parquet` | 정제 코퍼스 19,476행 + `semantic_text` | `data_loader` → `text_builder` |
| `artifacts/s1_v2/corpus_embeddings.npy` | 19476 × 1024 float32 (정규화) | `embed_qwen` |
| `artifacts/s1_v2/corpus_index.parquet` | `embedding_row ↔ steam_appid ↔ name` | `embed_qwen` |
| `artifacts/s1_v2/anchors_40.parquet` | 사람이 선별한 40개 앵커 | `anchor_builder finalize` |
| `artifacts/s1_v2/qwen_top100.parquet` | 앵커별 Top-100 (유사도만) | `retrieve` |
| `artifacts/s1_v2/ranked_top100_r*.parquet` | 부스트 적용 재랭킹 변형들 | `rank` |
| `artifacts/p1/profiles.parquet` | 합성 프로필 20개 + Dev/Val split | `build_p1_profiles` |
| `artifacts/p1_v2/p1_eval_*_blind.xlsx` + `*_mapping.parquet` | 블라인드 평가지 / 정답 매핑 분리 | `export_personalization_eval` |
| `artifacts/p1_review/p1_llm_proxy_evaluation_summary.json` | P1 최종 지표 + 동결 결정 | 수기 |
| `artifacts/trend_v1/trend_features.parquet` | 연령 코호트 / trend_signal | `trend.trend_features` |

**핵심 결합점**: `corpus_index.parquet`의 `embedding_row`가 `.npy`의 행 인덱스와 1:1이어야 한다.
`SeedLoader` · `CandidateRetriever` · `UserProfileBuilder`가 모두 이 가정에 의존한다.

---

## 5. 평가 하네스

- **판정자**: LLM proxy (GPT-5.6 Thinking). 사람 판정은 아직 0건 — S1 30앵커 검증(`validation_judgments.csv`, 369행)도 동일 방식.
- **척도**: `relevance` 0~3 (2 이상 = positive), `recommendation_confidence` 0~3.
- **블라인드**: P1은 전략 정보를 엑셀에서 제거하고 `*_mapping.parquet`으로 분리해 판정 후 조인한다.
- **프로토콜**: Dev(12 프로필)로 전략 동결 → Val(8 프로필)은 확인용, **재튜닝 금지**.

### P1 결과 (`p1_llm_proxy_evaluation_summary.json`)

| | MAX | MEAN | TOP2_MEAN |
|---|---|---|---|
| Dev NDCG@10 | **0.819** | 0.779 | 0.773 |
| Dev P@10 | **0.808** | 0.783 | 0.783 |
| Val NDCG@10 | 0.813 | 0.851 | **0.871** |
| Val P@10 | 0.800 | **0.900** | 0.888 |

> **Dev와 Val의 승자가 뒤집힌다.** 20개 프로필 규모로는 집계 전략의 우열을 판정할 수 없다는 증거로 기록되어 있으며, MAX를 "단순한 배포 가능 베이스라인"으로 동결하고 TOP2_MEAN을 챌린저로 보존한 상태다.

### T1 결과

`trend_signal > 0` 인 게임이 **19,476개 중 1개**. 10개 프로필 × 3개 가중치(0.5%/1%/2%) 전부에서 Top-10 겹침이 10/10.
원인은 코호트 백분위와 전역 백분위를 **같은 단일 스냅샷 지표**(`recommendations_total`)에서 뽑아 거의 완전 상관이라는 점 + 0\_90d(10개)·91\_365d(23개) 코호트가 최소 크기 50 미달.

---

## 6. 실행 순서

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### S1 (모듈 실행 — `src.` 프리픽스)
```bash
python -m src.data_loader
python -m src.text_builder
python -m src.anchor_builder generate      # → anchor_candidates_80.xlsx
#  (사람) include_YN 을 정확히 40개 Y 로
python -m src.anchor_builder finalize
python -m src.tfidf_baseline               # 대조군
python -m src.embed_qwen                   # CPU 기준 ~2.5시간
python -m src.retrieve
python -m src.rank --r1-boost 0.03
python -m src.rank --meta-boost 0.02
python -m src.export_evaluation [--pilot]
#  (사람/LLM) evaluation.xlsx Judgments 채점
python -m src.validate_evaluation fill  artifacts/s1_v2/evaluation.xlsx
python -m src.validate_evaluation check artifacts/s1_v2/evaluation.xlsx
python -m src.evaluate
python -m src.report
```

### P1 / T1 (스크립트 실행 — `src/`를 sys.path에 넣고)
```bash
python src/build_p1_profiles.py
python src/personalized_retrieve.py --liked-appids 730 578080 359550 --top-n 20
python src/export_personalization_eval.py
python src/eval_personalization.py
python src/trend/trend_features.py         # T1 피처
python src/eval_trend.py                   # T1 diff 분석
```

> ⚠️ 두 계열은 **import 규약이 다르다**(`from src.config` vs `from config`). 같은 프로세스에서 함께 import할 수 없다. 아래 §8-A2 참조.

### 테스트
```bash
pytest tests/          # 14개 파일
```

---

## 7. 파일 지도

```
src/
  config.py                     설정/아티팩트 경로 (항상 configs/s1_v2.yaml 로드)
  data_loader.py                jsonl → 정제 dataset.parquet
  text_builder.py               semantic_text 조립
  anchor_builder.py             층화 앵커 후보 생성 / 사람 선별 결과 확정
  embed_qwen.py                 Qwen3 임베딩 + 런타임 벤치마크
  tfidf_baseline.py             TF-IDF 대조군
  retrieve.py                   앵커 × 코퍼스 exact 내적 → Top-100
  rank.py                       R1(인기) / R2(메타크리틱) 부스트 재랭킹
  personalized_retrieve.py      P1 CLI 진입점
  personalization/
    seed_loader.py              appid → 임베딩
    candidate_retriever.py      전량 유사도 행렬
    score_aggregator.py         MAX / MEAN / TOP2_MEAN
    personalized_ranker.py      rec 부스트 + seed 제외 + Top-N
  trend/
    trend_features.py           출시일 파싱 → 연령 코호트 → trend_signal
    trend_ranker.py             PersonalizedRanker + 트렌드 항
  user_profile.py               (미사용) playtime 로그가중 단일벡터 프로파일
  steam_user_data.py            (미연결) Steam Web API 소유게임/플레이타임 조회
  build_p1_profiles.py          합성 프로필 20개 + Dev/Val split 검증
  export_*.py / eval_* / evaluate.py / report.py / validate_evaluation.py
```

---

## 8. 피드백이 필요한 부분

우선순위 순. **[Bug]** = 지금 결과를 왜곡하고 있는 것, **[Decision]** = 사람이 정해야 하는 것.

### A. 재현성 · 코드 위생

| # | 항목 | 상세 |
|---|---|---|
| A1 | **[Bug] 외부 절대경로 하드코딩** | `configs/s1_*.yaml`의 `input_path`, `trend_features.py:11`의 `RAW_DATA`, `steam_user_data.py:5`의 API 키 경로가 모두 `/home/jiho/projects/...`. 원본 `steam_games.jsonl`이 이 리포에 없어 **다른 머신에서 파이프라인 재현 불가**. 아티팩트만으로 P1/T1은 돌지만 S1 앞단은 못 돈다. |
| A2 | **[Bug] import 규약 이원화** | S1 계열은 `from src.config`(모듈 실행), P1/T1 계열은 `from config`(sys.path 주입). `eval_personalization.py`는 `sys.path.insert` + 함수 안에서 또 `sys.path.insert("src")`를 한다. 하나로 통일 필요(패키지화 권장). |
| A3 | **[Bug] 설정이 코드를 지배하지 않음 — 랭킹 신호가 실제로 빠져 있다** | `config.py`는 인자 없으면 **항상 `s1_v2.yaml`**을 읽고, `s1_v3.yaml`(= `ranking_path` 추가본)은 어디서도 안 쓰인다. 확인 결과 **`dataset.parquet`에 `steam_rank` 컬럼이 아예 없고, `semantic_text`에 `Steam Top Sellers` 문자열이 0건**이다(`Categories:`는 존재). 즉 `data/steam_ranking.parquet`을 커밋해두고 `text_builder`가 지원까지 하는데도 **Top Sellers 순위는 임베딩에 한 번도 들어간 적이 없다.** `p1.yaml`의 `top_k_per_seed`·`n_coherent`·`split`·`paths`도 코드가 안 읽는다(프로필은 `build_p1_profiles.py`에 하드코딩). |
| A4 | **[Decision] 프로필 정의가 두 벌** | `build_p1_profiles.py`의 20개(평가용)와 `eval_trend.py:12`의 10개(트렌드 diff용)가 별개로 하드코딩. 단일 소스로 합칠지. |
| A5 | **[Bug] 죽은 코드** | `rank.py:126-127` — `dataset_path`를 계산한 직후 덮어씀. `_run_all_debug()`의 부스트 공식(log10/6, 0.08/0.04, 1/log2)은 R1/R2와 완전히 다른 미채택 실험이 남아있음. |

### B. 평가 신뢰도 — **여기가 제일 급하다**

| # | 항목 | 상세 |
|---|---|---|
| B1 | **[Bug] leave-one-out 평가가 실제로 아무것도 측정하지 않음** | `eval_personalization.py:105` `evaluate_synthetic_leave_one_out()`은 seed를 하나 빼고 그것을 맞히는지 보지 않는다. `recall = len(ranked.head(k)) / len(liked)` — 즉 **k와 seed 개수만으로 결정되는 상수**(k=10, seed=3 → recall 3.33). 게다가 랭커가 `exclude_appids=set(liked)`로 seed를 이미 제거하므로 구조적으로 hold-out을 맞힐 수 없다. **이 지표로 판단한 결론이 있다면 전부 무효.** |
| B2 | **[Bug] NDCG 정의 불일치** | `evaluate.py:12`는 linear gain(`r / log2`), `eval_personalization.py:18`·`eval_ranking.py`는 exponential gain(`(2^r − 1) / log2`). `configs/s1_v2.yaml`은 `ndcg_gain: "linear"`로 선언. **S1 숫자와 P1 숫자를 나란히 비교할 수 없다.** |
| B3 | **[Bug] 품질 게이트가 조용히 죽어있음** | `evaluate.py:70`이 `steam_s1_tfidf_v1` / `steam_s1_qwen_v1`을 찾는데 실제 실험 ID는 `..._v2`(`embed_qwen.py:16`). 따라서 `comparison`·`quality_gate_passed(P@10 ≥ 0.70)`가 **한 번도 산출된 적 없다.** |
| B4 | **[Decision] 미판정 후보 처리 규칙 상충** | `eval_ranking.py`는 판정 없는 후보를 relevance 0으로 채우고, `eval_personalization.py`는 `notna()`로 제외한다. pooled 평가에서 전자는 커버리지 낮은 변형에 불리, 후자는 유리. 규칙을 하나로 정해야 변형 비교가 성립. |
| B5 | **[Decision] 정답이 LLM proxy 단독** | 사람 판정 0건. summary.json 스스로 "opaque ID를 썼어야 했다(profile_id 문자열이 의미 노출)"고 기록. **최소 규모라도 사람 블라인드 판정 세트**가 필요하다 — 특히 Dev/Val 역전(§5) 때문에. |
| B6 | **[Decision] 표본 크기** | 20 프로필 × Top-10 = 판정 388쌍으로 P@10 차이 0.025를 판정하려 했다. 신뢰구간이 전략 간 차이보다 훨씬 넓다. 프로필 수를 늘릴지, 아니면 "집계 전략은 결정 불가"로 못 박고 다른 축(다양성·필터)에 자원을 쓸지. |
| B7 | **[Bug] 평가 함수가 프로필 단위로 집계하지 않음** | `evaluate_from_judgments()`·`compute_pooled_ndcg()` 둘 다 `profile_id` groupby 없이 **split 전체를 한 덩어리로 놓고** `nlargest(10)`을 한다 → 절대 유사도가 높은 한두 프로필이 지표를 독식한다. 반면 `summary.json`의 Dev MAX P@10 `0.8083 = 97/120`은 **프로필별 P@10을 12개 평균**한 값이다. 즉 **공식 P1 숫자는 이 스크립트로 재생성되지 않는다** — 리포 밖(수기/LLM)에서 계산됐다. |

### C. 모델링

| # | 항목 | 상세 |
|---|---|---|
| C1 | **[Decision] 부스트가 유효한가** | 곱셈형 부스트 상한이 3%(R1) + 2%(R2)다. 코사인 유사도 상위권 간 격차가 이보다 크면 순위가 거의 안 바뀐다. `rank.py`가 `rank_delta`·`extreme jumps`를 출력하긴 하는데 **"부스트가 실제로 순위를 바꿨고 그게 좋은 방향이었다"는 결론이 문서화되어 있지 않다.** 부스트를 키울지, 가산형으로 바꿀지, 아예 뺄지. |
| C2 | **[Bug] 인기도 백분위 정의가 두 갈래** | `rank.py:16` — 결측은 명시적으로 `0.0`. `personalized_ranker.py:21` — `fillna(0)` 후 전체 랭크 → 결측 게임들이 하나의 큰 tie 블록으로 **중하위 백분위**를 받는다. S1과 P1이 서로 다른 인기도 척도를 쓰고 있다. |
| C3 | **[Decision] 다양성·필터 후처리 전무** | 설계 스펙 §5.4는 hard filter → soft penalty → diversity → Top-N 순서를 규정하는데 구현에는 **아무것도 없다.** 같은 시리즈 여러 편, 장르 편중, 성인/미출시/데모 제외 등 미적용. `rank.py`가 "4개 이상 앵커에 등장하는 후보"를 세는 saturation 로그는 있지만 대응 로직이 없다. |
| C4 | **[Decision] MAX의 seed 독점** | MAX는 seed 하나가 Top-10을 독식할 수 있다. `seed_dominance_from_rankings()`가 지표를 뽑긴 하는데 **임계값도 결론도 없다.** "한 seed가 Top-10의 N개 초과 시 페널티" 같은 규칙이 필요한지. |
| C5 | **[Decision] T1 처리** | 신호가 사실상 0으로 검증됐다(원인도 규명됨: 단일 스냅샷 누적지표라 코호트/전역 백분위가 거의 완전 상관). 진짜 트렌드에는 `recommendations_total`의 **시계열 스냅샷**이 필요하다. → (a) 스냅샷 수집을 시작할지, (b) `src/trend/` 코드를 제거할지, (c) 실패 기록으로 동결 보존할지. |
| C6 | **[Decision] 텍스트 표현** | `short_description`(짧음) + 장르만 쓴다. 태그·리뷰 텍스트·`detailed_description`은 미사용. 한국어/영어가 섞인 상태로 그대로 임베딩된다(예: 앵커는 한글 설명, 후보는 영문). 다국어 혼재가 유사도를 왜곡하는지 확인 필요. |

### D. 성능 · 서빙 갭

| # | 항목 | 상세 |
|---|---|---|
| D1 | **[Decision] 목표 아키텍처와의 단절** | 설계 스펙/M0·M1·M2 플랜(4,500줄)은 pgvector HNSW + Spring 서빙 + Airflow + fun_tag 크로스도메인을 전제하는데, **현재 코드는 Steam 단일 도메인 · numpy 전량 내적 · 수동 스크립트**다. M0~M2 구현 코드는 이 리포에 없다. 이 실험 자산(`corpus_embeddings.npy` 등)을 `aod_ai.content_embedding`으로 옮기는 계약이 아직 없다. |
| D2 | **[Bug] 랭커가 매 호출마다 19k 재랭킹** | `PersonalizedRanker.rank()`가 호출될 때마다 `recommendations_total.rank(pct=True)`를 전체 코퍼스에 대해 다시 계산하고, `.map(lambda x: pct.get(x))`로 행 단위 조회한다. `TrendRanker`는 여기에 `.loc` 행 단위 조회를 더 얹는다. 오프라인에선 견디지만 **서빙에선 못 쓴다.** 백분위는 사전 계산 후 조인해야 한다. |
| D3 | **[Decision] 실유저 데이터 미연결** | `steam_user_data.py`(소유게임 + playtime)와 `user_profile.py`(playtime 로그가중 단일벡터)가 있으나 **어느 파이프라인에도 안 붙어 있다.** P1이 multi-seed로 가면서 단일벡터 방식이 밀렸다. 유지/삭제 결정 + 실유저 라이브러리로 평가할지. |
| D4 | **[Decision] 피드백 루프 없음** | 임프레션·클릭 로깅 없음 → 온라인 지표 없음 → 오프라인 LLM proxy가 유일한 신호. 서빙 붙일 때 `rec_impression`/`rec_event`를 1일차부터 넣을지. |
| D5 | **[Decision] ANN 미도입** | 현재 전량 exact 내적(19,476개면 충분히 빠름). 코퍼스가 도메인 확장으로 커지면 언제 HNSW로 갈지 기준 필요. |

### E. 문서 정합성

| # | 항목 | 상세 |
|---|---|---|
| E1 | P1 스펙 §3은 임베딩을 **3584차원**이라고 적었으나 실제는 **1024차원**(Qwen3-Embedding-0.6B). |
| E2 | P1 스펙 §2는 "seed별 Top-100 → union"이라 적었으나 구현은 **전량 코퍼스 집계**(P1-Fix-01에서 의도적으로 바꿈). 스펙 갱신 필요. `p1.yaml`의 `top_k_per_seed: 100`도 잔재. |
| E3 | P1 스펙 §3은 `personalized_retrieve.py`에 `single`/`multi` 모드가 있다고 하나 실제 CLI에는 `multi`만 있다. |
| E4 | 이전 README가 가리키던 `2026-07-21-steam-s1-experiment-spec.md`는 **존재하지 않는다.** S1 스펙 문서가 리포에 없다. |

---

## 9. 해결 방안

### 권장 순서

```
1단계 (평가 신뢰 회복)  B3 → B7 → B1 → B2 → B4     ← 이게 끝나야 나머지 판단이 가능
2단계 (재현성 확보)      A1 → A2 → A3 → A5
3단계 (모델링 결정)      C1 → C2 → C4 → C3 → C5 → C6
4단계 (서빙 준비)        D2 → D1 → D4 → D3 → D5
상시                    E1~E4, A4, B5, B6
```

> **왜 B가 먼저인가**: C·D의 모든 항목은 "바꿨더니 좋아졌는가"를 판정해야 결론이 난다. 지금은 그 판정기가 고장나 있어서, C를 먼저 손대면 개선인지 퇴보인지 알 수 없다.

---

### B. 평가 신뢰도

#### B3 — 품질 게이트가 죽어있음 · 난이도 하 · 10분

**원인**: `evaluate.py:70`이 실험 ID를 문자열 리터럴로 비교하는데, `embed_qwen.py:16`이 `_v2`로 올라가면서 어긋났다. `.get()`이 `None`을 반환하고 `if t and q:` 가드가 조용히 통과시킨다.

**해결**: ID를 설정에서 읽고, **못 찾으면 실패**시킨다. 조용한 스킵이 근본 원인이다.

```python
# configs/s1_v2.yaml
evaluation:
  baseline_experiment_id: "steam_s1_tfidf_v2"
  treatment_experiment_id: "steam_s1_qwen_v2"
  precision_gate: 0.70

# src/evaluate.py
base_id = cfg["evaluation"]["baseline_experiment_id"]
treat_id = cfg["evaluation"]["treatment_experiment_id"]
missing = [i for i in (base_id, treat_id) if i not in summaries]
if missing:
    raise SystemExit(f"판정 데이터에 실험 ID 없음: {missing} (있는 것: {list(summaries)})")
```

**검증**: 기존 `evaluation_pilot.xlsx`로 재실행 → `metrics.json`에 `comparison` 블록이 처음으로 생긴다. 여기서 나오는 tfidf 대비 델타가 **"Qwen 임베딩을 쓸 근거"의 첫 실측치**다.

---

#### B7 — 평가 함수가 프로필 단위로 집계하지 않음 · 난이도 하 · 30분

**원인**: pooled 평가에서 프로필별로 랭킹 품질을 재고 **평균**해야 하는데, split 전체를 한 풀로 합쳐 상위 10개를 뽑는다. 유사도 절대값이 높은 프로필(예: FPS처럼 코퍼스에 유사 게임이 빽빽한 장르)이 지표를 독식한다.

**해결**: `profile_id`로 묶고 프로필별 NDCG/P를 계산한 뒤 macro-average. IDCG도 프로필별로 계산해야 한다.

```python
def evaluate_from_judgments(judgments_df, mapping_df, k=10, strategies=(...)):
    joined = judgments_df.merge(mapping_df, on="pair_key", how="inner")
    rows = []
    for strategy in strategies:
        score_key = f"{strategy}_score"
        per_profile = []
        for pid, grp in joined.groupby("profile_id"):          # ← 핵심
            scored = grp[grp[score_key].notna()]
            if scored.empty:
                continue
            topk = scored.nlargest(k, score_key)
            rels = topk["relevance"].astype(float).tolist()
            idcg_pool = sorted(grp["relevance"].astype(float), reverse=True)[:k]  # 프로필별 IDCG
            per_profile.append({
                "ndcg": dcg(rels) / dcg(idcg_pool) if dcg(idcg_pool) else 0.0,
                "p": float((np.array(rels) >= 2).mean()),
                "conf": float(np.mean(rels)),
                "n": len(rels),
            })
        rows.append({
            "strategy": strategy.upper(), "k": k, "profiles": len(per_profile),
            "NDCG": round(np.mean([x["ndcg"] for x in per_profile]), 4),
            f"P@{k}": round(np.mean([x["p"] for x in per_profile]), 4),
            f"Conf@{k}": round(np.mean([x["conf"] for x in per_profile]), 4),
        })
    return pd.DataFrame(rows)
```

**검증 (중요)**: 고쳐서 돌린 값이 `summary.json`의 Dev MAX `P@10 0.8083 / NDCG 0.8188`을 **재현하는지** 확인한다.
- 재현되면 → 공식 숫자가 검증됐고, 이후 실험을 코드로 돌릴 수 있다.
- 재현 안 되면 → **동결된 MAX 결정의 근거가 사라진 것**이므로 P1 결론을 다시 내려야 한다.

한 가지 주의: 판정 파일의 `relevance`는 0~3인데 `evaluate.py`는 linear gain, `eval_personalization.py`는 exponential gain을 쓴다. B2를 같이 처리해야 재현 비교가 성립한다.

---

#### B1 — leave-one-out이 아무것도 측정하지 않음 · 난이도 중 · 반나절

**원인**: 이름만 leave-one-out이고, 실제로는 `recall = len(ranked.head(k)) / len(liked)` — 결과 개수를 seed 개수로 나눈 상수다. 게다가 랭커가 `exclude_appids=set(liked)`로 seed를 미리 제거하므로 **hold-out을 맞히는 것 자체가 구조적으로 불가능**하다.

**해결**: seed를 하나 빼고 나머지로 추천한 뒤, **빠진 seed가 몇 등에 나오는지** 본다. 제외 목록에서도 hold-out을 빼야 한다.

```python
def evaluate_leave_one_out(profiles_df, ks=(10, 50, 100, 300), **rank_kw):
    rows = []
    for _, profile in profiles_df.iterrows():
        liked = list(profile["liked_appids"])
        if len(liked) < 2:
            continue
        for held_out in liked:
            remaining = [a for a in liked if a != held_out]
            results = run_multi(liked_appids=remaining, top_n=max(ks), **rank_kw)
            for strategy, ranked in results.items():
                # exclude 는 remaining 만 — held_out 은 후보로 남아야 한다
                hit = ranked.index[ranked["steam_appid"] == held_out]
                rank = int(ranked.loc[hit[0], "rank"]) if len(hit) else None
                rows.append({
                    "profile_id": profile["profile_id"], "strategy": strategy.upper(),
                    "held_out": held_out, "rank": rank,
                    "rr": 1.0 / rank if rank else 0.0,
                    **{f"hit@{k}": int(rank is not None and rank <= k) for k in ks},
                })
    return pd.DataFrame(rows)
```

**지표**: `HitRate@k`(빠진 seed가 상위 k에 들어온 비율), `MRR`(1/순위 평균).
**해석 주의**: LOO는 "사용자가 이미 좋아한다고 밝힌 게임"을 맞히는 과제라 **참신성을 전혀 측정하지 않는다.** 판정 없이 자동으로 돌릴 수 있는 **보조 회귀 지표**로만 쓰고, 전략 선택의 주 근거는 판정 기반 NDCG로 유지한다. 이 구분을 코드 주석과 출력에 명시할 것.

**부수 효과**: 20 프로필 × 3 seed = **60개의 무판정 평가 포인트**가 생긴다. LLM 판정을 새로 받지 않고도 C1(부스트 크기)·C4(seed 독점) 같은 파라미터 변경의 회귀를 감지할 수 있다.

---

#### B2 — NDCG 정의 불일치 · 난이도 하 · 1시간

**원인**: gain 함수가 3개 파일에 각각 하드코딩됐다. `evaluate.py`는 `r`, 나머지 둘은 `2^r − 1`. relevance 3점 척도에서 exponential은 3점을 7, linear는 3으로 계산 — **상위 1개의 만점 판정이 지표를 지배하는 정도가 완전히 다르다.**

**해결**: 공용 모듈 하나로 뽑고 설정에서 gain을 선택한다.

```python
# src/metrics.py  (신규)
def gain(r: float, mode: str) -> float:
    return float(r) if mode == "linear" else (2.0 ** r - 1.0)

def dcg(rels, mode): return sum(gain(r, mode) / math.log2(i + 2) for i, r in enumerate(rels))
def ndcg_at_k(rels, ideal, k, mode):
    idcg = dcg(sorted(ideal, reverse=True)[:k], mode)
    return dcg(rels[:k], mode) / idcg if idcg else 0.0
```

`evaluate.py` / `eval_ranking.py` / `eval_personalization.py`의 자체 구현을 전부 삭제하고 이걸 import한다. gain 모드는 `configs/*.yaml`의 `evaluation.ndcg_gain` 하나만 본다.

**결정 필요**: linear vs exponential 중 하나. **exponential(`2^r − 1`) 권장** — 3점 척도에서 "매우 타당(3)"과 "보통(2)"의 차이를 벌려주는 편이 추천 품질 판단에 맞고, IR 관행이기도 하다. 다만 이 경우 `s1_v2.yaml`의 `ndcg_gain: "linear"`를 바꿔야 하고, **기존 S1 숫자는 전부 재계산**해야 한다(판정 데이터는 그대로 있으니 재판정은 불필요).

---

#### B4 — 미판정 후보 처리 규칙 상충 · 난이도 하~중 · 2시간

**원인**: 두 평가 경로가 서로 다른 세계 가정을 쓴다.
- `eval_ranking.py`: 판정 없으면 `relevance = 0` (닫힌 세계)
- `eval_personalization.py`: `notna()`로 제외 (열린 세계)

S1은 4개 변형의 Top-10을 pooling했는데 변형별 커버리지가 다르면 0-채우기가 커버리지 낮은 변형에 불리하게 작용한다. P1은 3개 전략의 Top-10 union을 **전부** 판정하므로 각 전략의 Top-10이 100% 판정 안에 들어와 문제가 없다.

**해결**: **"pool은 비교 대상 전 변형의 Top-k union이고, pool 안은 100% 판정한다"**를 계약으로 못 박고, 판정 커버리지를 **검사**한다. 그러면 0-채우기냐 제외냐가 애초에 발생하지 않는다.

```python
def assert_pool_coverage(pooled_judgments, variants, k=10):
    for label, df in variants.items():
        topk = df[df["rank"] <= k][["anchor_steam_appid", "candidate_steam_appid"]]
        judged = set(map(tuple, pooled_judgments[["anchor_steam_appid", "candidate_steam_appid"]].values))
        missing = [t for t in map(tuple, topk.values) if t not in judged]
        if missing:
            raise ValueError(f"{label}: Top-{k} 중 {len(missing)}쌍 미판정 — pool 재생성 필요")
```

`eval_ranking.py`의 `gains.append(0)` 폴백을 이 검사로 교체한다. 새 변형을 추가하면 즉시 "pool 재생성 필요"로 실패하는 게 정상 동작이다 — 조용히 0점을 주는 것보다 낫다.

---

#### B5 / B6 — 판정자 신뢰도와 표본 크기 · 결정 필요

이 둘은 코드로 못 고친다. 셋 중 하나를 골라야 한다.

| 선택지 | 비용 | 얻는 것 |
|---|---|---|
| **(a) 소규모 사람 검증** — 20 프로필 중 5개(약 100쌍)를 사람이 블라인드 판정, LLM 판정과 상관계수(Spearman/Cohen's κ) 측정 | 1~2시간 | **LLM proxy를 계속 쓸 근거.** κ ≥ 0.6이면 이후 LLM 판정을 신뢰하고 자유롭게 확장 가능. 가장 가성비 높은 선택 |
| **(b) 프로필 확대** — 20 → 60개, LLM 판정 그대로 | LLM 호출 비용 | P@10 차이 0.025를 판정할 검정력. 단 **LLM이 틀렸다면 정밀하게 틀린 값**을 얻는다 |
| **(c) 집계 전략 결정 포기** — MAX 동결 유지, C3(다양성·필터)로 자원 이동 | 0 | MAX/TOP2 차이(NDCG 0.05 미만)보다 **다양성 후처리 부재(C3)의 체감 영향이 훨씬 클 가능성이 높다** |

**권장: (a) → (c) → 나중에 (b).** (a)로 판정기를 먼저 검증하고, 판정기가 믿을 만해지면 (c)의 개선을 그 판정기로 측정한다. 집계 전략 재대결(b)은 마지막이다 — Dev/Val 역전이 말해주는 건 "MAX가 틀렸다"가 아니라 "이 축의 차이가 노이즈보다 작다"이므로, 여기 더 투자해도 얻는 게 적다.

또한 `summary.json`이 스스로 지적한 **`profile_id` 의미 노출**(`coh_fps` 같은 문자열이 판정자에게 힌트를 준다)은 export 시 `p01`~`p20` 같은 불투명 ID로 치환하고 매핑을 별도 저장하면 된다 — `export_personalization_eval.py`의 blind/mapping 분리 패턴을 `profile_id`에도 적용하는 것뿐이다.

---

### A. 재현성 · 코드 위생

#### A1 — 외부 절대경로 · 난이도 하 · 1시간

**해결**: 경로를 환경변수 + `.env.example`로 외부화한다.

```python
# src/config.py
import os
def resolve_path(p: str) -> Path:
    return Path(os.path.expandvars(p)).expanduser()

# configs/s1_v2.yaml
data:
  input_path: "${AOD_BACK_ROOT}/steam_games.jsonl"
  ranking_path: "${AOD_STEAM_ROOT}/data/steam_ranking.parquet"
```

`trend_features.py:11`의 `RAW_DATA`, `steam_user_data.py:5`의 키 경로도 동일 처리. `.env.example`에 변수 목록과 설명을 남기고, 없으면 **명확한 에러 메시지로 실패**시킨다(`FileNotFoundError`에 "AOD_BACK_ROOT 를 설정하세요" 포함).

> **원본 데이터 자체**는 별개 문제다. `steam_games.jsonl`이 없으면 §6의 S1 앞단(data_loader~embed_qwen)은 못 돈다. 코퍼스 스냅샷의 **체크섬과 획득 방법**을 `data/README.md`에 기록해두면, 재현이 필요할 때 "무엇을 구해야 하는지"라도 명확해진다.

#### A2 — import 규약 이원화 · 난이도 중 · 2시간

**해결**: `src/`를 정식 패키지로 만들고 전부 `src.` 절대 import로 통일한다.

1. `pyproject.toml`에 `[tool.setuptools] packages = ["src", "src.personalization", "src.trend"]` 추가 후 `pip install -e .`
2. `from config import ...` → `from src.config import ...` (P1/T1 계열 6개 파일)
3. `sys.path.insert(...)` 전부 삭제 (`eval_personalization.py`, `export_personalization_eval.py`, `build_p1_profiles.py`)
4. 실행을 `python -m src.X`로 통일 → §6의 두 갈래가 하나가 된다

**검증**: `python -c "import src.evaluate, src.eval_personalization, src.trend.trend_ranker"` 가 한 프로세스에서 성공하면 끝. 지금은 실패한다.

#### A3 — 설정이 코드를 지배하지 않음 · 난이도 중

**두 부분으로 나뉜다.**

**(1) 설정 로딩** — `load_config()`가 기본값으로 `s1_v2.yaml`을 하드코딩한다. `AOD_CONFIG` 환경변수 또는 CLI `--config`를 **모든 진입점**에 노출하고, 산출물에 사용된 config를 함께 기록한다(`qwen_run_config.json`처럼).

```python
def load_config(path=None) -> dict:
    cfg_path = Path(path or os.environ.get("AOD_CONFIG") or PROJECT_ROOT / "configs" / "s1_v2.yaml")
    ...
```

**(2) Top Sellers 신호 — 진짜 문제** — `dataset.parquet`에 `steam_rank`가 없고 `semantic_text`에 `Steam Top Sellers`가 0건이다. 즉 이 신호는 **한 번도 실험된 적이 없다.** 결정이 필요하다:

- **(i) 넣고 재임베딩** — CPU 기준 **~2.5시간**(`runtime_benchmark.json`: 2.15 items/s × 19,476). 전체 코퍼스 재임베딩 + S1/P1 결과 전부 재생성 + 재판정까지 필요하다.
- **(ii) 안 넣기** — `s1_v3.yaml`과 `text_builder.py`의 `steam_rank` 분기를 삭제해서 "지원하는 척"을 없앤다.

**권장: (ii).** 판매 순위는 **인기도 신호이지 의미 신호가 아니다.** 이미 R1이 `recommendations_percentile`로 인기도를 랭킹 단계에서 다루고 있으므로, 같은 정보를 임베딩 텍스트에 섞으면 의미 공간이 오염되고(인기 게임끼리 가까워짐) 부스트와 이중 계산된다. 인기도를 더 강하게 반영하고 싶다면 C1(부스트 크기 조정)이 올바른 손잡이다. — 다만 (i)을 택한다면 A/B가 명확하므로 실험 가치는 있다. **결정만 내려주면 어느 쪽이든 정리 가능하다.**

#### A4 — 프로필 정의 두 벌 · 난이도 하 · 30분

`eval_trend.py:12`의 `PROFILES` 10개를 삭제하고 `artifacts/p1/profiles.parquet`을 읽게 한다. 트렌드 diff는 Dev 12개로 충분하다. 프로필 정의는 `build_p1_profiles.py` 단일 소스 유지.

#### A5 — 죽은 코드 · 난이도 하 · 20분

- `rank.py:126-127`: 첫 줄 삭제(다음 줄이 즉시 덮어씀).
- `rank.py:_run_all_debug()`: R1/R2와 완전히 다른 부스트 공식(log10/6, 0.08/0.04, 1/log2)의 미채택 실험. **삭제 권장** — 남겨두려면 함수 docstring에 "미채택, R1/R2와 공식 다름"을 명시. 지금은 `--r-all-debug` 플래그가 살아있어 실수로 쓰기 쉽다.

---

### C. 모델링

#### C1 — 부스트가 유효한가 · 난이도 하(측정) · 2시간

**원인**: 부스트가 곱셈형이고 상한이 3%+2%다. 코사인 유사도 상위권 간 격차가 이보다 크면 순위가 거의 안 바뀐다. `rank.py`가 `rank_delta`·`extreme jumps`·`saturation`을 출력하지만 **결론이 문서화되지 않았다.**

**해결**: 결정을 위한 근거를 먼저 만든다 — 코드 변경 없이 측정만.

1. **Top-10 변화량 측정**: `qwen_top100.parquet`(부스트 전)과 `ranked_top100_r1_final.parquet`의 앵커별 Top-10 자카드. **자카드 ≈ 1.0이면 R1은 사실상 no-op**이고, 그렇다면 지금까지의 "R1 baseline"이라는 표현 자체가 과장이다. (T1이 정확히 이 방식으로 무효 판정을 받았다 — 같은 검사를 R1/R2에도 적용해야 한다.)
2. **유사도 격차와 비교**: 앵커별 `sim[rank1] − sim[rank10]`의 중앙값을 구해 부스트 상한 3%와 비교. 격차 ≫ 부스트면 구조적으로 무효다.

**결과에 따른 처방**:
- 무효로 나오면 → **가산형으로 전환**: `final = sim + w · rec_pct` (`w`를 유사도 격차의 중앙값 스케일로 잡으면 부스트가 의미를 갖는다). 곱셈형은 `sim`이 클수록 부스트도 커져서 이미 강한 후보를 더 밀어주는, 의도와 반대인 성질도 있다.
- 유효로 나오면 → 지금 값을 문서화하고 넘어간다.

**검증**: B1의 LOO HitRate로 회귀 감지 + 변경된 Top-10만 재판정.

#### C2 — 인기도 백분위 정의 두 갈래 · 난이도 하 · 1시간

**원인**: `rank.py:16`은 결측을 명시적으로 `0.0`, `personalized_ranker.py:21`은 `fillna(0)` 후 전체 랭크 → 결측 게임들이 하나의 거대한 tie 블록을 이루고 `method="average"` 때문에 **0이 아니라 중하위 백분위**를 받는다. S1과 P1이 서로 다른 인기도 척도를 쓰고 있다.

**해결**: 정의를 하나로 뽑고 **사전 계산해서 parquet으로 저장**한다(D2도 같이 해결된다).

```python
# src/features/popularity.py  (신규)
def build_popularity_percentile(dataset: pd.DataFrame) -> pd.DataFrame:
    """결측(리뷰 수 미상)은 0.0 — '인기 없음'이 아니라 '부스트 없음'을 의미."""
    out = dataset[["steam_appid"]].copy()
    known = dataset["recommendations_total"].notna()
    out["recommendations_percentile"] = 0.0
    out.loc[known, "recommendations_percentile"] = (
        dataset.loc[known, "recommendations_total"].rank(method="average", pct=True)
    )
    return out
```

`rank.py`의 `build_corpus_percentile`(이쪽이 옳다)을 이 모듈로 옮기고, `PersonalizedRanker`·`TrendRanker`가 **저장된 parquet을 조인**해서 쓰게 한다.

**주의**: 이 수정은 P1 점수를 바꾸므로 **동결된 MAX 베이스라인 숫자가 달라진다.** B7 재현 검증을 끝낸 뒤에 적용하고, 변경 전후를 함께 기록할 것.

#### C3 — 다양성·필터 후처리 전무 · 난이도 중 · 1~2일 · **체감 효과 가장 클 가능성**

**원인**: 설계 스펙 §5.4가 `hard filter → soft penalty → diversity → Top-N` 순서를 규정하는데 구현에 아무것도 없다. `rank.py`가 "4개 이상 앵커의 Top-10에 등장하는 후보"를 세는 saturation 로그는 있으나 대응 로직이 없다.

**해결**: 스펙 순서 그대로 후처리 단계를 신설한다. 랭커 **뒤**에 붙는 독립 모듈이라 S1/P1 양쪽에 재사용된다.

```python
# src/postprocess.py  (신규)
def postprocess(ranked, dataset, top_n=10, *, franchise_max=1, genre_max=5):
    df = apply_hard_filters(ranked, dataset)   # 미출시, 성인 콘텐츠, 데모/사운드트랙
    df = apply_soft_penalty(df, dataset)       # 리뷰 수 극소(<50) 등은 감점, 배제는 아님
    return apply_diversity(df, dataset, top_n, franchise_max, genre_max)
```

- **hard filter**: `data_loader`가 이미 `type == "game"`으로 데모/사운드트랙 상당수를 거른다. 남은 건 미출시(`trend_features.parse_release_date`가 `"출시 예정"` 패턴을 이미 인식한다 — **재사용 가능**)와 성인 콘텐츠(`anchor_builder.py:9`의 `NON_CORE_GENRE_LABELS`에 `신체 노출`·`선정적 콘텐츠`가 이미 정의돼 있다 — **재사용 가능**).
- **diversity**: 프랜차이즈 중복이 가장 눈에 띈다. 이름 접두어 매칭(`"Fallout"`, `"Call of Duty"`)이나 퍼블리셔 기준으로 Top-N당 1편 제한. greedy MMR도 가능하지만 **프랜차이즈 dedup만으로도 체감 개선이 크다** — 먼저 이것만 해볼 것.

**검증**: 후처리 전/후 Top-10을 나란히 놓고 판정(B4의 pool 계약에 따라 union 전부 판정). 이건 **판정할 가치가 확실한 변경**이다 — C1과 달리 결과가 눈에 띄게 달라진다.

#### C4 — MAX의 seed 독점 · 난이도 하 · 3시간

**원인**: MAX는 seed 하나가 Top-10을 독식할 수 있다. `seed_dominance_from_rankings()`가 지표를 뽑지만 임계값도 결론도 없다.

**해결**: 먼저 **측정**한다. 20 프로필에 대해 `dominant_seed` 분포를 뽑고 "Top-10 중 한 seed가 차지한 최대 개수"의 분포를 본다.

- 중앙값이 5 이하 → 문제 없음, 지표만 문서화하고 종료.
- 7 이상이 흔하면 → **round-robin 인터리빙**을 권장한다: seed별 랭킹을 따로 만들고 번갈아 뽑아 각 seed가 최소 1~2개 자리를 갖게 한다. 페널티 방식보다 구현이 단순하고 "내가 고른 게임들이 골고루 반영됐다"는 게 사용자에게 직접 보인다.
- `dominant_seed` 컬럼이 이미 `ScoreAggregator`에 있으므로 데이터는 준비돼 있다.

**참고**: MEAN/TOP2_MEAN은 구조적으로 이 문제가 덜하다. 만약 seed 독점이 심하게 나온다면, **Val에서 TOP2_MEAN이 이겼던 것의 설명**일 수 있다 — B7/B5와 연결해서 볼 것.

#### C5 — T1 처리 · 결정 필요

원인 규명이 이미 끝났다: 코호트 백분위와 전역 백분위를 **같은 단일 스냅샷 누적 지표**에서 뽑아 거의 완전 상관 + 신생 코호트가 최소 크기 미달. 진짜 트렌드에는 `recommendations_total`의 **시계열 스냅샷**이 필요하다.

| 선택지 | 비용 | 판단 |
|---|---|---|
| **(a) 코드 삭제** | 30분 | 가장 깔끔. 커밋 `9e818ba`에 설계·원인·결과가 온전히 남아있어 지식은 보존된다 |
| **(b) 스냅샷 수집 시작** | 일 1회 크론 + 저장소, **최소 4~8주 후에야 신호 발생** | 트렌드가 제품에 정말 필요할 때만. 지금 시작해두면 나중에 데이터가 쌓여있다는 장점은 있다 |
| **(c) 동결 보존** | 0 | 코드가 남아 나중에 "이거 왜 안 쓰지?"를 유발. `src/trend/`에 `DEPRECATED.md` 필수 |

**권장: (a) + 별도로 (b)의 스냅샷 크론만 가볍게 시작.** 수집은 하루 한 번 `recommendations_total`을 덤프하는 것뿐이라 비용이 거의 없고, 데이터가 쌓이는 데 시간이 걸리므로 **일찍 시작할수록 좋다.** 반면 지금의 `trend_features.py` 구현은 스냅샷이 생기면 어차피 다시 써야 한다(코호트 백분위가 아니라 증가율 기반이 될 것).

#### C6 — 텍스트 표현 · 난이도 중 · **먼저 확인부터**

**원인**: `short_description`(짧음) + 장르만 쓴다. 그리고 `validation_judgments.csv`를 보면 **앵커는 한국어 설명, 후보는 영어 설명**이 섞여 있다(예: Counter-Strike 2는 한국어, Brawlhalla는 영어).

**먼저 확인할 것 (반나절, 재임베딩 불필요)**: 언어가 유사도를 왜곡하는지 측정한다.
- 코퍼스의 한/영 비율을 센다(한글 포함 여부로 간단히 분류).
- 앵커별 Top-10에서 **앵커와 같은 언어인 후보 비율**을 본다. 무작위 기대치(코퍼스 언어 비율)보다 유의하게 높으면 → **임베딩이 의미가 아니라 언어로 클러스터링하고 있다.** 이건 Top-10 품질에 직접 타격이고, 지금 P@10 0.8 안에 "같은 언어라서 뽑힌" 후보가 섞여 있다는 뜻이 된다.

**왜곡이 확인되면 해결**:
- (i) 언어를 통일한다 — 영문 `short_description`으로 정규화(Steam API가 `l=english`로 제공). **가장 확실하고, 재임베딩 2.5시간이면 된다.**
- (ii) 다국어 임베딩 모델로 교체 — Qwen3-Embedding은 다국어를 지원하지만 0.6B 모델의 교차언어 정렬은 약할 수 있다.

**표현 확장(태그·리뷰·detailed_description)은 그 다음**이다. 언어 혼재가 있는 상태에서 텍스트를 늘리면 왜곡도 같이 커진다. 순서를 지킬 것.

---

### D. 성능 · 서빙 갭

#### D2 — 랭커가 매 호출마다 19k 재랭킹 · 난이도 하 · 1시간

**원인**: `PersonalizedRanker.rank()`가 호출마다 `recommendations_total.rank(pct=True)`를 전체 코퍼스에 다시 계산하고, `.map(lambda x: pct.get(x))`로 행 단위 조회한다. `TrendRanker`는 `.loc` 행 단위 조회를 더 얹는다.

**해결**: C2의 사전 계산 parquet을 쓰고, 행 단위 `map`/`loc`을 **벡터 조인**으로 바꾼다.

```python
class PersonalizedRanker:
    def __init__(self, rec_boost=0.03, popularity_path=POPULARITY_PARQUET):
        self.pop = pd.read_parquet(popularity_path)   # 사전 계산, 생성자에서 1회
        self.rec_boost = rec_boost

    def rank(self, candidates, exclude_appids=None, top_n=300):
        df = candidates.merge(self.pop, on="steam_appid", how="left")   # 벡터 조인
        df["recommendations_percentile"] = df["recommendations_percentile"].fillna(0.0)
        df["final_score"] = df["seed_similarity"] * (1 + df["recommendations_percentile"] * self.rec_boost)
        if exclude_appids:
            df = df[~df["steam_appid"].isin(exclude_appids)]
        df = df.nlargest(top_n, "final_score").reset_index(drop=True)   # 전체 정렬 대신 nlargest
        df["rank"] = range(1, len(df) + 1)
        return df
```

`sort_values` → `nlargest`도 함께 바꾼다(19k 전체 정렬이 불필요하다). C2와 한 번에 처리하는 게 효율적이다.

#### D1 — 목표 아키텍처와의 단절 · 결정 필요 · **가장 큰 미해결 질문**

설계 스펙과 M0~M2 플랜(4,500줄)은 pgvector HNSW + Spring 서빙 + Airflow + fun_tag 크로스도메인을 전제하는데, 현재 코드는 Steam 단일 도메인 · numpy 전량 내적 · 수동 스크립트다. **M0~M2 구현 코드는 이 리포에 없다.**

**해결의 첫 단계는 코드가 아니라 계약이다.** 이 실험 자산을 서빙으로 넘기려면 세 가지를 정해야 한다:

1. **차원 불일치**: 현재 임베딩은 **1024차원**(Qwen3-0.6B)인데 스펙은 다른 모델/차원을 상정할 수 있다. `aod_ai.content_embedding`의 `vector(n)` 컬럼 정의가 이 값에 묶인다 — **바꾸려면 전량 재임베딩**이므로 지금 확정해야 한다.
2. **ID 매핑**: 실험은 `steam_appid`로, 서빙은 `content_id`(AOD 통합 ID)로 돈다. 변환 테이블이 어디서 오는지 정해지지 않았다.
3. **랭킹 로직의 소유자**: 스펙은 서빙(Java)이 스코어링 대부분을 요청 시 수행한다고 못 박았다. 그러면 `personalized_ranker.py`·`postprocess.py`의 로직은 **Java로 재구현**되고 Python은 asset 생산만 한다. 그 경우 **두 구현의 동치성을 어떻게 보장할지**(같은 입력 → 같은 Top-N 골든 테스트)가 필요하다.

**권장**: 위 3개를 `docs/superpowers/plans/2026-07-03-aod-rec-00-contracts.md`에 추가하고, **가장 얇은 수직 슬라이스** 하나를 먼저 뚫는다 — 임베딩 19,476건을 `aod_ai.content_embedding`에 적재하고 pgvector `<=>` 쿼리로 Top-100을 뽑아 **현재 numpy 결과와 일치하는지** 확인. 이게 되면 M1의 나머지는 반복 작업이고, 안 되면 위 3개 중 뭘 잘못 정했는지가 즉시 드러난다.

#### D3 — 실유저 데이터 미연결 · 결정 필요 · 난이도 하

`steam_user_data.py`(소유게임 + playtime)와 `user_profile.py`(playtime 로그가중 단일벡터)가 어느 파이프라인에도 안 붙어 있다. P1이 multi-seed로 가면서 단일벡터 방식이 밀렸다.

- **`user_profile.py`**: P1 스펙 §1이 "single weighted average는 정보 손실"이라고 명시적으로 기각했다. → **삭제 권장.** (남긴다면 "P1에서 기각됨" 주석 필수)
- **`steam_user_data.py`**: 반대로 **가치가 크다.** 실제 Steam 라이브러리 몇 개만 가져와도 합성 프로필 20개보다 훨씬 현실적인 평가 세트가 된다(장르가 뒤섞이고, 플레이타임이 극단적으로 치우치고, 안 하는 게임이 잔뜩 있는 진짜 분포). B6의 표본 문제를 부분적으로 해결한다.

**권장**: 본인 + 지인 계정 3~5개로 프로필을 만들어 **정성 확인용**으로 먼저 써본다. "내 라이브러리로 추천을 돌렸을 때 납득이 되는가"는 판정 20개보다 빠르게 문제를 드러낸다 — 특히 C3(프랜차이즈 중복)와 C6(언어 클러스터링)이 여기서 바로 눈에 띌 것이다.
개인정보: 실 Steam ID/라이브러리는 커밋하지 말 것(현재 `.gitignore`에 관련 규칙 없음 — `artifacts/user_*` 추가 필요).

#### D4 — 피드백 루프 없음 · 서빙 붙일 때 함께

임프레션·클릭 로깅이 없어 온라인 지표가 없다. 스펙에 `rec_impression`/`rec_event` 테이블과 grant가 이미 설계돼 있으므로(M2 Task 1), **서빙 1일차부터 켜는 것이 정답**이다. 나중에 붙이면 그 전 기간의 데이터가 영구히 없다. 오프라인 LLM proxy가 유일한 신호인 현 상태를 벗어나는 유일한 길이기도 하다.

#### D5 — ANN 미도입 · 지금은 불필요

19,476건 × 1024차원 전량 내적은 numpy로 수십 ms다. **지금 ANN은 과잉 설계다.** 다만 D1의 pgvector 이관 시 HNSW 인덱스는 어차피 따라온다. 판단 기준: 도메인 확장으로 코퍼스가 **10만 건을 넘거나** 서빙 p99가 목표를 못 맞출 때. 그전까지는 exact가 recall 100%라는 장점이 크다(ANN 도입 시 recall 손실을 별도로 측정해야 한다).

---

### E. 문서 정합성 · 난이도 하 · 합계 1시간

| # | 해결 |
|---|---|
| E1 | P1 스펙 §3의 `3584-dim` → `1024-dim` 수정 (실측: `qwen_run_config.json`) |
| E2 | P1 스펙 §2의 "seed별 Top-100 → union"을 **전량 코퍼스 집계**로 수정하고 P1-Fix-01에서 바꾼 이유를 적는다. `p1.yaml`의 `top_k_per_seed: 100` 삭제 |
| E3 | P1 스펙 §3의 `single`/`multi` 모드 언급 삭제 (실제 CLI엔 multi만) |
| E4 | S1 스펙 문서를 새로 쓰거나(권장: 이 README §2~5가 이미 대부분을 담고 있다), 링크를 제거한다 |

**재발 방지**: 스펙과 구현이 갈라지는 지점이 반복적으로 "구현 중 설계를 바꾸고 스펙을 안 고친 곳"이다. P1-Fix-01처럼 **설계 변경이 커밋 메시지에만 남는 패턴**을 피하려면, 변경 시 스펙 문서 수정을 같은 커밋에 포함시키는 게 가장 확실하다.
