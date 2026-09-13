# data/

## 커밋되어 있는 것

| 파일 | 내용 | 생산자 |
|---|---|---|
| `steam_ranking.parquet` | Steam Top Sellers 순위 (`steam_appid`, `steam_rank`, `steam_rank_name`) | `src/fetch_ranking.py` |
| `metacritic.parquet` | appid별 메타크리틱 점수 — R2 부스트 입력 | 외부 수집 |

## 커밋되어 있지 않은 것 — `steam_games.jsonl`

파이프라인의 원본 코퍼스다. 크기 때문에 리포에 넣지 않는다.

- **위치**: 백엔드 리포 루트. `.env` 의 `AOD_BACK_ROOT` 로 경로를 주입한다(`.env.example` 참고).
- **형식**: JSON Lines. Steam Store API `appdetails` 응답을 레코드당 한 줄씩 저장한 것.
- **필수 필드**: `type`, `steam_appid`, `name`, `short_description`, `genres[]`, `categories[]`,
  `metacritic.score`, `recommendations.total`, `release_date`
- **이 원본이 필요한 스크립트**: `src/data_loader.py`, `src/trend/trend_features.py`
  (`release_date` 를 원본에서 직접 읽는다). 나머지는 `artifacts/` 산출물만으로 동작한다.

### 스냅샷 정보

`artifacts/s1_v2/dataset.parquet` 기준:

- 정제 후 코퍼스: **19,476건** (`type == "game"`, `short_description >= 30자`, appid 중복 제거)
- 임베딩: `artifacts/s1_v2/corpus_embeddings.npy` (19,476 × 1024, float32, L2 정규화)
- 모델: `Qwen/Qwen3-Embedding-0.6B` — 상세는 `artifacts/s1_v2/qwen_run_config.json`

원본을 새로 수집하면 코퍼스 크기가 달라져 위 아티팩트와 행이 어긋난다.
그 경우 `data_loader` → `text_builder` → `embed_qwen` 을 다시 돌려야 한다
(CPU 기준 약 2.5시간, `artifacts/s1_v2/runtime_benchmark.json` 참고).
