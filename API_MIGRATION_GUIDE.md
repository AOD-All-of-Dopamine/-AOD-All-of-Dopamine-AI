# API 마이그레이션 가이드

CSV 파일 기반의 `make_bigraph.py`를 API 기반으로 변환했습니다.

## 주요 변경 사항

### 1. `make_bigraph.py` 수정 내용

#### 새로운 함수들:
- `load_json_to_df(json_data)`: JSON을 DataFrame으로 변환
- `set_api_data()`: API POST로 받은 JSON 데이터를 저장
- 수정된 `safe_read()`: CSV 모드와 API 모드를 모두 지원

#### 사용 모드:
```python
USE_API_MODE = False  # False: CSV 사용, True: API JSON 사용
```

### 2. API 서버 (`api_server.py`)

Flask 기반의 REST API 서버입니다.

#### 엔드포인트:

##### GET `/health`
헬스 체크
```bash
curl http://localhost:5000/health
```

##### POST `/build-graph`
그래프 생성 후 CSV로 저장

**요청:**
```json
{
    "contents": [
        {"content_id": 1, "domain": "AV", "master_title": "...", ...},
        ...
    ],
    "av": [
        {"content_id": 1, "genres.tmdb_genres.0.name": "Action", ...},
        ...
    ],
    "game": [
        {"content_id": 2, "genres_str": "Action;Adventure", ...},
        ...
    ],
    "webnovel": [
        {"content_id": 3, "genres": "판타지|로맨스", ...},
        ...
    ],
    "raw_item": [
        {"raw_id": 1, "genres_str": "Action|Adventure", ...},
        ...
    ]
}
```

**응답:**
```json
{
    "success": true,
    "message": "그래프가 성공적으로 생성되었습니다.",
    "stats": {
        "content_nodes_count": 100,
        "meta_nodes_count": 50,
        "edges_count": 200
    }
}
```

##### POST `/build-graph-get-result`
그래프 생성 후 메모리에서 결과 반환 (대용량 데이터에는 부담)

**요청:** `/build-graph`와 동일

**응답:**
```json
{
    "success": true,
    "stats": {
        "content_nodes_count": 100,
        "meta_nodes_count": 50,
        "edges_count": 200
    },
    "nodes": [...],
    "meta_nodes": [...],
    "edges": [...]
}
```

### 3. API 클라이언트 (`api_client.py`)

Python 클라이언트 라이브러리입니다.

#### 사용 방법:

```python
from api_client import call_build_graph_api, call_build_graph_get_result_api
import pandas as pd

# CSV 파일을 데이터로 로드
contents_data = pd.read_csv("contents.csv").to_dict('records')
av_data = pd.read_csv("av_contents.csv").to_dict('records')

# 방법 1: 결과를 CSV로 저장
result = call_build_graph_api(contents_data, av_data=av_data)

# 방법 2: 결과를 메모리에서 조회
result = call_build_graph_get_result_api(contents_data, av_data=av_data)
if result["success"]:
    nodes_df = result["nodes"]
    meta_df = result["meta_nodes"]
    edges_df = result["edges"]
```

## 서버 실행 방법

### 1. 필수 패키지 설치
```bash
pip install flask requests pandas
```

### 2. 서버 시작
```bash
python api_server.py
```

기본 포트: `5000`

### 3. 클라이언트 테스트
```bash
python api_client.py
```

## 데이터 포맷 예시

### contents.csv → JSON
```json
{
    "content_id": 1,
    "domain": "AV",
    "master_title": "Sample Title",
    "original_title": "Original",
    "release_year": 2020,
    "poster_image_url": "http://...",
    "created_at": "2023-01-01",
    "updated_at": "2023-12-01",
    "synopsis": "..."
}
```

### av_contents.csv → JSON
```json
{
    "content_id": 1,
    "genres.tmdb_genres.0.name": "Action",
    "genres.tmdb_genres.1.name": "Crime",
    "..."
}
```

### game_contents.csv → JSON
```json
{
    "content_id": 2,
    "genres_str": "Action;Adventure;Indie",
    "..."
}
```

### webnovel_contents.csv → JSON
```json
{
    "content_id": 3,
    "genres": "판타지|로맨스|액션",
    "..."
}
```

### raw_item.csv → JSON
```json
{
    "raw_id": 1,
    "genres_str": "Action|Adventure|Indie",
    "..."
}
```

## 마이그레이션 전략

### 옵션 1: 로컬 테스트 (CSV 유지)
- `USE_API_MODE = False`로 유지
- 기존 CSV 파일 사용
- API 서버는 선택 사항

### 옵션 2: API 전환
- API 클라이언트에서 JSON 생성
- POST 요청으로 데이터 전송
- 서버에서 처리

### 옵션 3: 하이브리드
- 개발: CSV 모드
- 프로덕션: API 모드
- `USE_API_MODE` 플래그로 전환

## 문제 해결

### 1. JSON 데이터 포맷 오류
- 모든 필수 컬럼이 포함되었는지 확인
- JSON 형식이 올바른지 확인 (Python dict 또는 JSON string)

### 2. 인코딩 문제
- JSON은 UTF-8 사용
- 한글 데이터가 포함된 경우 특별히 주의

### 3. 메모리 부족
- 대용량 데이터는 `/build-graph` 사용 (CSV로 저장)
- `/build-graph-get-result`는 메모리 상에서 처리하므로 주의

## 출력 파일

API 사용 시에도 다음 CSV 파일이 생성됩니다:
- `graph_nodes.csv`: 콘텐츠 노드
- `content_raw_genres.csv`: 원본 장르 데이터 (wide format)
- `meta_nodes.csv`: 메타노드 (도메인, 장르)
- `graph_edges_bipartite.csv`: 이분 그래프 엣지
