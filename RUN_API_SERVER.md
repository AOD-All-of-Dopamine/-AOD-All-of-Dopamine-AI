# API 서버 실행 가이드

## 1. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

또는 직접 설치:
```bash
pip install flask flask-cors flasgger requests pandas
```

## 2. 서버 시작

### Windows (cmd)
```bash
python api_server.py
```

### Mac/Linux
```bash
python3 api_server.py
```

## 3. API 문서 보기

서버 실행 후 **브라우저**에서 다음 URL 접속:

### Swagger UI (권장)
```
http://localhost:5000/apidocs
```

### ReDoc (대안)
```
http://localhost:5000/redoc
```

## 4. 헬스 체크

```bash
curl http://localhost:5000/health
```

응답:
```json
{
  "status": "ok",
  "message": "API is running"
}
```

## 5. API 호출 예시

### Python 클라이언트 사용

```python
from api_client import call_build_graph_api
import pandas as pd

# CSV 데이터 로드
contents_data = pd.read_csv("csv 데이터/clean/contents.csv").to_dict('records')
av_data = pd.read_csv("csv 데이터/clean/av_contents.csv").to_dict('records')

# API 호출
result = call_build_graph_api(contents_data, av_data=av_data)

# 결과
if result.get("success"):
    print("✅ 성공!")
    print(f"노드: {result['stats']['content_nodes_count']}")
    print(f"엣지: {result['stats']['edges_count']}")
```

### curl 사용 (JSON 파일)

1. `payload.json` 생성:
```json
{
  "contents": [
    {"content_id": 1, "domain": "AV", "master_title": "Title1"},
    {"content_id": 2, "domain": "GAME", "master_title": "Title2"}
  ],
  "av": [
    {"content_id": 1, "genres.tmdb_genres.0.name": "Action"}
  ]
}
```

2. 요청 전송:
```bash
curl -X POST http://localhost:5000/build-graph \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Postman 사용

1. Postman 실행
2. **New Request** → **POST**
3. URL: `http://localhost:5000/build-graph`
4. **Body** → **raw** → **JSON** 선택
5. 아래 JSON 입력:

```json
{
  "contents": [
    {"content_id": 1, "domain": "AV", "master_title": "Test Content"}
  ]
}
```

6. **Send** 클릭

## 6. 출력 파일 위치

API 호출 후 다음 파일이 생성됩니다:
- `csv 데이터/clean/graph_nodes.csv`
- `csv 데이터/clean/content_raw_genres.csv`
- `csv 데이터/clean/meta_nodes.csv`
- `csv 데이터/clean/graph_edges_bipartite.csv`

## 7. API 명세

### GET /health
헬스 체크

**응답:**
```json
{
  "status": "ok",
  "message": "API is running"
}
```

---

### POST /build-graph
그래프 생성 (CSV로 저장)

**요청:**
```json
{
  "contents": [Array of objects],
  "av": [Array of objects (optional)],
  "game": [Array of objects (optional)],
  "webnovel": [Array of objects (optional)],
  "raw_item": [Array of objects (optional)]
}
```

**응답 (성공):**
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

**응답 (실패):**
```json
{
  "success": false,
  "error": "에러 메시지"
}
```

---

### POST /build-graph-get-result
그래프 생성 후 결과 반환 (메모리에서)

**요청:** `/build-graph`와 동일

**응답 (성공):**
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

## 8. 문제 해결

### 포트 5000이 이미 사용 중인 경우

**api_server.py 수정:**
```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)  # 포트 변경
```

### 모듈 import 오류

```bash
pip install --upgrade flask flask-cors flasgger
```

### 인코딩 문제

JSON 데이터가 UTF-8로 인코딩되어 있는지 확인하세요.

## 9. 개발 vs 프로덕션

### 개발 모드 (현재)
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### 프로덕션 모드 (배포 시)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```
