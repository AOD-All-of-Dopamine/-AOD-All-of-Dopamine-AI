# 🚀 완전한 API 시스템 (All-of-Dopamine AI 추천 파이프라인)

## 📋 시스템 개요

모든 그래프/임베딩 생성 프로세스가 REST API로 통합되었습니다. CSV 기반의 독립적 스크립트에서 **통합 API 서버**로 진화했습니다.

| 단계 | 모듈 | 입력 | 출력 | 엔드포인트 |
|------|------|------|------|----------|
| 1️⃣ | `make_bigraph.py` | 콘텐츠 CSV (4개) | 그래프 노드/엣지/메타노드 (4개) | `/build-graph` + `-get-result` |
| 2️⃣ | `make_itemgraph_api.py` | 그래프 노드 + 장르 | 아이템-아이템 엣지 | `/build-item-graph` + `-get-result` |
| 3️⃣ | `make_itemembedding_api.py` | 아이템 엣지 | 64차원 임베딩 | `/build-item-embeddings` + `-get-result` |
| 4️⃣ | `make_userembedding_api.py` | 아이템 임베딩 + 선호도 | 사용자 임베딩 | `/build-user-embeddings` + `-get-result` |

---

## 🎯 전체 워크플로우

```
Raw Content Data (CSV)
    ↓
[1] POST /build-graph-get-result
    ├─ Inputs: contents, av, game, webnovel, raw_item
    ├─ Output: nodes, raw_genres, meta_nodes, edges
    ↓
[2] POST /build-item-graph-get-result
    ├─ Inputs: nodes, raw_genres (from step 1)
    ├─ Output: item_edges
    ↓
[3] POST /build-item-embeddings-get-result
    ├─ Inputs: item_edges (from step 2)
    ├─ Output: item_embeddings (64-dim)
    ↓
[4] POST /build-user-embeddings-get-result
    ├─ Inputs: 
    │   - item_embeddings (from step 3)
    │   - raw_genres (from step 1)
    │   - user_preferred_genres
    ├─ Output: user_embeddings
    ↓
🎯 Recommendation Ready!
```

---

## 📡 API 엔드포인트 상세 명세

### 🔗 1. 이분 그래프 생성 API

#### `POST /build-graph` (CSV 저장)
콘텐츠 데이터 → 그래프 생성 → CSV 4개 파일 저장

**입력:**
```json
{
  "contents": [
    {"id": "c1", "title": "Anime 1", "updated_date": "2024-01-01", ...}
  ],
  "av": [...],
  "game": [...],
  "webnovel": [...],
  "raw_item": [...]
}
```

**출력:**
```json
{
  "success": true,
  "message": "그래프가 성공적으로 생성되었습니다.",
  "stats": {
    "nodes": 15342,
    "meta_nodes": 2854,
    "edges": 94028
  }
}
```

**저장 파일:**
- `graph_nodes.csv` (노드 정보)
- `content_raw_genres.csv` (원본 장르)
- `meta_nodes.csv` (메타 노드)
- `graph_edges_bipartite.csv` (이분 그래프 엣지)

---

#### `POST /build-graph-get-result` (JSON 반환)
CSV 저장 없이 모든 결과를 JSON으로 즉시 반환

**입력:** 위와 동일
**출력:**
```json
{
  "success": true,
  "stats": {"nodes": 15342, "meta_nodes": 2854, "edges": 94028},
  "nodes": [...],
  "raw_genres": [...],
  "meta_nodes": [...],
  "edges": [...]
}
```

---

### 🔗 2. 아이템-아이템 그래프 생성 API

#### `POST /build-item-graph` (CSV 저장)
아이템 노드 + 장르 정보 → 아이템 간 유사도 엣지 생성

**입력:**
```json
{
  "nodes": [{"id": "c1", "title": "..."}, ...],
  "raw_genres": [{"content_id": "c1", "source": "...", "raw_genre_1": "..."}, ...]
}
```

**출력:**
```json
{
  "success": true,
  "message": "아이템 그래프가 성공적으로 생성되었습니다.",
  "stats": {
    "edges_count": 156430
  }
}
```

**저장 파일:**
- `item_edges.csv` (아이템-아이템 엣지)

---

#### `POST /build-item-graph-get-result` (JSON 반환)
아이템 그래프 엣지를 JSON으로 반환

**입력:** 위와 동일
**출력:**
```json
{
  "success": true,
  "stats": {"edges_count": 156430},
  "edges": [
    {"src_content_id": "c1", "dst_content_id": "c2", "weight": 0.85},
    ...
  ]
}
```

---

### 🧠 3. 아이템 임베딩 생성 API (Node2Vec)

#### `POST /build-item-embeddings` (CSV 저장)
아이템 그래프 → Node2Vec + Skip-gram → 64차원 임베딩 생성

**입력:**
```json
{
  "edges": [
    {"src_content_id": "c1", "dst_content_id": "c2", "weight": 0.85},
    ...
  ],
  "dim": 64,
  "walk_length": 40,
  "num_walks": 10,
  "epochs": 3,
  "batch_size": 8192,
  "lr": 0.025
}
```

**출력:**
```json
{
  "success": true,
  "message": "아이템 임베딩이 성공적으로 생성되었습니다.",
  "stats": {
    "items_count": 15342,
    "embedding_dim": 64
  }
}
```

**저장 파일:**
- `item_embeddings_torch.csv` (content_id, emb_0, ..., emb_63)

**하이퍼파라미터:**
| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| dim | 64 | 임베딩 차원 |
| walk_length | 40 | 랜덤 워크 길이 |
| num_walks | 10 | 노드당 워크 개수 |
| epochs | 3 | 학습 에포크 |
| batch_size | 8192 | 배치 사이즈 |
| lr | 0.025 | 학습률 |

---

#### `POST /build-item-embeddings-get-result` (JSON 반환)
아이템 임베딩을 JSON으로 반환

**입력:** 위와 동일
**출력:**
```json
{
  "success": true,
  "stats": {"items_count": 15342, "embedding_dim": 64},
  "embeddings": [
    {"content_id": "c1", "emb_0": 0.123, "emb_1": -0.456, ...},
    ...
  ]
}
```

---

### 👤 4. 사용자 임베딩 생성 API

#### `POST /build-user-embeddings` (CSV 저장)
사용자 선호 장르 + 아이템 임베딩 → 사용자 임베딩 생성

**입력:**
```json
{
  "item_embeddings": [
    {"content_id": "c1", "emb_0": 0.123, "emb_1": -0.456, ...}
  ],
  "raw_genres": [
    {"content_id": "c1", "source": "...", "raw_genre": "Action"}
  ],
  "user_preferred_genres": [
    {"user_id": "u1", "genre": "Action", "username": "user_name"}
  ],
  "contents": [...]
}
```

**출력:**
```json
{
  "success": true,
  "message": "유저 임베딩이 성공적으로 생성되었습니다.",
  "stats": {
    "users_count": 8521
  }
}
```

**저장 파일:**
- `user_embeddings.csv` (user_id, username, emb_0, ..., emb_63)

---

#### `POST /build-user-embeddings-get-result` (JSON 반환)
사용자 임베딩을 JSON으로 반환

**입력:** 위와 동일
**출력:**
```json
{
  "success": true,
  "stats": {"users_count": 8521},
  "embeddings": [
    {"user_id": "u1", "username": "user_name", "emb_0": 0.234, ...},
    ...
  ]
}
```

---

## 🛠️ 기술 스택

| 기술 | 버전 | 용도 |
|------|------|------|
| Flask | 3.1.2 | REST API 프레임워크 |
| Flasgger | 0.9.7.1 | Swagger UI 자동 생성 |
| Pandas | 2.x | DataFrame 조작 |
| NumPy | 1.x | 수치 연산 |
| PyTorch | 2.x | 신경망 학습 (Node2Vec) |
| Python | 3.12 | 런타임 |

---

## 🚀 서버 실행 및 테스트

### 1. 서버 시작
```bash
cd "c:\Users\LG\Desktop\2025-2\AOD\-AOD-All-of-Dopamine-AI"
python api_server.py
```

**성공 메시지:**
```
[ItemEmbedding] Using device: cpu
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.0.7:5000
Press CTRL+C to quit
```

### 2. Swagger UI 접속
```
http://localhost:5000/apidocs
```

모든 엔드포인트의 대화형 문서 및 테스트 인터페이스 제공

### 3. Health Check
```bash
curl http://localhost:5000/health
# 응답: {"status": "ok", "message": "API is running"}
```

---

## 📝 Python 클라이언트 예제

### 전체 파이프라인 실행

```python
import requests
import json

BASE_URL = "http://localhost:5000"

# 1️⃣ 이분 그래프 생성
print("Step 1: 이분 그래프 생성...")
graph_data = {
    "contents": [...],
    "av": [...],
    "game": [...],
    "webnovel": [...],
    "raw_item": [...]
}
r1 = requests.post(f"{BASE_URL}/build-graph-get-result", json=graph_data)
nodes = r1.json()['nodes']
raw_genres = r1.json()['raw_genres']
edges = r1.json()['edges']

# 2️⃣ 아이템 그래프 생성
print("Step 2: 아이템 그래프 생성...")
r2 = requests.post(f"{BASE_URL}/build-item-graph-get-result", json={
    "nodes": nodes,
    "raw_genres": raw_genres
})
item_edges = r2.json()['edges']

# 3️⃣ 아이템 임베딩 생성 (Node2Vec)
print("Step 3: 아이템 임베딩 생성...")
r3 = requests.post(f"{BASE_URL}/build-item-embeddings-get-result", json={
    "edges": item_edges,
    "dim": 64,
    "epochs": 3
})
item_embeddings = r3.json()['embeddings']

# 4️⃣ 사용자 임베딩 생성
print("Step 4: 사용자 임베딩 생성...")
user_prefs = [
    {"user_id": "u1", "genre": "Action"},
    {"user_id": "u1", "genre": "Fantasy"},
]
r4 = requests.post(f"{BASE_URL}/build-user-embeddings-get-result", json={
    "item_embeddings": item_embeddings,
    "raw_genres": raw_genres,
    "user_preferred_genres": user_prefs
})
user_embeddings = r4.json()['embeddings']

print(f"✅ 완료! {len(user_embeddings)}명의 사용자 임베딩 생성")
```

---

## 📊 성능 특성

### 처리 시간 (예시)

| 단계 | 입력 크기 | 처리 시간 |
|------|----------|---------|
| 이분 그래프 | 15K 콘텐츠 | ~30초 |
| 아이템 그래프 | 156K 엣지 | ~5초 |
| 아이템 임베딩 | 156K 엣지 | ~2-3분 (3 epochs) |
| 사용자 임베딩 | 8.5K 사용자 | ~5초 |
| **전체 파이프라인** | - | **~3-4분** |

### 메모리 사용량
- 기본 데이터: ~500MB
- Node2Vec 모델: ~100MB
- 피크 메모리: ~800MB

---

## 🔧 하이퍼파라미터 가이드

### Node2Vec (아이템 임베딩)

```python
# 빠른 학습 (테스트용)
{
    "dim": 32,           # 차원 감소
    "walk_length": 20,   # 워크 길이 감소
    "num_walks": 5,      # 워크 개수 감소
    "epochs": 1,         # 1 에포크
    "batch_size": 16384  # 배치 증가
}

# 정확한 학습 (프로덕션)
{
    "dim": 128,          # 고차원
    "walk_length": 80,   # 긴 워크
    "num_walks": 20,     # 많은 워크
    "epochs": 5,         # 많은 에포크
    "batch_size": 8192   # 표준 배치
}
```

---

## 📁 파일 구조

```
-AOD-All-of-Dopamine-AI/
├── api_server.py                 # 메인 Flask 서버 (6개 엔드포인트)
├── make_bigraph.py               # 이분 그래프 생성 (API 모드 지원)
├── make_itemgraph_api.py         # 아이템-아이템 그래프 API 모듈
├── make_itemembedding_api.py     # Node2Vec + Skip-gram API 모듈 ✨ NEW
├── make_userembedding_api.py     # 사용자 임베딩 API 모듈
├── requirements.txt              # 의존성
├── COMPLETE_API_SYSTEM.md        # 이 문서
└── csv 데이터/clean/            # 출력 디렉토리
    ├── graph_nodes.csv
    ├── content_raw_genres.csv
    ├── meta_nodes.csv
    ├── graph_edges_bipartite.csv
    ├── item_edges.csv
    ├── item_embeddings_torch.csv
    └── user_embeddings.csv
```

---

## ✅ 완성도

| 컴포넌트 | 상태 | 설명 |
|---------|------|------|
| 이분 그래프 API | ✅ 완료 | 2개 엔드포인트 |
| 아이템 그래프 API | ✅ 완료 | 2개 엔드포인트 |
| 아이템 임베딩 API | ✅ 완료 | 2개 엔드포인트 + Node2Vec |
| 사용자 임베딩 API | ✅ 완료 | 2개 엔드포인트 |
| Swagger UI | ✅ 완료 | 모든 엔드포인트 문서화 |
| 에러 핸들링 | ✅ 완료 | 모든 엔드포인트 |
| **총 엔드포인트** | **8개** | 6개 처리 + 1개 health + 1개 docs |

---

## 🎓 사용 시나리오

### 시나리오 1: 전체 파이프라인 실행
```bash
# 단일 POST 요청으로 전체 프로세스 자동화
python full_pipeline.py
```

### 시나리오 2: 중간 결과 저장
```bash
# CSV 저장 엔드포인트 사용
POST /build-graph
POST /build-item-graph
POST /build-item-embeddings
POST /build-user-embeddings
```

### 시나리오 3: 메모리 기반 실시간 처리
```bash
# JSON 반환 엔드포인트 사용
POST /build-graph-get-result
POST /build-item-graph-get-result
POST /build-item-embeddings-get-result
POST /build-user-embeddings-get-result
```

---

## 🐛 트러블슈팅

### 문제 1: "torch" 모듈 not found
```bash
pip install torch
```

### 문제 2: 포트 5000 이미 사용 중
```bash
# 다른 포트로 변경 (api_server.py line 마지막)
app.run(debug=True, host='0.0.0.0', port=5001)
```

### 문제 3: 메모리 부족
```python
# Node2Vec 하이퍼파라미터 감소
{
    "dim": 32,
    "num_walks": 5,
    "batch_size": 16384
}
```

---

## 📚 참고 자료

- **Flask 문서**: https://flask.palletsprojects.com/
- **Flasgger 문서**: https://flasgger.readthedocs.io/
- **Node2Vec 논문**: https://arxiv.org/abs/1607.00653
- **Skip-gram 모델**: https://arxiv.org/abs/1301.3781

---

## 🎉 시스템 완성!

모든 그래프 생성 및 임베딩 프로세스가 **통합 REST API**로 제공됩니다.

- ✅ CSV 파일 입출력
- ✅ JSON POST/GET 통신
- ✅ Swagger UI 문서
- ✅ 에러 핸들링
- ✅ 하이퍼파라미터 커스터마이징
- ✅ 메모리/파일 저장 옵션

**추천 시스템 준비 완료!** 🚀
