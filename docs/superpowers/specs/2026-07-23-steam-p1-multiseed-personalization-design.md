# P1: Multi-Seed Explicit Personalization — Design Spec

- 작성일: 2026-07-23
- 상태: 초안
- 상위: AOD 개인화 추천 시스템 (2026-06-28)

## 0. Problem & Motivation

S1은 콘텐츠 기반(cold-start) Top-k retrieval이었다. P1은 사용자가 **좋아하는 게임 3~5개를 선택**하면, 그 seed 게임들의 임베딩을 기반으로 개인화된 추천을 생성한다.

핵심 질문: **여러 seed 게임의 유사도를 어떻게 하나의 점수로 aggregate하는가?**

## 1. Approach: Multi-Seed with Aggregation Comparison

Single-vector weighted average (기존 `UserProfileBuilder`) 대신, seed별로 개별 유사도를 계산하고 **명시적 aggregation 함수**로 결합한다.

### Three Aggregation Strategies

| Strategy | Formula | Intuition |
|---|---|---|
| **MAX** | `score(c) = max(sim(c, s1), ..., sim(c, sn))` | "이 게임이 seed 중 하나와 매우 비슷하다" — narrow but precise |
| **MEAN** | `score(c) = mean(sim(c, s1), ..., sim(c, sn))` | "모든 seed와 평균적으로 비슷하다" — broad consensus |
| **TOP2_MEAN** | `score(c) = mean(top2(sim(c, s1), ..., sim(c, sn)))` | "가장 비슷한 두 seed의 평균" — MAX와 MEAN의 절충 |

### Why not weighted average?

Single weighted average는 embedding 공간에서 정보 손실(여러 방향의 선호도를 하나의 벡터로 평균내면서 희석됨)이 발생한다. Multi-seed per-seed similarity는 이 손실 없이 각 seed의 방향을 보존한다.

## 2. Pipeline

```
liked_appids [3~5개]
    ↓
seed_loader: appid → embedding lookup
    ↓ (각 seed별)
candidate_retriever: cosine Top-100 per seed → union
    ↓ (공통 candidate pool)
score_aggregator: MAX / MEAN / TOP2_MEAN
    ↓
personalized_ranker: R1 final_score (rec_boost) 적용, seed 제외
    ↓
ranked results per aggregation
```

## 3. Components

### `src/personalization/seed_loader.py`
- Input: list[int] steam_appid
- Output: dict[int, np.ndarray] mapping appid → embedding vector (3584-dim)
- Validates all appids exist in corpus; raises ValueError if not

### `src/personalization/candidate_retriever.py`
- Input: seed_embeddings dict, top_k_per_seed=100
- For each seed: cosine similarity, sort, take top_k
- Union all candidates across seeds (dedup by appid)
- Output: DataFrame with columns [steam_appid, name]

### `src/personalization/score_aggregator.py`
- Input: candidate pool, seed_embeddings dict
- Per seed: compute cosine similarity for all candidates
- Aggregate: MAX/MEAN/TOP2_MEAN (configurable)
- Output: DataFrame with [steam_appid, aggregated_score, strategy]

### `src/personalization/personalized_ranker.py`
- Input: aggregated candidates
- Apply R1 ranking: `final_score = aggregated_score × (1 + rec_percentile × 0.03)`
- Exclude seed appids from final results
- Output: ranked DataFrame with [steam_appid, name, seed_similarity, recommendations_percentile, final_score, rank]

### `configs/p1.yaml`
Experiment configuration.

### `src/personalized_retrieve.py` (CLI)
- Unified entry point
- Modes: `single` (legacy UserProfileBuilder), `multi` (new pipeline)
- For multi: runs all 3 aggregations, saves results per strategy
- Output: `artifacts/p1/ranked_{strategy}.parquet`

## 4. Evaluation

### Synthetic Profiles (20개)
| Type | Count | Description |
|---|---|---|
| Coherent (유사 장르) | 10 | e.g., FPS 게임 3-4개, RPG 3-4개 |
| Multi-interest (다양 장르) | 10 | e.g., [FPS, 퍼즐, 시뮬레이션] 섞임 |

- Split: Dev 12 (6+6), Val 8 (4+4)
- 각 profile: profile_id, display_name, description, liked_appids[], owned_appids[]

### Pooled Blind Evaluation
- Profile별로 MAX∪MEAN∪TOP2_MEAN Top-10 union 생성
- A/B/C blind 판정 (이 사람이 이 게임들을 좋아할 때 이 candidate가 타당한가?)
- 점수: 0-3 (0=부적절, 1=약간, 2=보통, 3=매우 타당)
- 평가자: LLM proxy (human blind evaluation reserve)

### Metrics
- `NDCG@10`: pooled 내 ranking quality
- `P@10`: precision at 10
- `Conf@10`: confidence-weighted precision
- **Seed Dominance**: MAX에서 특정 seed가 dominant한 비율

### Comparison with baseline
- S1 R1-final (non-personalized) vs P1-MAX vs P1-MEAN vs P1-TOP2
- Expected: P1 variants > S1 baseline for synthetic profiles
- Dev 결과로 aggregation freeze → Val 최종 평가

## 5. File Changes Summary

| File | Status |
|---|---|
| `configs/p1.yaml` | **NEW** |
| `src/personalization/__init__.py` | **NEW** |
| `src/personalization/seed_loader.py` | **NEW** |
| `src/personalization/candidate_retriever.py` | **NEW** |
| `src/personalization/score_aggregator.py` | **NEW** |
| `src/personalization/personalized_ranker.py` | **NEW** |
| `src/personalized_retrieve.py` | **REWRITE** (add multi mode) |
| `src/build_p1_profiles.py` | **NEW** |
| `src/export_personalization_eval.py` | **NEW** |
| `src/eval_personalization.py` | **UPDATE** (multi-seed aware) |

## 6. Out of Scope

- R2 Metacritic boost in P1 (multi-seed aggregation 효과를 먼저 측정)
- Real Steam user data (P1은 synthetic으로 충분)
- Online serving (P1은 batch evaluation만)
- S2 cross-domain (fun_tag) — P1 완료 후 논의
