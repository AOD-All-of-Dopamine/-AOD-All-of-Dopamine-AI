"""시드 → 후보. 시드별 질의(평균 내지 않음) → 접기 → 정렬 → 후처리.

`next_page()` 계약은 세 플랫폼과 같다: 이미 보여준 것을 `exclude` 로 받아 그 아래를 잇는다.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from src.config import artifact_dir, PRODUCTION, POSTPROCESS
from src.personalization.personalized_ranker import PersonalizedRanker
from src.postprocess import postprocess

POOL_FLOOR = 400   # 후보 풀 고정 하한 (k 와 무관하게 같은 정렬을 보장)


class Engine:
    def __init__(self, artifacts=None):
        d = artifact_dir(artifacts)
        self.emb = np.load(d / "corpus_embeddings.npy").astype(np.float32)
        self.emb /= (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9)
        self.ds = pd.read_parquet(d / "dataset.parquet").reset_index(drop=True)
        self.row = {int(v): i for i, v in enumerate(self.ds["item_id"])}
        self.centroid = self.emb.mean(axis=0)
        self.centroid /= (np.linalg.norm(self.centroid) + 1e-9)
        from src.personalization.personalized_ranker import clean_tags   # 랭커와 같은 정지어 규칙
        self._tags = clean_tags(self.ds)                    # 현행
        self._tags_ng = clean_tags(self.ds, drop_genre=True)  # T-10 축: 장르명 태그 제외

    # ── 검색 ──
    def similarity(self, seed_ids, hub_lambda=None) -> np.ndarray:
        """시드 × 코퍼스 행렬. 시드마다 따로 — 취향이 갈리는 사람은 평균이 아무 데도 안 가리킨다."""
        hl = PRODUCTION["hub_lambda"] if hub_lambda is None else hub_lambda
        V = []
        for s in seed_ids:
            v = self.emb[self.row[int(s)]].copy()
            if hl:
                v = v - hl * self.centroid
                v /= (np.linalg.norm(v) + 1e-9)
            V.append(v)
        return np.array(V, dtype=np.float32) @ self.emb.T

    @staticmethod
    def fold(sim: np.ndarray, strategy: str):
        """접기: 시드별 유사도를 후보 하나의 점수로."""
        if strategy == "mean" or sim.shape[0] <= 2:
            return sim.mean(axis=0)
        if strategy == "max":
            return sim.max(axis=0)
        return np.sort(sim, axis=0)[-2:].mean(axis=0)      # top2_mean

    def recommend(self, seed_ids, k=50, exclude=None, *, strategy=None, pop_boost=None,
                  star_boost=None, tag_w=None, hub_lambda=None, creator_w=None, tag_drop_genre=None,
                  disliked_ids=None, dislike_w=None, dislike_floor=None, **pp):
        """`disliked_ids` — 싫어요(DISLIKE)한 작품. 항상 결과에서 빠지고 **같은 시리즈도** 빠진다(제품 결정 2026-09-15).
        `dislike_w > 0` 이면 접힌 점수에서 `w × Σ_i max(0, sim(후보, 싫어요_i) − dislike_floor)` 를 뺀다 (T-11).
        좋아요와 같은 `similarity`(허브 보정 포함)로 계산해 척도가 같다. Steam `run_multi` 와 같은 공식이다."""
        seed_ids = [int(s) for s in seed_ids if int(s) in self.row]
        if not seed_ids:
            return pd.DataFrame(columns=["item_id", "name", "rank"])
        disliked = [int(d) for d in (disliked_ids or []) if int(d) in self.row]
        strategy = strategy or PRODUCTION["strategy"]
        sim = self.similarity(seed_ids, hub_lambda)
        folded = self.fold(sim, strategy)
        dw = PRODUCTION["dislike_w"] if dislike_w is None else dislike_w
        dfl = PRODUCTION["dislike_floor"] if dislike_floor is None else dislike_floor
        if disliked and dw:
            # 임계 아래는 0 — 무관한 싫어요가 목록을 흔들지 않게 한다. 합산 — 비슷한 것을 여러 번 싫어요 하면 누적된다.
            folded = folded - dw * np.clip(self.similarity(disliked, hub_lambda) - dfl, 0.0, None).sum(axis=0)
        dominant = [seed_ids[i] for i in sim.argmax(axis=0)]
        # 명시적 0 은 0 이다 — `or` 는 0.0 을 설정 기본값으로 바꿔 버린다 (TMDB 하네스 D-55 와 같은 함정).
        tag_w = PRODUCTION["tag_w"] if tag_w is None else tag_w
        tdg = PRODUCTION["tag_drop_genre"] if tag_drop_genre is None else tag_drop_genre
        tags = self._tags_ng if tdg else self._tags
        seed_tags = [tags[self.row[s]] for s in seed_ids] if tag_w else None
        r = PersonalizedRanker(
            self.ds,
            pop_boost=PRODUCTION["pop_boost"] if pop_boost is None else pop_boost,
            star_boost=PRODUCTION["star_boost"] if star_boost is None else star_boost,
            tag_w=PRODUCTION["tag_w"] if tag_w is None else tag_w,
            creator_w=PRODUCTION["creator_w"] if creator_w is None else creator_w,
            tag_drop_genre=tdg,
        )
        ex = set(seed_ids) | set(int(x) for x in (exclude or [])) | set(disliked)
        # **후보 풀 깊이를 k 에 묶지 않는다.** 세 플랫폼은 전부 `rank_n = k × 상수` 라
        # 같은 시드라도 요청한 k 에 따라 상위 목록이 달라진다 — 후처리(시리즈 상한·시드
        # 교차)가 풀 전체를 보고 재배치하기 때문이다. 실측(2026-09-04 검수): Steam 에서
        # 서빙(k=20)과 평가(k=50)의 top-20 이 15프로필 중 7개에서 갈렸다(최소 겹침 85%).
        # 여기서는 고정 하한을 둬서 k 가 결과를 바꾸지 못하게 한다.
        seed_creators = [r._creators[self.row[s]] for s in seed_ids] if r.creator_w else None
        ranked = r.rank(folded, exclude_ids=ex, seed_tags=seed_tags, seed_creators=seed_creators,
                        top_n=max(POOL_FLOOR, k * 8), dominant=dominant)
        opts = {**POSTPROCESS, **pp}
        return postprocess(ranked, self.ds, top_n=k, seed_ids=seed_ids,
                           series_max=opts["series_max"], artist_max=opts["artist_max"],
                           drop_adult=opts["drop_adult"], drop_series_ids=disliked)

    def next_page(self, seed_ids, k=50, seen=None, **kw):
        """제품 경로. `seen` 아래를 잇는다 — 새로고침해도 앞 페이지가 다시 나오지 않는다.
        싫어요는 `disliked_ids=` 로 매 호출 현재 목록을 넘긴다 (`recommend` 참고)."""
        return self.recommend(seed_ids, k=k, exclude=seen or [], **kw)
