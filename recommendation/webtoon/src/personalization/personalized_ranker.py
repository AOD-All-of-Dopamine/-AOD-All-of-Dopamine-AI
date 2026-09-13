"""정렬 — `final = 유사도 × (1 + 보정항들)`.

세 플랫폼과 **같은 모양**이다: 곱셈이라 보정항은 더하기가 아니라 배율이다.
웹툰에서 **확정된 항은 아직 하나도 없다.** 아래는 전부 축이고, 각각 사전등록으로 판정한다.

    pop_boost   : 관심 수 백분위    (웹소설 D-66 의 0.03 을 이식, 웹툰 미측정)
    star_boost  : 평점 백분위       (웹소설에선 신호 없었다 D-62, 웹툰 미측정)
    tag_w       : 태그 피복률       (웹툰만의 기회 — 작품당 태그 10~12개. Steam D-49 형태)
                  max_i |후보태그 ∩ 시드ᵢ태그| / |시드ᵢ태그|  — 자카드가 아니다.
                  분모에 시드만 넣는다(후보의 추가 태그를 벌주지 않는다).
"""
from __future__ import annotations
import numpy as np, pandas as pd


def creators_of(r) -> frozenset:
    """작가 집합 — `artists`(글·그림·원작 목록)가 있으면 그것, 없으면 `author` 를 ' / ' 로 쪼갠다."""
    a = getattr(r, "artists", None)
    if a is not None and hasattr(a, "__len__") and len(a):
        return frozenset(str(x).strip() for x in a if str(x).strip())
    return frozenset(x.strip() for x in str(getattr(r, "author", "") or "").split("/") if x.strip())


class PersonalizedRanker:
    def __init__(self, dataset: pd.DataFrame, pop_boost=0.0, star_boost=0.0, tag_w=0.0, creator_w=0.0):
        self.ds = dataset.reset_index(drop=True)
        self.pop_boost, self.star_boost, self.tag_w = pop_boost, star_boost, tag_w
        # T-7 축. **임베딩 텍스트에 작가가 없다**(제목·장르·태그·줄거리, 작가명은 줄거리 홍보문구에
        # 우연히 8.6% 만 등장). TMDB V-2 에서 같은 결손을 감독 정합 항으로 메워 +0.056 을 얻었다.
        # 식은 tag_w 와 같은 시드 **개별** 최대 피복률 |A∩S_i|/|S_i| — 작가 집합은 글·그림·원작.
        self.creator_w = creator_w
        self._creators = [creators_of(r) for r in self.ds.itertuples(index=False)] if creator_w else None
        fav = pd.to_numeric(self.ds.get("favorite_count"), errors="coerce").fillna(0.0)
        self.pop_pct = fav.rank(pct=True).to_numpy(dtype=np.float32)
        star = pd.to_numeric(self.ds.get("star_score"), errors="coerce").fillna(0.0)
        self.star_pct = star.rank(pct=True).to_numpy(dtype=np.float32)
        # 태그 피복은 **표현과 같은 정지어 규칙**을 쓴다. `완결로맨스` 같은 라벨을 세면
        # 피복률이 "같은 장르 + 같은 연재 상태"를 재게 된다 — 취향 신호가 아니다 (T-3 전에 고침).
        from src.text_builder import _TAG_STOP, _TAG_STOP_PREFIX
        def _clean(t):
            xs = t if t is not None and not isinstance(t, str) else ([t] if t else [])
            return frozenset(x for x in xs if x not in _TAG_STOP and not str(x).startswith(_TAG_STOP_PREFIX))
        self._tags = [_clean(t) for t in self.ds.get("tags", pd.Series([None] * len(self.ds)))]

    def rank(self, sim: np.ndarray, *, exclude_ids=None, seed_tags=None, top_n=200,
             dominant=None, seed_creators=None) -> pd.DataFrame:
        """`sim` 은 후보별 접힌 유사도(행 = 코퍼스 순서)."""
        score = sim.astype(np.float32).copy()
        boost = np.zeros_like(score)
        if self.pop_boost:  boost += self.pop_boost * self.pop_pct
        if self.star_boost: boost += self.star_boost * self.star_pct
        if self.tag_w and seed_tags:
            cov = np.array([max((len(self._tags[i] & s) / len(s) if s else 0.0) for s in seed_tags)
                            for i in range(len(self.ds))], dtype=np.float32)
            boost += self.tag_w * cov
        if self.creator_w and seed_creators:
            # 작가 정보가 없는 후보는 0 — 감점이 아니라 무보정이다.
            cf = np.array([max((len(self._creators[i] & c) / len(c) if c else 0.0) for c in seed_creators)
                           for i in range(len(self.ds))], dtype=np.float32)
            boost += self.creator_w * cf
        final = score * (1.0 + boost)
        out = pd.DataFrame({"item_id": self.ds["item_id"].values, "name": self.ds["name"].values,
                            "seed_similarity": score, "final_score": final})
        if dominant is not None: out["dominant_seed"] = dominant
        if exclude_ids: out = out[~out["item_id"].isin(set(int(x) for x in exclude_ids))]
        out = out.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
        out["rank"] = range(1, len(out) + 1)
        return out
