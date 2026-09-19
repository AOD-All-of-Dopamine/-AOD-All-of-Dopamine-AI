"""웹소설 — `webnovel/src/personalized_retrieve.next_page`. 싫어요·관심 없음 인자가 없어 `seen_ids` 에 합친다.
이 경로는 `drop_excluded_series` 로 같은 판본(작가|제목 키)도 함께 빠진다 — 알려진 차이(§8-3).
항목마다 dataset 의 `episode_count` 를 싣는다 — 전체 탭 M6 의 "20화 미만 제외"가 쓴다."""
from __future__ import annotations
from aod_serving.engine.adapters.base import EngineAdapter, IntKeyMixin, frame_items, row_mask


class WebnovelAdapter(IntKeyMixin, EngineAdapter):
    platform = "webnovel"
    factor_schema = "webnovel.v1"
    FACTORS = {"interest_pct": "interest_percentile"}

    def load(self) -> None:
        import numpy as np, pandas as pd
        from src.personalized_retrieve import build_components, next_page
        comps = build_components(pop_boost=self.config.production["pop_boost"], artifacts=str(self.artifacts))
        loader, retriever, ds = comps[0], comps[1], comps[3].dataset   # dataset 은 item_id 로 인덱싱돼 있다
        episodes = {int(i): (None if pd.isna(v) else int(v)) for i, v in ds["episode_count"].items()}
        # 코퍼스 키 스냅샷 — Steam 어댑터와 같은 이유·같은 순서(`full_corpus_frame()` 의 행 순서).
        self._bind(next_page, comps, known={int(a) for a in loader.id_to_row}, first=int(loader.index["item_id"].iloc[0]),
                   episodes=episodes,
                   corpus_keys=np.asarray(retriever.index["item_id"].to_numpy(), dtype=np.int64))

    def _bind(self, fn, comps, *, known: set[int], first: int, episodes: dict[int, int | None], corpus_keys=None) -> None:
        self._fn, self._comps, self._known, self._first, self._episodes = fn, comps, known, first, episodes
        self._corpus_keys = corpus_keys

    def first_key(self) -> str:
        return str(self._first)

    def _restrict_from(self, allowed: set):
        """`blocked_rows` — 코퍼스 행 순서의 불리언 배열(True = 후보에서 뺀다)."""
        import numpy as np
        m = np.logical_not(row_mask(self._corpus_keys, allowed))
        m.flags.writeable = False
        return m

    def _next_page(self, *, k, seeds, disliked, excluded, seen, media, restrict=None):
        p = self.config.production
        # 목록은 `seen_ids` 에 합치지 않는다 — 그쪽은 `drop_excluded_series` 로 같은 작품의 다른
        # 판본까지 지우는 경로라, 목록 안에 있는 판본이 목록 밖 판본 때문에 사라진다(W-6).
        extra = {} if restrict is None else {"blocked_rows": restrict}
        return self._fn(seeds, seen_ids=set(seen) | set(excluded) | set(disliked), page_size=k, strategy=p["strategy"],
                        pop_boost=p["pop_boost"], components=self._comps, min_interest_count=p["min_interest_count"],
                        drop_excluded_series=p["drop_excluded_series"], **extra)

    def _items(self, frame):
        return frame_items(frame, key_col="item_id", to_key=lambda v: str(int(v)), dominant_to_key=lambda v: str(int(v)),
                           factor_cols=self.FACTORS, episode_of=lambda v: self._episodes.get(int(v)))
