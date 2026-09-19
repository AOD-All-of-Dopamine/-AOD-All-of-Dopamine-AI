"""웹소설 — `webnovel/src/personalized_retrieve.next_page`. 싫어요·관심 없음 인자가 없어 `seen_ids` 에 합친다.
이 경로는 `drop_excluded_series` 로 같은 판본(작가|제목 키)도 함께 빠진다 — 알려진 차이(§8-3).
항목마다 dataset 의 `episode_count` 를 싣는다 — 전체 탭 M6 의 "20화 미만 제외"가 쓴다."""
from __future__ import annotations
from aod_serving.engine.adapters.base import EngineAdapter, IntKeyMixin, frame_items


class WebnovelAdapter(IntKeyMixin, EngineAdapter):
    platform = "webnovel"
    factor_schema = "webnovel.v1"
    FACTORS = {"interest_pct": "interest_percentile"}

    def load(self) -> None:
        import pandas as pd
        from src.personalized_retrieve import build_components, next_page
        comps = build_components(pop_boost=self.config.production["pop_boost"], artifacts=str(self.artifacts))
        loader, ds = comps[0], comps[3].dataset          # dataset 은 item_id 로 인덱싱돼 있다
        episodes = {int(i): (None if pd.isna(v) else int(v)) for i, v in ds["episode_count"].items()}
        self._bind(next_page, comps, known={int(a) for a in loader.id_to_row}, first=int(loader.index["item_id"].iloc[0]),
                   episodes=episodes)

    def _bind(self, fn, comps, *, known: set[int], first: int, episodes: dict[int, int | None]) -> None:
        self._fn, self._comps, self._known, self._first, self._episodes = fn, comps, known, first, episodes

    def first_key(self) -> str:
        return str(self._first)

    def _next_page(self, *, k, seeds, disliked, excluded, seen, media):
        p = self.config.production
        return self._fn(seeds, seen_ids=set(seen) | set(excluded) | set(disliked), page_size=k, strategy=p["strategy"],
                        pop_boost=p["pop_boost"], components=self._comps, min_interest_count=p["min_interest_count"],
                        drop_excluded_series=p["drop_excluded_series"])

    def _items(self, frame):
        return frame_items(frame, key_col="item_id", to_key=lambda v: str(int(v)), dominant_to_key=lambda v: str(int(v)),
                           factor_cols=self.FACTORS, episode_of=lambda v: self._episodes.get(int(v)))
