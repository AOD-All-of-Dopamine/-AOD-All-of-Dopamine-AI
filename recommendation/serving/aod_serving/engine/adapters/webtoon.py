"""웹툰 — `webtoon/src/personalized_retrieve.Engine`. 싫어요는 그 작품과 같은 시리즈를 뺀다(감점 0, T-11), 관심 없음은 seen 에 합친다."""
from __future__ import annotations
from aod_serving.engine.adapters.base import EngineAdapter, IntKeyMixin, frame_items


class WebtoonAdapter(IntKeyMixin, EngineAdapter):
    platform = "webtoon"
    factor_schema = "webtoon.v1"

    def load(self) -> None:
        from src.personalized_retrieve import Engine
        eng = Engine(str(self.artifacts))
        self._bind(eng.next_page, known={int(k) for k in eng.row}, first=int(eng.ds["item_id"].iloc[0]))

    def _bind(self, fn, *, known: set[int], first: int) -> None:
        self._fn, self._known, self._first = fn, known, first

    def first_key(self) -> str:
        return str(self._first)

    def _next_page(self, *, k, seeds, disliked, excluded, seen, media):
        merged = sorted(set(seen) | set(excluded))
        return self._fn(seeds, k=k, seen=merged, disliked_ids=list(disliked) or None,
                        **self.config.production, **self.config.postprocess)

    def _items(self, frame):
        return frame_items(frame, key_col="item_id", to_key=lambda v: str(int(v)), dominant_to_key=lambda v: str(int(v)), factor_cols={})
