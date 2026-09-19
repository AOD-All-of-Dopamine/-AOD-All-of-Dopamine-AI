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

    def _restrict_from(self, allowed: set) -> frozenset:
        """웹툰만 **플랫폼 코드 변경이 없다.** `Engine.next_page(seen=…)` 는 제외 전용이라
        (랭커의 `exclude_ids` 로만 가고 후처리에는 안 간다 — `personalized_retrieve.next_page`
        docstring) 목록 밖 id 를 그냥 `seen` 에 합치면 된다. 코퍼스가 3,687편이라 집합 합치기
        비용도 무시할 수 있다."""
        return frozenset(self._known - allowed)

    def _next_page(self, *, k, seeds, disliked, excluded, seen, media, restrict=None):
        merged = set(seen) | set(excluded)
        if restrict is not None:
            merged |= restrict
        return self._fn(seeds, k=k, seen=sorted(merged), disliked_ids=list(disliked) or None,
                        **self.config.production, **self.config.postprocess)

    def _items(self, frame):
        return frame_items(frame, key_col="item_id", to_key=lambda v: str(int(v)), dominant_to_key=lambda v: str(int(v)), factor_cols={})
