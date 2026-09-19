"""TMDB — 밖에서는 `item_id`(`movie_603`), 엔진 안에서는 코퍼스 **행 번호**. 행 번호는 코퍼스 버전마다 달라져 밖으로 내보내지 않는다.
싫어요·관심 없음 인자가 없어 전부 `seen_rows` 에 합친다(제외만, §8-3)."""
from __future__ import annotations
from aod_serving.engine.adapters.base import EngineAdapter, frame_items, index_mask


class TmdbAdapter(EngineAdapter):
    platform = "tmdb"
    factor_schema = "tmdb.v1"
    supports_media = True

    def load(self) -> None:
        from src.personalized_retrieve import build_components, next_page
        p = self.config.production
        comps = build_components(artifacts=str(self.artifacts), **{k: v for k, v in p.items() if k != "strategy"})
        self._bind(next_page, comps, item_ids=[str(x) for x in comps[3].dataset["item_id"]])

    def _bind(self, fn, comps, *, item_ids: list[str]) -> None:
        self._fn, self._comps, self._item_ids = fn, comps, item_ids
        self._row_of = {iid: row for row, iid in enumerate(item_ids)}

    def parse_key(self, key: str):
        return self._row_of.get(key) if isinstance(key, str) else None

    def corpus_keys(self) -> list[str]:
        return list(self._item_ids)

    def first_key(self) -> str:
        return self._item_ids[0]

    def _restrict_from(self, allowed: set):
        """`servable_rows` — 허용 행만 True 인 불리언 배열. TMDB 는 키가 곧 행이라 색인으로 바로 켠다.

        `_item_ids`(적재 때 만든 파이썬 문자열 리스트)와 `parse_key`(dict 조회)만 쓰므로 여기서
        pandas 문자열 배열이 새로 생기지 않는다 — 타이머 스레드에서 불려도 Arrow 를 안 건드린다.
        """
        return index_mask(len(self._item_ids), allowed)

    def _next_page(self, *, k, seeds, disliked, excluded, seen, media, restrict=None):
        # `media` 와 **같은 자리**에서 AND 로 합쳐진다 — 풀 깊이 rank_n 이 유지된다(recommend 참고).
        extra = {} if restrict is None else {"servable_rows": restrict}
        return self._fn(seeds, seen_rows=set(seen) | set(excluded) | set(disliked), page_size=k, components=self._comps,
                        strategy=self.config.production["strategy"], postprocess_kwargs=dict(self.config.postprocess),
                        media=media, **extra)

    def _items(self, frame):
        return frame_items(frame, key_col="item_id", to_key=str, dominant_to_key=lambda row: self._item_ids[int(row)], factor_cols={})
