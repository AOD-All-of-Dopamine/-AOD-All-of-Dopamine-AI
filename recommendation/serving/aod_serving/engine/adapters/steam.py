"""Steam — `steam/src/personalized_retrieve.next_page`. 싫어요는 제외 + 유사 감점(w 2.0), 관심 없음은 seen 에 합친다(§8-3)."""
from __future__ import annotations
from aod_serving.engine.adapters.base import EngineAdapter, IntKeyMixin, frame_items, row_mask


class SteamAdapter(IntKeyMixin, EngineAdapter):
    platform = "steam"
    factor_schema = "steam.v1"
    FACTORS = {"rec_pct": "recommendations_percentile", "quality": "quality", "tag_fit": "tag_fit", "has_mc": "has_mc"}

    def load(self) -> None:
        import numpy as np
        from src.personalized_retrieve import build_components, next_page
        p = self.config.production
        comps = build_components(rec_boost=p["rec_boost"], artifacts=str(self.artifacts), trend_weight=p["trend_weight"],
                                 quality_w=p["quality_w"], quality_cap=p["quality_cap"], quality_src=p["quality_src"],
                                 tag_w=p["tag_w"], mc_w=p["mc_w"])
        loader, retriever = comps[0], comps[1]
        # 코퍼스 키를 **계산 스레드에서** numpy 로 떠 둔다 — 서빙 가능 목록 갱신(타이머 스레드)이
        # pandas·Arrow 를 만지지 않게 하려는 것이다(app.py 머리말). 순서는 `full_corpus_frame()`
        # 이 쓰는 `retriever.index` 행 순서 그대로여야 `blocked_rows` 가 행끼리 맞는다.
        self._bind(next_page, comps, known={int(a) for a in loader.appid_to_row},
                   first=int(loader.index["steam_appid"].iloc[0]),
                   corpus_keys=np.asarray(retriever.index["steam_appid"].to_numpy(), dtype=np.int64))

    def _bind(self, fn, comps, *, known: set[int], first: int, corpus_keys=None) -> None:
        self._fn, self._comps, self._known, self._first = fn, comps, known, first
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
        # 목록이 꺼져 있으면 인자를 **아예 넘기지 않는다** — 평가 경로 호출 모양이 예전 그대로다.
        extra = {} if restrict is None else {"blocked_rows": restrict}
        return self._fn(seeds, seen_appids=set(seen) | set(excluded), page_size=k, strategy=p["strategy"],
                        rec_boost=p["rec_boost"], components=self._comps, disliked_appids=list(disliked) or None,
                        **extra)

    def _items(self, frame):
        return frame_items(frame, key_col="steam_appid", to_key=lambda v: str(int(v)),
                           dominant_to_key=lambda v: str(int(v)), factor_cols=self.FACTORS)
