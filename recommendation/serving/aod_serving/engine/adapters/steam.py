"""Steam — `steam/src/personalized_retrieve.next_page`. 싫어요는 제외 + 유사 감점(w 2.0), 관심 없음은 seen 에 합친다(§8-3)."""
from __future__ import annotations
from aod_serving.engine.adapters.base import EngineAdapter, IntKeyMixin, frame_items


class SteamAdapter(IntKeyMixin, EngineAdapter):
    platform = "steam"
    factor_schema = "steam.v1"
    FACTORS = {"rec_pct": "recommendations_percentile", "quality": "quality", "tag_fit": "tag_fit", "has_mc": "has_mc"}

    def load(self) -> None:
        from src.personalized_retrieve import build_components, next_page
        p = self.config.production
        comps = build_components(rec_boost=p["rec_boost"], artifacts=str(self.artifacts), trend_weight=p["trend_weight"],
                                 quality_w=p["quality_w"], quality_cap=p["quality_cap"], quality_src=p["quality_src"],
                                 tag_w=p["tag_w"], mc_w=p["mc_w"])
        loader = comps[0]
        self._bind(next_page, comps, known={int(a) for a in loader.appid_to_row}, first=int(loader.index["steam_appid"].iloc[0]))

    def _bind(self, fn, comps, *, known: set[int], first: int) -> None:
        self._fn, self._comps, self._known, self._first = fn, comps, known, first

    def first_key(self) -> str:
        return str(self._first)

    def _next_page(self, *, k, seeds, disliked, excluded, seen, media):
        p = self.config.production
        return self._fn(seeds, seen_appids=set(seen) | set(excluded), page_size=k, strategy=p["strategy"],
                        rec_boost=p["rec_boost"], components=self._comps, disliked_appids=list(disliked) or None)

    def _items(self, frame):
        return frame_items(frame, key_col="steam_appid", to_key=lambda v: str(int(v)),
                           dominant_to_key=lambda v: str(int(v)), factor_cols=self.FACTORS)
