"""엔진·라우터 요청/응답 모델. 계약: REC_TAB_DESIGN §4-2 (키는 문자열 — 스펙 2026-09-19 §4)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator
from pydantic.alias_generators import to_camel

Platform = Literal["steam", "tmdb", "webtoon", "webnovel"]
Tab = Literal["all", "movie", "tv", "game", "webtoon", "webnovel"]
Key = StrictStr

MAX_K = 100
MAX_SEEDS = 50
MAX_EXCLUSIONS = 5000      # disliked + excluded + seen 합계


class _Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class EngineRequest(_Model):
    k: int = Field(ge=1, le=MAX_K)
    seeds: list[Key] = Field(max_length=MAX_SEEDS)
    disliked: list[Key] = []
    excluded: list[Key] = []
    seen: list[Key] = []
    media: Literal["movie", "tv"] | None = None      # TMDB 전용

    @model_validator(mode="after")
    def _limit_exclusions(self):
        n = len(self.disliked) + len(self.excluded) + len(self.seen)
        if n > MAX_EXCLUSIONS:
            raise ValueError(f"disliked+excluded+seen 합계 {n} > {MAX_EXCLUSIONS}")
        return self


class Score(_Model):
    final: float
    sim: float
    factors: dict[str, float] = {}


class EngineItem(_Model):
    key: Key
    rank: int
    dominant_seed: Key | None = None
    score: Score
    episode_count: int | None = None        # 웹소설만 — M6 의 "20화 미만 제외"가 쓴다


class VersionInfo(_Model):
    sha: str
    config: str
    corpus: str


class EngineResponse(_Model):
    platform: Platform
    items: list[EngineItem]
    exhausted: bool
    dropped_seeds: list[Key] = []
    factor_schema: str
    version: VersionInfo
    took_ms: int


class RouterRequest(_Model):
    tab: Tab
    k: int = Field(default=20, ge=1, le=50)
    buffer: int = Field(default=10, ge=0, le=50)
    seeds: dict[Platform, list[Key]]
    disliked: dict[Platform, list[Key]] = {}
    excluded: dict[Platform, list[Key]] = {}
    seen: dict[Platform, list[Key]] = {}


class RouterItem(_Model):
    platform: Platform
    key: Key
    rank: int
    dominant_seed: Key | None = None
    candidate_source: str = "content_sim"
    is_exploration: bool = False
    propensity: float = 1.0
    score: Score
    factor_schema: str


class RouterVersions(_Model):
    router: str
    engines: dict[Platform, VersionInfo] = {}


class RouterResponse(_Model):
    items: list[RouterItem]
    exhausted: dict[Platform, bool]
    dropped_seeds: dict[Platform, list[Key]] = {}
    partial: list[Platform] = []
    versions: RouterVersions
