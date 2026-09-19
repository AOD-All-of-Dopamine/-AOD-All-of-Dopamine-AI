"""엔진 어댑터 — 문자열 키로 받은 요청을 플랫폼의 평가 경로(`next_page`)에 확정값 그대로 넘긴다.

규칙(REC_TAB_DESIGN §8-2): 코퍼스 밖 시드는 예외 대신 제외(droppedSeeds) · 남은 시드 0 = 빈 결과 + exhausted ·
랭킹에는 손대지 않는다.
"""
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Hashable

from aod_serving.engine.overrides import EffectiveConfig


@dataclass(frozen=True)
class AdapterItem:
    key: str
    dominant_seed: str | None
    final: float
    sim: float
    factors: dict[str, float] = field(default_factory=dict)
    episode_count: int | None = None


@dataclass(frozen=True)
class AdapterResult:
    items: list[AdapterItem]
    dropped_seeds: list[str]
    exhausted: bool


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def frame_items(frame, *, key_col: str, to_key: Callable, dominant_to_key: Callable, factor_cols: dict[str, str],
                episode_of: Callable[[Hashable], int | None] | None = None) -> list[AdapterItem]:
    """랭커가 돌려준 프레임 → 항목. 점수 인자는 **프레임에 이미 있는 컬럼만** 싣는다(랭커를 계측하지 않는다)."""
    if len(frame) == 0:
        return []
    present = {name: col for name, col in factor_cols.items() if col in frame.columns}
    has_dom = "dominant_seed" in frame.columns
    out = []
    for row in frame.to_dict("records"):
        factors = {name: v for name, col in present.items() if (v := _num(row[col])) is not None}
        dom = row["dominant_seed"] if has_dom else None
        out.append(AdapterItem(
            key=to_key(row[key_col]),
            dominant_seed=None if dom is None or (isinstance(dom, float) and math.isnan(dom)) else dominant_to_key(dom),
            final=v if (v := _num(row.get("final_score"))) is not None else 0.0,
            sim=v if (v := _num(row.get("seed_similarity"))) is not None else 0.0,
            factors=factors, episode_count=episode_of(row[key_col]) if episode_of else None))
    return out


class EngineAdapter(ABC):
    platform: ClassVar[str]
    factor_schema: ClassVar[str]
    supports_media: ClassVar[bool] = False

    def __init__(self, artifacts: str | Path, config: EffectiveConfig):
        self.artifacts = Path(artifacts)
        self.config = config

    @abstractmethod
    def load(self) -> None: """무거운 적재(임베딩·dataset). 프로세스당 한 번."""
    @abstractmethod
    def parse_key(self, key: str) -> Hashable | None: """문자열 키 → 플랫폼 키. 형식이 틀렸거나 코퍼스에 없으면 None."""
    @abstractmethod
    def first_key(self) -> str: """예열용 고정 시드 — dataset 의 첫 키."""
    @abstractmethod
    def _next_page(self, *, k: int, seeds: list, disliked: list, excluded: list, seen: list, media: str | None): ...
    @abstractmethod
    def _items(self, frame) -> list[AdapterItem]: ...

    def _known_only(self, keys) -> list:
        out, dup = [], set()
        for s in keys or ():
            n = self.parse_key(s)
            if n is not None and n not in dup:
                dup.add(n); out.append(n)
        return out

    def recommend(self, *, k: int, seeds, disliked=(), excluded=(), seen=(), media: str | None = None) -> AdapterResult:
        if media is not None and not self.supports_media:
            raise ValueError(f"{self.platform} 은 media 를 받지 않는다")
        dis = self._known_only(disliked); dis_set = set(dis)
        # 파싱을 먼저 하고 나서 중복을 없앤다 — 원문 문자열로 먼저 걸러내면 "7"·"007" 처럼 같은
        # 코퍼스 키를 가리키는 다른 표기가 둘 다 살아남아 네이티브 시드가 중복된다(§8-2 재발 방지).
        native, dropped, native_seen, dropped_seen = [], [], set(), set()
        for s in seeds:                      # 순서를 지킨다 — 시드 순서가 버킷 순서다(§6-4)
            n = self.parse_key(s)
            if n is None:
                if s not in dropped_seen:      # 모르는/잘못된 시드는 원문 문자열 기준으로 중복 제거
                    dropped_seen.add(s); dropped.append(s)
            elif n in dis_set:                 # 싫어요가 이긴다. 코퍼스 밖이 아니므로 dropped 에는 넣지 않는다
                continue
            elif n not in native_seen:         # 파싱된(네이티브) 키 기준으로 중복 제거
                native_seen.add(n); native.append(n)
        if not native:
            return AdapterResult([], dropped, True)
        frame = self._next_page(k=k, seeds=native, disliked=dis, excluded=self._known_only(excluded),
                                seen=self._known_only(seen), media=media)
        items = self._items(frame)
        return AdapterResult(items, dropped, exhausted=len(items) < k)


class IntKeyMixin:
    """정수 키 플랫폼(Steam·웹툰·웹소설). `self._known` = 코퍼스에 있는 키 집합."""
    _known: set[int]

    def parse_key(self, key: str) -> int | None:
        if not (isinstance(key, str) and key.isascii() and key.isdigit()):
            return None
        if len(key) > 19:          # int64 자릿수 상한 — 이보다 길면 코퍼스 키일 수 없다(파싱만으로도 비싸질 수 있다)
            return None
        n = int(key)
        return n if n in self._known else None
