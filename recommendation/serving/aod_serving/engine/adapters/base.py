"""엔진 어댑터 — 문자열 키로 받은 요청을 플랫폼의 평가 경로(`next_page`)에 확정값 그대로 넘긴다.

규칙(REC_TAB_DESIGN §8-2): 코퍼스 밖 시드는 예외 대신 제외(droppedSeeds) · 남은 시드 0 = 빈 결과 + exhausted ·
랭킹에는 손대지 않는다.

서빙 가능 목록(spec3 §10)도 여기서 붙는다 — `set_catalog` 이 "백엔드에 있는 키"를 받아
**후보에서 뺄 것**을 갱신 때 한 번 계산해 두고, `recommend` 가 그것을 플랫폼 함수의 선택
인자로 넘긴다. 시드는 목록과 무관하다(좋아요한 작품은 목록에 문제가 있어도 유효한 시드다).
자세한 계약은 `aod_serving/engine/catalog.py` 머리말.
"""
from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Hashable, Iterable

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
    #: 랭커에 실제로 넘긴 네이티브 시드 수(파싱 후 중복 제거 · 싫어요 제외) — 0 = 시드 없음.
    #: 라우터가 M6 쿼터에 쓴다(§8-4, 원문 문자열 중복 제거가 아니라 이 값을 믿는다).
    used_seeds: int = 0


@dataclass(frozen=True)
class Catalog:
    """갱신 때 **한 번** 만들어 통째로 갈아 끼우는 불변 목록.

    `restrict` 는 플랫폼 함수에 그대로 넘기는 값이라 플랫폼마다 모양이 다르다:
    Steam·웹소설은 `blocked_rows`(코퍼스 행 순서의 불리언 배열, True = 후보에서 뺀다),
    TMDB 는 `servable_rows`(True = 후보로 남긴다), 웹툰은 제외할 `item_id` 의 frozenset.
    셋 다 읽기 전용이고 만든 뒤에 고치지 않는다 — 그래서 요청 스레드와 갱신 스레드가
    락 없이 같이 봐도 안전하다.
    """
    restrict: object
    size: int                 # 받은 목록의 키 개수(중복 제거 후)
    matched: int              # 그중 코퍼스에 실제로 있는 것


def row_mask(corpus_keys, allowed: set):
    """코퍼스 행 순서의 불리언 배열 — `allowed` 에 든 키의 행만 True.

    **numpy 만 쓴다**(pandas·pyarrow 금지, `catalog.py` 머리말). `corpus_keys` 는 어댑터가
    `load()` 때 계산 스레드에서 떠 둔 numpy 배열이다.
    """
    import numpy as np
    picks = np.fromiter(allowed, dtype=corpus_keys.dtype, count=len(allowed))
    m = np.isin(corpus_keys, picks)
    m.flags.writeable = False                # 불변 — 만든 뒤에는 아무도 못 고친다
    return m


def index_mask(n: int, rows: set):
    """길이 `n` 의 불리언 배열 — `rows`(행 번호) 만 True. TMDB 처럼 키가 곧 행일 때 쓴다."""
    import numpy as np
    m = np.zeros(n, dtype=bool)
    if rows:
        m[np.fromiter(rows, dtype=np.int64, count=len(rows))] = True
    m.flags.writeable = False
    return m


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
    #: 서빙 가능 목록. `None` = 기능 꺼짐(전체 코퍼스). **통째로 대입해서만** 바꾼다.
    _catalog: Catalog | None = None

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
    def _next_page(self, *, k: int, seeds: list, disliked: list, excluded: list, seen: list, media: str | None,
                   restrict=None): ...
    @abstractmethod
    def _items(self, frame) -> list[AdapterItem]: ...

    # ── 서빙 가능 목록 ────────────────────────────────────────────────
    def corpus_keys(self) -> list[str]:
        """코퍼스에 있는 **외부 키** 전부. 도구·테스트가 목록을 만들 때 쓴다."""
        raise NotImplementedError

    def _restrict_from(self, allowed: set) -> object:
        """코퍼스 안 허용 키 집합 → 플랫폼별 제외 구조(`Catalog.restrict`). numpy·파이썬만 쓴다."""
        raise NotImplementedError

    def set_catalog(self, keys: Iterable[str] | None) -> tuple[int, int]:
        """외부 키 목록을 적용하고 `(키 개수, 코퍼스에 있는 개수)` 를 돌려준다. `None` 이면 끈다.

        **pandas·pyarrow 를 부르지 않는다** — 타이머 스레드에서 불릴 수 있기 때문이다
        (`catalog.py` 머리말의 Arrow 스레드 규칙). 쓰는 것은 `parse_key`(순수 파이썬 dict/set
        조회)와 `load()` 가 계산 스레드에서 떠 둔 numpy 배열뿐이다.

        제외 구조를 **다 만든 뒤에 한 번의 대입으로** 갈아 끼운다 — 요청 쪽에서 반쯤 지어진
        상태가 보이지 않는다(대입 자체는 CPython 에서 원자적이고, `recommend` 은 이 속성을
        한 번만 읽는다).
        """
        if keys is None:
            self._catalog = None
            return 0, 0
        raw = dict.fromkeys(keys)                   # 중복 제거(순서 유지) — 개수를 정직하게 세려고
        allowed = {n for k in raw if (n := self.parse_key(k)) is not None}   # 모르는 키는 조용히 버린다
        self._catalog = Catalog(restrict=self._restrict_from(allowed), size=len(raw), matched=len(allowed))
        return len(raw), len(allowed)

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
            return AdapterResult([], dropped, True, used_seeds=0)
        # 목록은 **후보에만** 건다 — 시드는 위에서 이미 확정됐다(목록에 없는 작품도 유효한 시드다).
        # 속성을 여기서 딱 한 번 읽어 둔다: 갱신이 요청 도중에 끼어들어도 한 요청은 한 목록만 본다.
        cat = self._catalog
        frame = self._next_page(k=k, seeds=native, disliked=dis, excluded=self._known_only(excluded),
                                seen=self._known_only(seen), media=media,
                                restrict=None if cat is None else cat.restrict)
        items = self._items(frame)
        return AdapterResult(items, dropped, exhausted=len(items) < k, used_seeds=len(native))


class IntKeyMixin:
    """정수 키 플랫폼(Steam·웹툰·웹소설). `self._known` = 코퍼스에 있는 키 집합."""
    _known: set[int]

    def corpus_keys(self) -> list[str]:
        return sorted(str(k) for k in self._known)

    def parse_key(self, key: str) -> int | None:
        if not (isinstance(key, str) and key.isascii() and key.isdigit()):
            return None
        if len(key) > 19:          # int64 자릿수 상한 — 이보다 길면 코퍼스 키일 수 없다(파싱만으로도 비싸질 수 있다)
            return None
        n = int(key)
        return n if n in self._known else None
