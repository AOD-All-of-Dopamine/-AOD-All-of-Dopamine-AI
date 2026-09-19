"""전체 탭 혼합 — 평가된 `crossdomain/mix.py` 의 M6 을 **그대로** 불러 쓴다(재구현하면 평가와 어긋난다).

mix.py 는 순수 파이썬이다(numpy 없음). 라우터 이미지에는 이 파일 하나만 복사한다.
"""
from __future__ import annotations
import importlib.util, os
from functools import lru_cache
from pathlib import Path

from aod_serving.common.models import EngineItem

#: 전체 탭에 들어가는 플랫폼(평가된 그대로 — 웹툰 포함은 사전등록 평가 후, REC_TAB_DESIGN §8-4)
ALL_TAB_PLATFORMS = ("steam", "tmdb", "webnovel")
_TO_MIX = {"webnovel": "wn"}
_FROM_MIX = {v: k for k, v in _TO_MIX.items()}


def _mix_path() -> Path:
    root = Path(os.environ.get("AOD_REC_ROOT") or Path(__file__).resolve().parents[3])
    return Path(os.environ.get("MIX_PATH") or root / "crossdomain" / "mix.py")


@lru_cache(maxsize=1)
def load_m6():
    spec = importlib.util.spec_from_file_location("aod_crossdomain_mix", _mix_path())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.M6


def mix_all(lists: dict[str, list[EngineItem]], n_seeds: dict[str, int], k: int) -> list[tuple[str, EngineItem]]:
    """플랫폼별 목록 → M6 순서의 (플랫폼, 항목). `n_seeds` 에 없는 플랫폼은 섞지 않는다."""
    seeds = {_TO_MIX.get(p, p): n for p, n in n_seeds.items() if n > 0}
    if not seeds:
        return []
    by_key = {(p, i.key): i for p, its in lists.items() for i in its}
    keys = {_TO_MIX.get(p, p): [i.key for i in its] for p, its in lists.items() if _TO_MIX.get(p, p) in seeds}
    episodes = {i.key: (i.episode_count or 0) for i in lists.get("webnovel", [])}
    mixed = load_m6()(keys, seeds, None, k=k, episodes=episodes)
    return [(_FROM_MIX.get(p, p), by_key[(_FROM_MIX.get(p, p), key)]) for p, key, _rank in mixed]
