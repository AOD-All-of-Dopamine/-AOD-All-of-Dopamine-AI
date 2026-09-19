"""코퍼스별 설정 덮어쓰기 — `<코퍼스 폴더>/config.json` (REC_TAB_DESIGN §8-6).

코드 기본값 = 확정값(PRODUCTION). config.json 은 허용된 키만 덮어쓴다. "새 코퍼스 + 그 코퍼스에서 판정받은 설정"이
한 폴더에 묶여 함께 교체되고 함께 되돌려진다.
"""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from pathlib import Path

from aod_serving.engine.bootstrap import BASELINE_CORPORA

F, S, B, I = (int, float), (str,), (bool,), (int,)
NONE = type(None)
#: 플랫폼 → {"production": {키: 허용 타입}, "postprocess": {…}}. 어댑터가 함수 인자로 넘길 수 있는 키만 둔다.
ALLOWED: dict[str, dict[str, dict[str, tuple]]] = {
    "steam": {"production": {"strategy": S, "rec_boost": F, "quality_w": F, "quality_cap": F, "quality_src": S,
                             "tag_w": F, "mc_w": F, "trend_weight": F}, "postprocess": {}},
    "tmdb": {"production": {"strategy": S, "hub_lambda": F, "rating_boost": F, "media_w": F, "genre_w": F,
                            "align_w": F, "director_w": F},
             "postprocess": {"franchise_max": I, "seed_franchise_max": I, "interleave": B, "drop_seed_iter": S}},
    "webtoon": {"production": {"strategy": S, "pop_boost": F, "hub_lambda": F, "star_boost": F, "tag_w": F, "creator_w": F,
                               "tag_drop_genre": B, "dislike_w": F, "dislike_floor": F},
                "postprocess": {"series_max": I, "artist_max": I, "drop_adult": B}},
    "webnovel": {"production": {"strategy": S, "pop_boost": F, "min_interest_count": I + (NONE,), "drop_excluded_series": B},
                 "postprocess": {}},
}
_TOP_KEYS = {"production", "postprocess", "verdict", "approved"}


class ConfigError(Exception):
    """config.json 이 잘못됐다. 메시지는 `/health` 의 reason 으로 나간다."""


@dataclass(frozen=True)
class EffectiveConfig:
    production: dict
    postprocess: dict
    hash: str
    approved: bool
    verdict: dict | None


def _check(section: str, given: dict, allowed: dict[str, tuple]) -> None:
    for k, v in given.items():
        if k not in allowed:
            raise ConfigError(f"config.json {section}.{k} 는 허용된 키가 아니다 — 허용: {sorted(allowed)}")
        types = allowed[k]
        ok = isinstance(v, types) and not (isinstance(v, bool) and bool not in types)   # bool 은 int 의 하위형이라 따로 막는다
        if not ok:
            raise ConfigError(f"config.json {section}.{k}={v!r} 의 타입이 맞지 않는다")


def resolve_config(platform: str, d: Path, *, defaults: dict, post_defaults: dict, corpus_version: str,
                   mode: str = "dev") -> EffectiveConfig:
    allowed = ALLOWED[platform]
    body: dict = {}
    p = Path(d) / "config.json"
    if p.exists():
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"config.json 을 읽을 수 없다: {e}") from e
        if not isinstance(body, dict) or set(body) - _TOP_KEYS:
            raise ConfigError(f"config.json 최상위 키는 {sorted(_TOP_KEYS)} 만 — {sorted(set(body) - _TOP_KEYS) if isinstance(body, dict) else body!r}")
    _check("production", body.get("production") or {}, allowed["production"])
    _check("postprocess", body.get("postprocess") or {}, allowed["postprocess"])

    production = {k: defaults[k] for k in allowed["production"] if k in defaults} | (body.get("production") or {})
    postprocess = {k: post_defaults[k] for k in allowed["postprocess"] if k in post_defaults} | (body.get("postprocess") or {})
    approved, verdict = bool(body.get("approved", False)), body.get("verdict")
    if mode == "prod" and corpus_version != BASELINE_CORPORA[platform]:
        if not approved:
            raise ConfigError(f"운영 모드: 새 코퍼스 {corpus_version!r} 에 approved: true 가 없다")
        if not (isinstance(verdict, dict) and verdict.get("id")):
            raise ConfigError(f"운영 모드: 새 코퍼스 {corpus_version!r} 에 verdict(판정 id) 가 없다")
    canon = json.dumps({"production": production, "postprocess": postprocess}, sort_keys=True, ensure_ascii=False)
    return EffectiveConfig(production, postprocess, hashlib.sha256(canon.encode()).hexdigest()[:12], approved, verdict)


def platform_defaults(platform: str) -> tuple[dict, dict]:
    """플랫폼 코드의 확정값. **`enter_platform` 뒤에만** 부를 수 있다."""
    import src.config as c
    post = getattr(c, "PRODUCTION_POSTPROCESS", None) or getattr(c, "POSTPROCESS", None) or {}
    prod = dict(c.PRODUCTION)
    if platform == "steam":
        prod.setdefault("trend_weight", 0.0)       # build_components 기본값. PRODUCTION 에는 없다
    return prod, dict(post)


def effective_config_for(platform: str, d: Path, *, corpus_version: str | None = None, mode: str = "dev") -> EffectiveConfig:
    from aod_serving.engine.bootstrap import enter_platform
    enter_platform(platform, artifacts=Path(d))
    prod, post = platform_defaults(platform)
    return resolve_config(platform, d, defaults=prod, post_defaults=post,
                          corpus_version=corpus_version or Path(d).name, mode=mode)
