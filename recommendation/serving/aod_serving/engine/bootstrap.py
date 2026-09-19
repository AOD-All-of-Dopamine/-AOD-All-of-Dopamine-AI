"""엔진 프로세스를 한 플랫폼 전용으로 만든다.

네 플랫폼이 모두 최상위 패키지 이름으로 `src` 를 쓴다 — 한 프로세스에 둘을 올릴 수 없다.
그래서 엔진은 플랫폼당 프로세스 하나이고, 이 모듈이 그 전제를 코드로 지킨다.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

PLATFORMS = ("steam", "tmdb", "webtoon", "webnovel")
#: 평가에 쓰인 기준 코퍼스. 이 밖의 코퍼스는 운영 모드에서 승인(config.json)이 있어야 서빙한다.
BASELINE_CORPORA = {"steam": "tags_full", "tmdb": "tmdb_v1", "webtoon": "wt_v1", "webnovel": "wn_v6"}
#: 플랫폼 코드가 아티팩트 폴더를 찾는 환경변수. TMDB 는 환경변수가 없어 인자로만 받는다.
_ARTIFACT_ENV = {"steam": "AOD_ARTIFACTS", "webnovel": "AOD_ARTIFACTS", "webtoon": "AOD_WT_ARTIFACTS"}

_entered: str | None = None


def rec_root() -> Path:
    """`recommendation/` 폴더. 이미지 안에서는 AOD_REC_ROOT 로 준다."""
    return Path(os.environ.get("AOD_REC_ROOT") or Path(__file__).resolve().parents[3])


def default_artifacts(platform: str, corpus_version: str | None = None) -> Path:
    base = Path(os.environ.get("ARTIFACTS_ROOT") or rec_root() / platform / "artifacts")
    return base / (corpus_version or BASELINE_CORPORA[platform])


def enter_platform(platform: str, platform_root: Path | None = None, artifacts: Path | None = None) -> Path:
    """chdir + sys.path + 아티팩트 환경변수. 이후 `import src.…` 는 이 플랫폼의 것이다. 아티팩트 경로를 돌려준다."""
    global _entered
    if platform not in PLATFORMS:
        raise ValueError(f"모르는 플랫폼 {platform!r} — {PLATFORMS}")
    if _entered not in (None, platform):
        raise RuntimeError(f"이 프로세스는 이미 {_entered} 전용이다 — {platform} 은 다른 프로세스에서 띄운다")
    root = Path(platform_root or os.environ.get("PLATFORM_ROOT") or rec_root() / platform)
    art = Path(artifacts) if artifacts is not None else default_artifacts(platform)
    os.chdir(root)
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    env = _ARTIFACT_ENV.get(platform)
    if env:
        os.environ[env] = str(art)
    _entered = platform
    return art
