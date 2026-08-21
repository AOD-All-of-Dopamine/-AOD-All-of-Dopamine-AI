"""TMDB 추천기 설정."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"

def artifact_dir(name: str | Path | None = None) -> Path:
    if name is None: return ARTIFACTS / "tmdb_v1"
    p = Path(name)
    return p if p.is_absolute() else ARTIFACTS / p
