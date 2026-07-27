# src/config.py
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "s1_v2"

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "s1_v2.yaml"
ENV_FILE = PROJECT_ROOT / ".env"


def _load_env_file() -> None:
    """`.env` 를 환경변수로 올린다(이미 설정된 값은 덮어쓰지 않는다)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def resolve_path(raw: str | Path) -> Path:
    """`${AOD_BACK_ROOT}/steam_games.jsonl` 형태를 실제 경로로 푼다.

    예전에는 설정과 소스에 `/home/jiho/projects/...` 가 하드코딩돼 있어 다른 머신에서
    파이프라인이 전혀 돌지 않았다. 미설정 변수는 조용히 빈 문자열이 되지 않고 실패한다.
    """
    _load_env_file()
    text = str(raw)
    expanded = os.path.expandvars(text)
    if "$" in expanded:
        missing = {
            token.strip("${}")
            for token in expanded.replace("}", "} ").split()
            if token.startswith("$")
        }
        raise SystemExit(
            f"경로에 미설정 환경변수가 있습니다: {sorted(missing)}\n"
            f"  원본: {text}\n"
            f"  {ENV_FILE} 또는 셸 환경에 설정하세요 (.env.example 참고)."
        )
    return Path(expanded).expanduser()


def load_config(path: str | Path | None = None) -> dict:
    """설정 로드 우선순위: 인자 > AOD_CONFIG 환경변수 > configs/s1_v2.yaml."""
    _load_env_file()
    cfg_path = Path(path or os.environ.get("AOD_CONFIG") or DEFAULT_CONFIG)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        raise SystemExit(f"설정 파일이 없습니다: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
