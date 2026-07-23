from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "s1_v2"


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "s1_v2.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
