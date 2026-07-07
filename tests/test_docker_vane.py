from pathlib import Path

import yaml

VANE_DIR = Path(__file__).resolve().parent.parent / "docker" / "vane"
COMPOSE = VANE_DIR / "docker-compose.yml"


def test_vane_compose_builds_pinned_ref_single_service_port_3000():
    # §10 #9에서 v1.12.2로 고정: SearXNG 내장 단일 이미지, 설정은 data 볼륨(providers API).
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = data["services"]
    assert "vane" in services
    vane = services["vane"]
    # 조작 이미지 태그가 아니라 고정 상류 ref 로 빌드 (리네임된 Vane.git)
    assert "build" in vane
    ctx = yaml.safe_dump(vane["build"])
    assert "${VANE_REF}" in ctx
    assert "ItzCrazyKns/Vane.git" in ctx
    assert any(str(p) == "3000:3000" for p in vane["ports"])
    # v1.12.2: SearXNG 는 이미지 내장 — 별도 서비스 없음
    assert "searxng" not in services
    # providers 설정(SQLite)이 재기동에도 유지되도록 data 볼륨 필수
    volumes = [str(v) for v in vane.get("volumes", [])]
    assert any("/home/vane/data" in v for v in volumes)
    # 구버전(config.toml 마운트) 잔재 없음
    assert not any("config.toml" in v for v in volumes)
