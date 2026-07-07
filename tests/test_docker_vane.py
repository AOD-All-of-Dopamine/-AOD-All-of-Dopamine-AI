from pathlib import Path

import yaml

VANE_DIR = Path(__file__).resolve().parent.parent / "docker" / "vane"
COMPOSE = VANE_DIR / "docker-compose.yml"
SEARXNG_SETTINGS = VANE_DIR / "searxng" / "settings.yml"


def test_vane_compose_declares_services_and_maps_port_3000():
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    services = data["services"]
    assert "vane" in services
    assert "searxng" in services
    vane = services["vane"]
    # 조작 이미지 태그가 아니라 고정 상류 ref 로 빌드
    assert "build" in vane
    assert "${VANE_REF}" in yaml.safe_dump(vane["build"])
    assert any(str(p) == "3000:3000" for p in vane["ports"])
    # SearXNG JSON 은 env 플래그가 아니라 마운트된 settings.yml 로 주입
    mounts = [str(v) for v in services["searxng"].get("volumes", [])]
    assert any("./searxng:/etc/searxng" in m for m in mounts)


def test_searxng_settings_yaml_enables_json_format():
    settings = yaml.safe_load(SEARXNG_SETTINGS.read_text(encoding="utf-8"))
    assert "json" in settings["search"]["formats"]  # spec §4.1
