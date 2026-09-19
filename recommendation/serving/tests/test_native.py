"""네이티브 런타임 굳히기 — Arrow 메모리 풀 (aod_serving/native.py 머리말에 근거)."""
from __future__ import annotations
import re
import sys
from pathlib import Path

from aod_serving.native import NATIVE_ENV, arrow_pool_backend, harden_native_env, pool_warning

SERVING = Path(__file__).resolve().parents[1]


def test_arrow_pool_is_pinned_away_from_mimalloc():
    assert NATIVE_ENV["ARROW_DEFAULT_MEMORY_POOL"] == "system"


def test_harden_sets_missing_and_respects_operator():
    env: dict[str, str] = {}
    assert harden_native_env(env)["ARROW_DEFAULT_MEMORY_POOL"] == "system"
    assert env["ARROW_DEFAULT_MEMORY_POOL"] == "system"

    chosen = {"ARROW_DEFAULT_MEMORY_POOL": "jemalloc"}     # 운영자가 이미 정했으면 덮지 않는다
    assert harden_native_env(chosen)["ARROW_DEFAULT_MEMORY_POOL"] == "jemalloc"


def test_importing_the_package_already_hardened_the_env():
    """어떤 기동 방식이든(엔진·라우터·pytest) `import aod_serving` 만으로 잡혀 있어야 한다."""
    import os
    assert os.environ.get("ARROW_DEFAULT_MEMORY_POOL") in ("system", "jemalloc")


def test_effective_arrow_pool_is_not_mimalloc():
    """실제로 먹었는지 — 환경변수는 libarrow 적재 시점에만 읽히므로 값이 아니라 결과를 본다.

    여기서 깨지면 이 런타임은 세그폴트에 노출돼 있다. pytest 는 conftest 가 pandas(→pyarrow)를
    `import aod_serving` 보다 먼저 올리므로 코드 쪽 설정만으로는 늦는다 — Dockerfile `ENV` 가 받친다.
    즉 이 테스트는 "이미지 ENV 가 살아 있는가" 를 지키는 회귀 테스트이기도 하다.
    """
    import pyarrow  # noqa: F401
    assert arrow_pool_backend() is not None
    assert pool_warning() is None, pool_warning()


def test_backend_probe_does_not_import_pyarrow():
    assert "pyarrow" in sys.modules or arrow_pool_backend() is None


def test_dockerfile_pins_the_pool_too():
    """이미지에도 박아 둔다 — 코드 경로를 타지 않는 기동(예: 다른 엔트리포인트)까지 받친다."""
    text = (SERVING / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^ENV .*ARROW_DEFAULT_MEMORY_POOL=system", text, re.M)
