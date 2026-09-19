"""네이티브 런타임 굳히기 — **pyarrow 가 import 되기 전에** 돌아야 한다.

그래서 `aod_serving/__init__.py` 에서 부른다. 패키지 안의 무엇을 import 하든(엔진 · 라우터 ·
`uvicorn --factory` · pytest) 그 전에 한 번 돌고, 운영자가 기억해야 할 것은 없다.
컨테이너는 Dockerfile `ENV` 로 한 겹 더 받쳐 둔다.

왜 `ARROW_DEFAULT_MEMORY_POOL=system` 인가
------------------------------------------
pyarrow 25 의 Arrow 기본 메모리 풀은 libarrow 에 정적으로 박힌 **mimalloc** 이다. mimalloc 은
스레드마다 힙을 두고 TLS 로 찾아가는데, Arrow 풀을 쓰던 스레드가 **종료**하면 glibc 가 그
스택·TLS 블록을 다음 스레드에 그대로 재활용한다. 그때 새 스레드의 첫 Arrow 할당이
`mi_thread_init()` 에서 남은 TLS 를 믿고 죽은 힙을 따라가 SIGSEGV 를 낸다.

    #0  mi_thread_init ()                     ← pyarrow/libarrow.so.2500
    #1  _mi_malloc_generic ()
    #2  mi_theap_malloc_zero_aligned_at_generic ()
    #3  arrow::BaseMemoryPoolImpl<MimallocAllocator>::Allocate(long, long, unsigned char**)
    …   arrow::py::ConvertPySequence → pandas ArrowStringArray._from_sequence → pd.DataFrame(...)

`system`(= glibc malloc)에는 스레드 지역 힙이 없어 그 전제 자체가 사라진다. 할당자만 바뀔 뿐
연산은 그대로라 **수치 결과가 달라지지 않는다**(dtype·pandas 옵션은 건드리지 않는다).
실측 지연도 차이가 없다 — 웹툰 60회 연속 추천: mimalloc 9.2s · system 9.1~9.4s · jemalloc 9.5~9.8s.

재현은 `scripts/repro_segv.py`(소켓 없이 빠르게), 회귀 확인은 `scripts/stress_engine.sh`(실제 HTTP).
둘째 방어선은 `engine/app.py` 의 전용 계산 스레드다 — 그쪽이 "스레드가 죽는다" 는 전제를 없앤다.

**스레드 개수 환경변수(OMP_NUM_THREADS 등)는 일부러 건드리지 않는다.** BLAS 스레드 수가 바뀌면
누적 순서가 바뀌어 부동소수점 결과가 달라질 수 있고, 그러면 평가 서버와의 동일성이 깨진다.
"""
from __future__ import annotations

import os
import sys

#: 이미 설정돼 있으면 존중한다 — 재현 하네스가 일부러 mimalloc 을 되돌려 보기 때문이다.
NATIVE_ENV = {"ARROW_DEFAULT_MEMORY_POOL": "system"}

#: 스레드 지역 힙을 쓰는 풀. 위 SIGSEGV 의 전제를 만든다.
UNSAFE_POOLS = ("mimalloc",)


def harden_native_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """`NATIVE_ENV` 를 적용하고, 적용 뒤의 실제 값을 돌려준다."""
    target = os.environ if env is None else env
    for key, value in NATIVE_ENV.items():
        target.setdefault(key, value)
    return {key: target[key] for key in NATIVE_ENV}


def is_too_late() -> bool:
    """pyarrow 가 이미 올라온 뒤라면 환경변수는 효과가 없다 — 호출 순서가 틀렸다는 신호."""
    return "pyarrow.lib" in sys.modules


def arrow_pool_backend() -> str | None:
    """지금 쓰고 있는 Arrow 메모리 풀 이름. pyarrow 가 아직 없으면 None (일부러 import 하지 않는다)."""
    pa = sys.modules.get("pyarrow")
    if pa is None:
        return None
    try:
        return pa.default_memory_pool().backend_name
    except Exception:                                    # noqa: BLE001 — 진단용이라 절대 실패시키지 않는다
        return None


def pool_warning() -> str | None:
    """위험한 풀로 돌고 있으면 사람이 읽을 경고 문구, 아니면 None.

    환경변수는 libarrow 가 **적재되는 순간**에만 읽힌다. 그래서 `import aod_serving` 보다 먼저
    pandas(→pyarrow)를 올리는 경로가 하나라도 있으면 코드 쪽 설정은 헛돈다 — 실제로 pytest 는
    conftest 가 pandas 를 먼저 올린다. 그 경우를 잡아내려고 값이 아니라 **실제 backend** 를 본다.
    컨테이너에서는 Dockerfile `ENV` 가 파이썬보다 먼저라 항상 먹는다.
    """
    backend = arrow_pool_backend()
    if backend in UNSAFE_POOLS:
        return (f"Arrow 메모리 풀이 {backend} 다 — Arrow 를 쓰던 스레드가 죽고 그 TLS 블록을 재활용한 "
                f"새 스레드가 Arrow 를 할당하면 mi_thread_init() 에서 SIGSEGV 가 난다. "
                f"ARROW_DEFAULT_MEMORY_POOL=system 을 pyarrow import 전에(=Dockerfile ENV 로) 잡아라")
    return None
