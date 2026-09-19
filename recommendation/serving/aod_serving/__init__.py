"""AOD 추천 서빙 — 엔진(플랫폼당 1 프로세스)과 라우터. 설계: recommendation/REC_TAB_DESIGN.md §8."""
from aod_serving.native import harden_native_env

# pyarrow 보다 **먼저** 돌아야 한다 — 이유는 native.py 머리말(mimalloc 스레드 힙 SIGSEGV) 참고.
harden_native_env()

__version__ = "0.1.0"
