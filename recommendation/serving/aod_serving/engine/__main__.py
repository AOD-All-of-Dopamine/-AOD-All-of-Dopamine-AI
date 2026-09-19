"""python -m aod_serving.engine — 환경변수로 설정한다(app.py 머리말 참고).

주의: `WORKERS>1` 은 워커 프로세스마다 코퍼스 전체를 따로 적재한다(Steam 이면 워커당 상주 메모리
약 1.9GB) — `mem_limit` 을 워커 수에 맞춰 늘려야 한다. 또 `/health` 는 그중 아무 워커나 붙잡고
답하므로, 그 워커가 아직 적재 중이면 다른 워커가 준비돼 있어도 503 을 돌려줄 수 있다.
"""
# ── 이 두 줄이 맨 위에 있어야 한다 ──────────────────────────────────────────
# `aod_serving` 패키지를 import 하는 순간 `harden_native_env()` 가 돌아 pyarrow 보다 먼저
# `ARROW_DEFAULT_MEMORY_POOL` 을 잡는다(이유: aod_serving/native.py 머리말). 아래 import 들
# 위로 다른 무거운 import 를 올리지 말 것.
import aod_serving  # noqa: F401  — 부작용(네이티브 환경 굳히기)이 목적이다
from aod_serving.native import is_too_late
# ──────────────────────────────────────────────────────────────────────────
import faulthandler, logging, os

import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    faulthandler.enable()          # 네이티브 크래시가 나면 파이썬 스택이라도 남긴다
    if is_too_late():
        logging.getLogger("aod.engine").warning(
            "pyarrow 가 이미 올라온 뒤에 환경을 잡았다 — ARROW_DEFAULT_MEMORY_POOL 이 안 먹었을 수 있다")
    uvicorn.run("aod_serving.engine.app:app_from_env", factory=True, host="0.0.0.0",
                port=int(os.environ.get("PORT", "8000")), workers=int(os.environ.get("WORKERS", "1")))
