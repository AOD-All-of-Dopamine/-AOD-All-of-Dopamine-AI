"""테스트는 확정 코퍼스(wn_v6)에서 돈다.

`src/config.ARTIFACTS_DIR` 기본값도 wn_v6 이지만, 셸에 다른 `AOD_ARTIFACTS` 가 남아 있으면
테스트 시드(예: 14499247)가 없는 코퍼스로 돌아 `KeyError` 로 깨진다. 여기서 한 번 더 고정한다.
일부러 다른 코퍼스로 돌리려면 실행 전에 환경변수를 직접 설정하면 된다(setdefault 라 존중된다).
"""
import os
from pathlib import Path

os.environ.setdefault("AOD_ARTIFACTS", str(Path(__file__).resolve().parents[1] / "artifacts" / "wn_v6"))
