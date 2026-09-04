"""웹툰 판 설정. `webnovel/src/config.py` 와 같은 계약, 술어만 웹툰 것으로 바꿨다."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "wt_v1"
DATA_FILE = PROJECT_ROOT / "data" / "webtoons.jsonl"


def artifact_dir(path: str | Path | None = None) -> Path:
    p = Path(path or os.environ.get("AOD_WT_ARTIFACTS") or ARTIFACTS_DIR)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_artifacts_dir(path: str | Path | None = None) -> Path:
    d = artifact_dir(path); d.mkdir(parents=True, exist_ok=True); return d


# ── 확정값 ────────────────────────────────────────────────────────────────────
# **아직 아무것도 확정되지 않았다.** 아래는 웹소설(가장 가까운 도메인)의 확정값을 그대로
# 가져온 **출발점**이고, 각 축은 웹툰에서 따로 사전등록·판정해야 한다.
# 값 옆에 D-번호가 없다는 것은 "이 도메인에서 측정된 적 없음"을 뜻한다.
PRODUCTION = {
    "strategy": "top2_mean",   # 미측정 — 웹소설에서 이식
    "pop_boost": 0.03,         # 미측정 — 웹소설 D-66 값을 이식
    "hub_lambda": 0.0,         # 미측정 — 웹소설과 같게 끔
    "star_boost": 0.0,         # 미측정 — 웹소설은 평점 신호가 없었다(D-62)
    "tag_w": 0.0,              # 미측정 — 웹툰은 태그가 10~12개라 Steam 형 피복률이 후보다
    "min_favorite": None,      # 미측정
}
POSTPROCESS = {
    "series_max": 1,           # 같은 시리즈 1편
    "artist_max": 2,           # 같은 작가 2편 (웹소설 cap_author 자리)
    "drop_adult": True,        # 성인 웹툰 제외 (수집 단계에서도 401 로 대부분 빠진다)
}
