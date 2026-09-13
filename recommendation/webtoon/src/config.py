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
# **표현: rep_v2 (제목+장르+태그+줄거리) — T-1 확정 (2026-09-04).**
#   rep_v1 대비 +0.0716 (0.7313 → 0.8030), 프로필 35:10:22, 저관심 대역 +0.090. `artifacts/wt_v1`.
#   태그 정지어 12종(`완결*` 등, text_builder._TAG_STOP*)은 이 판정의 일부다.
# 아래 정렬 값은 **아직 확정되지 않았다.** 웹소설의 확정값을 가져온 출발점이고, 각 축은
# 웹툰에서 따로 사전등록·판정한다. 값 옆에 T-번호가 없다는 것은 "이 도메인에서 측정된 적 없음"이다.
REPRESENTATION = "rep_v2"   # T-1
PRODUCTION = {
    "strategy": "top2_mean",   # **T-4 확정**: mean −0.033(mix −0.067, 중심점이 빈 곳을 가리킴) · max −0.008(넓지만 적합률 못 삼)
    "pop_boost": 0.0,          # **T-2 확정 (0 유지)**: 0.03 −0.012 · 0.10 +0.009, 저관심 −0.030. 배율은 적합률 대신 관심수 중앙만 2.4배로 밀었다
    "hub_lambda": 0.0,         # **T-5 확정**: 0.35 −0.006 · 0.50 +0.002. 허브 0편이라 뺄 것이 없다. 0.50 은 ILS −0.018·고유 +113 (다양성 후보, 잣대 밖)
    "star_boost": 0.0,         # **T-6 확정**: 0.10 +0.019(문턱 0.02 미달, LOO 12/67) · 0.20 +0.018. 정체는 인기 배율(관심수 52.7k→80.5k). 평점 자체 corr(적합) +0.004
    "tag_w": 0.2,              # **T-3 확정**: +0.075 (mix +0.158) · ILS 불변 · 피복 0.69→0.86. 0.4 는 −0.019. Steam D-49 와 달리 태그가 편집된 취향 라벨이라 통한다
    "min_favorite": None,      # 미측정
    "creator_w": 0.0,          # **T-7 보류 (0 유지)**: 0.10 +0.028(문턱 +0.04 미달, 단조·안전 OK, 세 역할 모두 +) · 0.25 +0.019. mix +0.092. 재현 필요
}
POSTPROCESS = {
    "series_max": 1,           # 같은 시리즈 1편
    "artist_max": 2,           # 같은 작가 2편 (웹소설 cap_author 자리)
    "drop_adult": True,        # 성인 웹툰 제외 (수집 단계에서도 401 로 대부분 빠진다)
}
