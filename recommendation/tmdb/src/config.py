"""TMDB 추천기 설정."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "artifacts"

def artifact_dir(name: str | Path | None = None) -> Path:
    if name is None: return ARTIFACTS / "tmdb_v1"
    p = Path(name)
    return p if p.is_absolute() else ARTIFACTS / p


# 확정 설정 — 사전등록 실험으로 확증된 값만 여기에 둔다.
# 값 옆의 D-번호가 그 값을 확정한 실험이다 (recommendation/eval/*_verdict.md).
PRODUCTION = {
    "strategy": "mean",
    "hub_lambda": 0.35,     # D-19
    "rating_boost": 0.15,   # D-31
    "media_w": 0.20,        # D-42. 시드 매체 집합 밖이면 (1−media_w)
    "genre_w": 0.40,        # D-44. 시드 개별 최대 피복률 |A∩S|/|S|
    # align_w 는 **확정값이 아니다.** D-42 이후 변형 문자열에 0.3 이 적혀 있었으나
    # variant_recs 가 넘기지 않아 실제로는 한 번도 켜진 적이 없다 (D-43 verdict 부수 결함).
    "align_w": 0.0,
    # V-2 (2026-09-13, 사전등록 cd9687a9…) **(A) 채택.** 역할 B · 대상군 50 · k=50:
    #   0 → 0.7560 · **0.10 → 0.8120 (+0.056)** · 0.25 → 0.7940 (역U — 많이 밀면 필터가 된다)
    # 롱테일 13 역할 A Δ ±0.000 (감점이 아니라 보너스 보류라 롱테일 시드는 발화하지 않는다).
    # 3역할 다수결도 +0.072 로 같은 방향 — 판정이 잣대 선택에 기대지 않는다.
    "director_w": 0.10,
}

#: 후처리 확정값. `build_components` 가 아니라 `postprocess` 로 가는 것들이라 분리한다.
#: `recommend(postprocess_kwargs=None)` 이면 이것이 쓰인다 — 평가 하네스는 변형별로
#: 명시해 넘기므로 영향받지 않는다.
PRODUCTION_POSTPROCESS = {
    "franchise_max": 1,
    "seed_franchise_max": 0,
    "interleave": True,
    "drop_seed_iter": "any",   # D-46 확정. 시드 제목을 품고 더 뻗은 후보를 뺀다
}
