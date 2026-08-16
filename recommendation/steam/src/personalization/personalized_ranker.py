from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, artifact_dir

TREND_DIR = PROJECT_ROOT / "artifacts" / "trend_v2"


class PersonalizedRanker:
    """개인화 랭킹 — 유사도가 주항이고 인기도·트렌드가 보정항이다.

        final_score = seed_similarity × (1 + rec_percentile × rec_boost
                                           + trend_norm      × trend_weight)

    곱셈 구조라 **유사하지 않은 게임은 아무리 인기 있어도 못 올라온다.** 이게 "개인화가
    가장 중요하고 트렌드도 반영한다"는 설계를 코드로 표현한 형태다.

    **`rec_percentile` 의 유효 폭이 코퍼스에 따라 변한다** — 결측을 0으로 눕히고 전체에
    순위를 매기므로, 리뷰 없는 게임 비율이 커지면 나머지가 좁은 구간으로 압축된다.
    실측: 코퍼스 19,476(리뷰 없음 61%) → 폭 0.387, 173,691(87%) → 폭 0.126 으로 3.1배 줄었다.
    즉 코퍼스를 키우는 것만으로 인기도 보정이 약해진다. 이 성질을 알고 rec_boost 를 봐야 한다.

    ────────────────────────────────────────────────────────────────────────
    2026-08-15 — **위 성질은 "약해진다"가 아니라 결함이다. 아직 안 고쳤다.**

    `fillna(0).rank(pct=True)` 는 리뷰 미보고 151,799개(코퍼스의 87%)를 전부 동점으로
    눕히고 pandas 가 동점에 **평균 순위**를 준다. 그 결과 실측값이 이렇다:

        미보고(리뷰 <100)   pct 0.437   ← 코퍼스 최하위인데 중간 점수를 받는다
        리뷰 101개          pct 0.879
        리뷰 9,264개        pct 0.991
        리뷰 520만개        pct 1.000

    **리뷰 101개부터 520만개까지가 0.879~1.000 안에 다 들어간다.** rec_boost=0.15 를
    곱하면 final_score 차이가 0.018 이다. 즉 인기도 보정은 사실상 "미보고냐 아니냐"만
    보는 이진 플래그이고, 보고된 게임끼리는 구분하지 못한다. 실패 축의 적합/부적합이
    정확히 그 구간에 있다(six_action_adv 적합 중앙 24,384 · 부적합 중앙 600).

    고치면(미보고를 순위에서 빼고 fillna(0)) 미보고 0.000 · 101개 0.037 ·
    9,264개 0.931 · 520만개 1.000 으로 해상도가 생긴다.

    **그런데 rec_boost=0.15 는 이 결함 위에서 맞춰진 값이라 같이 옮겨야 한다.**
    52프로필 · k=50 · 미판정 0 실측:

        설정                     평균      미달   비고
        현행(결함 유지)          0.8488     12
        수정 + boost 0.15        0.8238*    -     coh_mmo 0.80→0.64, niche_soulslike_solo 0.84→0.76
        수정 + boost 0.08        0.8573     13    18개 개선 / 8개 악화

    boost 0.08 은 평균을 올리지만(+0.0085) 미달은 12→13 이다. 개선이 넓고 얕은 대신
    (seven_horror_coop +0.14 · six_action_adv +0.10 · two_bigaction +0.10 · ten_cozy +0.08)
    손실이 좁고 깊다(two_cozy_puzzle -0.18 · ten_jrpg -0.12).

    **그래서 결함만 기록하고 아직 안 바꿨다.** 이 위에 얹혀 맞춰진 값이 셋 더 있다 —
    `REFRESH_REC_BOOST` · `REFRESH_SEED_SCALED_FLOOR` · `REFRESH_MIN_REVIEWS`.
    고치려면 그 셋을 함께 다시 재야 하고, 그건 한 번에 하나씩 옮기는 지금 방식으로는
    안 된다. 부분만 고치면 지금처럼 이기고 지는 것을 맞바꾸게 된다.
    ────────────────────────────────────────────────────────────────────────

    **`trend_signal` 은 정규화해서 쓴다.** 원값은 나이 코호트 대비 초과 백분위 × 신선도라
    실측 최대가 0.0915 다. `trend_weight=0.01` 이면 최대 기여가 0.09% 로 유사도 스프레드
    앞에서 무의미하다. [0,1] 로 정규화하면 weight 가 곧 "최대 몇 % 밀어줄지"가 된다.

    기본값 `trend_weight=0.0` — 끄면 기존 실험이 그대로 재현된다.
    """

    def __init__(self, rec_boost: float = 0.03, artifacts: str | Path | None = None,
                 trend_weight: float = 0.0, trend_dir: str | Path | None = None):
        self.artifacts = artifact_dir(artifacts)
        self.dataset = pd.read_parquet(self.artifacts / "dataset.parquet")
        self.rec_boost = rec_boost
        self.trend_weight = trend_weight
        self.dataset = self.dataset.set_index("steam_appid")
        self.trend = self._load_trend(trend_dir) if trend_weight else None

    @staticmethod
    def _load_trend(trend_dir: str | Path | None) -> pd.Series:
        """appid → 정규화된 trend_signal ∈ [0, 1]."""
        d = Path(trend_dir) if trend_dir else TREND_DIR
        if not d.is_absolute():
            d = PROJECT_ROOT / d
        path = d / "trend_features.parquet"
        if not path.exists():
            raise SystemExit(
                f"{path} 없음 — trend_weight 를 쓰려면 먼저 실행하세요:\n"
                "  python -m src.trend.trend_features --dataset artifacts/full_v1/dataset.parquet "
                "--out artifacts/trend_v2"
            )
        t = pd.read_parquet(path).set_index("steam_appid")["trend_signal"].astype(float)
        peak = float(t.max())
        return t / peak if peak > 0 else t

    def rank(
        self,
        candidates: pd.DataFrame,
        exclude_appids: set[int] | None = None,
        top_n: int = 300,
    ) -> pd.DataFrame:
        result = candidates.copy()

        rec_col = self.dataset["recommendations_total"].fillna(0)
        pct = rec_col.rank(pct=True, ascending=True)

        result["recommendations_percentile"] = result["steam_appid"].map(
            lambda x: pct.get(x, 0.0)
        )
        boost = result["recommendations_percentile"] * self.rec_boost

        if self.trend is not None:
            result["trend_norm"] = result["steam_appid"].map(lambda x: float(self.trend.get(x, 0.0)))
            boost = boost + result["trend_norm"] * self.trend_weight

        result["final_score"] = result["seed_similarity"] * (1 + boost)

        if exclude_appids:
            result = result[~result["steam_appid"].isin(exclude_appids)]

        result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        result = result.head(top_n)
        return result
