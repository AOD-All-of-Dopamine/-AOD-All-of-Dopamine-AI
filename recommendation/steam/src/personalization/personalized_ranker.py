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

    ────────────────────────────────────────────────────────────────────────
    2026-08-22 — **위 결함은 D-37 에서 고쳤다. 백분위를 손대는 대신 축을 하나 더 놓았다.**

    여기 적혀 있던 비교표("현행 0.8488 · 수정+boost 0.15 → 0.8238 · boost 0.08 → 0.8573")는
    **폐기한다.** 그 수치들은 기준이 관대한 구 등급 은행에서 나왔고, 눈가림 재채점 결과
    구 기준과 현 기준의 적합률 차가 0.375 였다(D-24). 비교 자체가 성립하지 않는다.

    `rec_percentile` 은 건드리지 않았다 — 한 번에 하나만 옮기기 위해서다. 대신
    `quality_w × clip(log10(1+리뷰수)/5, 0, 1)` 을 보정항에 더했다. 로그는 동점 붕괴가
    없어 백분위의 결함을 우회한다. `rec_boost 0.03` · `REFRESH_SEED_SCALED_FLOOR` ·
    `REFRESH_MIN_REVIEWS` 는 그대로 두었으므로 함께 다시 잴 필요가 없었다.

    단일 기준 은행 · 52프로필 × k=50 · 미채점 0 실측:

        quality_w   적합률    0.8 미만   0.5 미만
        0.00        0.4638    51/52      34/52
        **0.10**    0.6608    46/52       3/52     ← 확정 (사전등록 규칙)
        0.50        0.7692    26/52       2/52     ← 더 좋지만 미확증, D-38

    51개 프로필이 오르고 1개만 내렸다(`lowrev_deckbuilder` −0.08, 사전등록 허용치 이내).
    ────────────────────────────────────────────────────────────────────────

    **`trend_signal` 은 정규화해서 쓴다.** 원값은 나이 코호트 대비 초과 백분위 × 신선도라
    실측 최대가 0.0915 다. `trend_weight=0.01` 이면 최대 기여가 0.09% 로 유사도 스프레드
    앞에서 무의미하다. [0,1] 로 정규화하면 weight 가 곧 "최대 몇 % 밀어줄지"가 된다.

    기본값 `trend_weight=0.0` — 끄면 기존 실험이 그대로 재현된다.
    """

    def __init__(self, rec_boost: float = 0.03, artifacts: str | Path | None = None,
                 trend_weight: float = 0.0, trend_dir: str | Path | None = None,
                 quality_w: float = 0.0, quality_cap: float = 5.0):
        self.artifacts = artifact_dir(artifacts)
        self.dataset = pd.read_parquet(self.artifacts / "dataset.parquet")
        self.rec_boost = rec_boost
        self.trend_weight = trend_weight
        self.quality_w = quality_w          # D-37. 0 이면 끔 — 기존 실험이 그대로 재현된다
        self.quality_cap = quality_cap
        self.dataset = self.dataset.set_index("steam_appid")
        self.trend = self._load_trend(trend_dir) if trend_weight else None
        self._q = self._build_quality() if quality_w else None

    def _build_quality(self) -> pd.Series:
        """appid → 품질 사전분포 ∈ [0, 1]. `log10(1+리뷰수) / cap` 을 자른 값이다.

        백분위를 쓰지 않는 이유는 위 docstring 의 결함 때문이다 — 코퍼스의 87.4%가
        리뷰 0 이라 동점 평균순위가 최하위에 0.437 을 준다. 로그는 동점 붕괴가 없고
        단조라 리뷰 0 → 0.00, 100개 → 0.40, 1만개 → 0.80, 10만개 → 1.00 이 된다.

        **이건 "인기 있는 걸 밀어준다"가 아니다.** 리뷰 0 개인 Steam 항목은 대체로
        미출시·방치·양산형이라 애초에 추천으로 성립하지 않는다. 실측(D-37, 2,600쌍):
        리뷰 0 구간 적합률 0.271 · 1천~1만 0.815 · 1만+ 0.922 이고, 프로필 내부
        상관은 52개 중 50개가 양수(중앙 +0.543)다. 저인기 취향 프로필도 마찬가지다.
        """
        import numpy as np
        rec = pd.to_numeric(self.dataset["recommendations_total"], errors="coerce").fillna(0.0)
        return (np.log10(1.0 + rec) / self.quality_cap).clip(0.0, 1.0)

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

        if self._q is not None:
            result["quality"] = result["steam_appid"].map(lambda x: float(self._q.get(x, 0.0)))
            boost = boost + result["quality"] * self.quality_w

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
