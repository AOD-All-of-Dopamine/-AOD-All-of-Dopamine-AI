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
