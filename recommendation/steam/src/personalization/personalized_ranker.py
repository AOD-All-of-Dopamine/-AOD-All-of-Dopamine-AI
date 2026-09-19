from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, artifact_dir

TREND_DIR = PROJECT_ROOT / "artifacts" / "trend_v2"
REVIEW_DIR = PROJECT_ROOT / "artifacts" / "reviews"


def review_dir(artifacts: Path) -> Path:
    """리뷰 파일 폴더. **코퍼스 폴더 안(`<코퍼스>/reviews`)이 먼저** — 코퍼스와 함께 교체·되돌림된다(REC_TAB_DESIGN §8-6).
    없으면 예전 고정 경로(실험 재현용)."""
    d = Path(artifacts) / "reviews"
    return d if d.is_dir() else REVIEW_DIR


def trend_file(artifacts: Path, trend_dir: str | Path | None) -> Path:
    """트렌드 파일. 명시한 `trend_dir` > 코퍼스 폴더 안 > 예전 고정 경로."""
    if trend_dir:
        d = Path(trend_dir)
        return (d if d.is_absolute() else PROJECT_ROOT / d) / "trend_features.parquet"
    inside = Path(artifacts) / "trend_features.parquet"
    return inside if inside.exists() else TREND_DIR / "trend_features.parquet"


def _wilson_lower(pos, n, z: float = 1.96, index=None):
    """긍정 비율의 Wilson score 95% 하한 ∈ [0, 1].

    비율이 나쁘거나 **표본이 적으면** 낮아진다. 리뷰 3개 전원 긍정(1.00)은 0.29 로,
    리뷰 1만개 중 90% 긍정은 0.89 로 내려간다. 양과 질을 한 항으로 합치는 셈이라
    "무명이지만 좋은 것"을 "무명이고 나쁜 것"과 가를 수 있다 — 지금 데이터에 없는 축이다.
    n=0 이면 0.
    """
    import numpy as np
    if index is None and hasattr(pos, "index"):
        index = pos.index          # 인덱스를 잃으면 곱셈에서 정렬이 깨진다
    n = np.asarray(n, dtype=float); pos = np.asarray(pos, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, pos / np.maximum(n, 1), 0.0)
        den = 1.0 + z * z / np.maximum(n, 1)
        cen = p + z * z / (2 * np.maximum(n, 1))
        mrg = z * np.sqrt(np.maximum(p * (1 - p) / np.maximum(n, 1)
                                     + z * z / (4 * np.maximum(n, 1) ** 2), 0.0))
    return pd.Series(np.where(n > 0, np.clip((cen - mrg) / den, 0.0, 1.0), 0.0), index=index)


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

        quality_w   k=50 적합률   P@10     0.8 미만   0.5 미만
        0.00        0.4638       —        51/52      34/52
        0.10        0.6608       0.7981   46/52       3/52
        **0.50**    **0.7692**   0.8923   26/52       2/52   ← 확정 (D-38)

    D-37 은 사전등록 문장을 잘못 써서(제목 "동률 처리" · 본문 "여럿이면 최솟값")
    0.10 을 뽑았다. D-38 에서 **아직 관측하지 않은 통계(P@10)** 로 다시 정했고
    0.50 이 +0.0942 로 문턱 +0.05 를 넘었다. 하락은 최대 2개 항목(=0.20)으로
    "0.20 초과 하락" 조건에 걸리지 않는다.
    ────────────────────────────────────────────────────────────────────────

    **`trend_signal` 은 정규화해서 쓴다.** 원값은 나이 코호트 대비 초과 백분위 × 신선도라
    실측 최대가 0.0915 다. `trend_weight=0.01` 이면 최대 기여가 0.09% 로 유사도 스프레드
    앞에서 무의미하다. [0,1] 로 정규화하면 weight 가 곧 "최대 몇 % 밀어줄지"가 된다.

    기본값 `trend_weight=0.0` — 끄면 기존 실험이 그대로 재현된다.
    """

    def __init__(self, rec_boost: float = 0.03, artifacts: str | Path | None = None,
                 trend_weight: float = 0.0, trend_dir: str | Path | None = None,
                 quality_w: float = 0.0, quality_cap: float = 5.0,
                 quality_src: str = "dataset", tag_w: float = 0.0, mc_w: float = 0.0):
        self.artifacts = artifact_dir(artifacts)
        self.dataset = pd.read_parquet(self.artifacts / "dataset.parquet")
        self.rec_boost = rec_boost
        self.trend_weight = trend_weight
        self.quality_w = quality_w          # D-37. 0 이면 끔 — 기존 실험이 그대로 재현된다
        self.quality_cap = quality_cap
        self.quality_src = quality_src      # D-39. dataset|volume|wilson|blend
        self.dataset = self.dataset.set_index("steam_appid")
        self.trend = self._load_trend(trend_dir) if trend_weight else None
        self._q = self._build_quality() if quality_w else None
        # D-49. 시드 개별 최대 태그 피복률. 0 이면 끔 — 기존 실험이 그대로 재현된다.
        self.tag_w = tag_w
        self._tags = ({a: frozenset(str(t) for t in (v if v is not None else []))
                       for a, v in self.dataset["tags"].items()} if tag_w else None)
        # D-56. **점수가 아니라 보유 여부**가 신호다 — "언론이 다뤘는가"가 격이고,
        # 그 다음 미세 구분은 리뷰 수(quality)가 이미 잡는다.
        # 잔차상관(quality·tag_fit 통제 후) 보유 +0.131 · 점수 +0.060 · 개발사격 +0.019.
        self.mc_w = mc_w
        self._mc = (self.dataset["has_metacritic"].fillna(False).astype(float)
                    if mc_w else None)

        # ── 요청과 무관한 값은 여기서 한 번만 만든다 (서빙 지연, 2026-09-19) ──
        # 예전에는 rank() 가 요청마다 17만 행에 백분위를 다시 매기고 행별 파이썬 람다로
        # 조회했고, tag_fit 은 후보 17만 × 시드 N 번의 파이썬 집합 교집합이었다.
        # 값은 그대로다 — steam/eval/s3_pages.json 과 id·점수가 완전히 같아야 한다.
        self._pct = (self.dataset["recommendations_total"].fillna(0)
                     .rank(pct=True, ascending=True).astype("float64"))
        self._qf = self._q.astype("float64") if self._q is not None else None
        self._pos = None         # appid → 행 번호. 태그 분기에서만 쓴다
        self._tag_rows = None
        if self._tags is not None:
            import numpy as np
            self._pos = pd.Series(range(len(self.dataset)), index=self.dataset.index)
            rows: dict[str, list[int]] = {}
            for i, a in enumerate(self.dataset.index):
                for t in self._tags[a]:
                    rows.setdefault(t, []).append(i)
            self._tag_rows = {t: np.asarray(v, dtype=np.int64) for t, v in rows.items()}
        self._tag_df = None     # 태그 문서 빈도 캐시 — 합의/희귀 태그가 쓴다

    def tag_document_frequency(self) -> tuple[dict[str, int], int]:
        """태그 → 그 태그를 가진 작품 수, 그리고 전체 작품 수.

        `postprocess.rare/consensus_seed_tags` 가 호출마다 17만 행을 세던 값이다.
        같은 함수로 세므로 값이 같다. 두 번 만들어도 같은 값이라 경합에 안전하다.

        태그 열이 없는 코퍼스(rep_v2·s1_v2·full_v1)에서는 빈 빈도를 준다 —
        저쪽 태그 함수들도 태그 열이 없으면 `set()` 로 조용히 넘어가므로 같은 결과다.
        """
        if self._tag_df is None:
            from src.postprocess import tag_document_frequency
            self._tag_df = (tag_document_frequency(self.dataset["tags"])
                            if "tags" in self.dataset.columns
                            else ({}, len(self.dataset)))
        return self._tag_df

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
        if self.quality_src == "dataset":            # 구 경로. 재현용으로 남긴다
            rec = pd.to_numeric(self.dataset["recommendations_total"], errors="coerce").fillna(0.0)
            return (np.log10(1.0 + rec) / self.quality_cap).clip(0.0, 1.0)

        r = self._load_reviews()
        vol = (np.log10(1.0 + r["total_reviews"]) / self.quality_cap).clip(0.0, 1.0)
        if self.quality_src == "volume":
            return vol
        n = r["total_positive"] + r["total_negative"]
        wil = _wilson_lower(r["total_positive"], n)
        if self.quality_src == "wilson":
            return wil
        if self.quality_src == "blend":
            return (vol * (0.5 + 0.5 * wil)).clip(0.0, 1.0)
        raise ValueError(f"quality_src={self.quality_src!r} 를 모른다")

    def _load_reviews(self) -> pd.DataFrame:
        """D-39 크롤 결과. 못 받은 appid 는 0 으로 채운다(미출시작·상장폐지)."""
        d = review_dir(self.artifacts)
        parts = sorted(d.glob("part-*.parquet"))
        if not parts:
            raise SystemExit(f"{d} 가 비었다 — 먼저 `python -m src.crawl_reviews` 를 돌려라")
        rv = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
        rv = rv[rv["total_reviews"] >= 0].drop_duplicates("steam_appid").set_index("steam_appid")
        out = rv.reindex(self.dataset.index).fillna(0.0)
        return out

    def _load_trend(self, trend_dir: str | Path | None) -> pd.Series:
        """appid → 정규화된 trend_signal ∈ [0, 1]."""
        path = trend_file(self.artifacts, trend_dir)
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
        seed_appids=None,
    ) -> pd.DataFrame:
        """`seed_appids` — D-49. `tag_w > 0` 일 때 태그 피복률을 계산하는 데 쓴다.

        꼬리에서 유사도가 평평해(20위 0.4886 → 50위 0.4834) 변별력이 사라지고
        그 자리를 quality 항이 채운다. 그래서 태그가 한 개만 걸친 유명작
        (Iron Harvest RTS·Mechs 리뷰 11,628 g=0)이 올라온다. 태그 정합으로 가른다.
        신호 실측(2,600쌍): 프로필 내부 상관 중앙 +0.390 · 부호일치 **52/52**.
        """
        import numpy as np
        result = candidates.copy()

        result["recommendations_percentile"] = (
            result["steam_appid"].map(self._pct).fillna(0.0).astype("float64"))
        boost = result["recommendations_percentile"] * self.rec_boost

        if self._qf is not None:
            result["quality"] = result["steam_appid"].map(self._qf).fillna(0.0).astype("float64")
            boost = boost + result["quality"] * self.quality_w

        if self.trend is not None:
            result["trend_norm"] = (
                result["steam_appid"].map(self.trend).fillna(0.0).astype("float64"))
            boost = boost + result["trend_norm"] * self.trend_weight

        if self.tag_w and seed_appids and self._tags is not None:
            # 시드 **개별** 최대 피복 |A∩S_i|/|S_i|. 합집합은 시드가 많을수록 느슨해진다.
            # 분모에 후보 A 를 넣지 않는다 — TMDB D-43 에서 자카드가 후보의 추가 태그에
            # 벌점을 줘 품질 높은 것을 밀어냈다.
            #
            # 교집합 크기는 역색인(태그 → 행 번호)으로 센다. 정수 나눗셈이 그대로라
            # 행별 파이썬 집합 교집합과 값이 완전히 같다 (2026-09-19 서빙 지연).
            ss = [self._tags.get(int(a), frozenset()) for a in seed_appids]
            ss = [x for x in ss if x]
            if ss:
                n = len(self.dataset)
                best = np.zeros(n, dtype=np.float64)
                for s in ss:
                    cnt = np.zeros(n, dtype=np.int64)
                    for t in s:
                        cnt[self._tag_rows[t]] += 1
                    np.maximum(best, cnt / len(s), out=best)
                pos = self._pos.reindex(result["steam_appid"]).to_numpy(dtype="float64")
                known = ~np.isnan(pos)
                fit = np.zeros(len(result), dtype=np.float64)
                fit[known] = best[pos[known].astype(np.int64)]
                result["tag_fit"] = fit
                boost = boost + result["tag_fit"] * self.tag_w
        if self.mc_w and self._mc is not None:
            result["has_mc"] = result["steam_appid"].map(self._mc).fillna(0.0)
            boost = boost + result["has_mc"] * self.mc_w

        result["final_score"] = result["seed_similarity"] * (1 + boost)

        if exclude_appids:
            result = result[~result["steam_appid"].isin(exclude_appids)]

        result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        result = result.head(top_n)
        return result
