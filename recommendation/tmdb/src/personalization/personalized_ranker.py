"""개인화 랭킹 — 유사도가 주항, 평점과 **인기 정합**이 보정항.

    final = sim × (1 + rating_norm×rating_boost) × (1 − align_w×|vote_pct − seed_pct|)

**`vote_boost` 는 제거했다 (D-31).** 전역 상수 boost 는 모든 프로필을 같은
방향으로 민다. 대작 취향에는 맞지만 롱테일 취향에는 정확히 반대다.
Steam D-6 이 이미 증명했다 — "리뷰 수의 어떤 단조 변환으로도 대작 취향과
저리뷰 취향을 동시에 만족시킬 수 없다."

대신 **프로필 의존 정합 항**을 쓴다. 시드 인기 백분위 중앙 `seed_pct` 를
목표로 삼고 후보가 거기서 멀수록 감점한다. 시드가 대작이면 대작을,
롱테일이면 롱테일을 뽑는다. 방향이 프로필마다 달라진다.

측정 근거(시드 대비 top-50 인기 백분위 편차 중앙):
    일반군 36개  시드 0.992 → 구설정 -0.06   ← 구설정이 옳았던 구간
    저인기군 16개 시드 0.649 → 구설정 +0.21
    롱테일 5개   시드 0.07~0.37 → 구설정 +0.52 (레버 OFF 에서도 +0.38)
검색 자체가 저인기 시드에서 인기작을 끌어오고, boost 가 악화시켰다.

곱셈이므로 **유사하지 않은 작품은 아무리 인기 있어도 못 올라온다.**

────────────────────────────────────────────────────────────────────────────
**Steam 의 인기도 백분위 결함이 TMDB 에는 없다.**

Steam: `recommendations_total` 이 87% 결측 → `fillna(0).rank(pct=True)` 가
미보고 151,799건을 동점으로 눕히고 pandas 가 평균 순위를 준다. 결과가
  미보고 0.437 · 리뷰 101개 0.879 · 520만개 1.000 — **리뷰 101~520만이 폭 0.12 안에.**
사실상 "미보고냐 아니냐" 이진 플래그였고, `rec_boost` 셋이 그 결함 위에서 맞춰졌다.

TMDB: `vote_count` **비결측 100% · 0인 것 0건.** 분위가
  p10 35 · p25 46 · p50 87 · p75 248 · p90 892 · p99 7,792 · 최대 40,789.
⇒ 백분위가 **전 구간에서 해상도를 갖는다.** Steam 이 못 한 것을 여기서는 할 수 있다.

`vote_average` 도 100% 비결측(중앙 6.5)이라 품질 신호를 따로 쓸 수 있다.
Steam 의 metacritic 은 2.4% 밖에 없어서 못 썼다.
────────────────────────────────────────────────────────────────────────────

보정 계수는 **측정 전까지 0.0** 이다. 근거 없이 켜지 않는다.
"""
from pathlib import Path
import numpy as np, pandas as pd
from src.config import artifact_dir


class PersonalizedRanker:
    def __init__(self, artifacts=None, rating_boost: float = 0.0, align_w: float = 0.0,
                 vote_boost: float = 0.0, media_w: float = 0.0):
        d = artifact_dir(artifacts)
        idx = pd.read_parquet(d / "corpus_index.parquet").sort_values("embedding_row")
        ds = pd.read_parquet(d / "dataset.parquet").set_index("item_id")
        self.dataset = ds.loc[idx["item_id"].to_numpy()].reset_index()
        self.dataset["row"] = np.arange(len(self.dataset))
        self.rating_boost = rating_boost
        self.align_w = align_w
        self.vote_boost = vote_boost   # 구설정 재현용. 신규 설계에서는 0 이다
        self.media_w = media_w         # D-42. 시드 매체 집합 밖이면 (1−media_w) 를 곱한다
        self._media = self.dataset["media"].to_numpy()
        vc = self.dataset["vote_count"].astype(float)
        self.vote_pct = vc.rank(pct=True).to_numpy()          # 결측 없음 → 눕힐 것이 없다
        va = self.dataset["vote_average"].astype(float)
        self.rating_norm = ((va - 5.0) / 5.0).clip(-1, 1).to_numpy()   # 5점=0, 10점=1

    def rank(self, scored: pd.DataFrame, exclude_rows=None, top_n: int = 300,
             servable_mask: np.ndarray | None = None,
             seed_pct: float | None = None, seed_medias=None) -> pd.DataFrame:
        """`seed_pct` — 시드 인기 백분위 중앙. 없으면 정합 항을 끈다.

        `seed_medias` — 시드의 media 집합 (D-42). `media_w > 0` 이고 후보 media 가
        이 집합에 없으면 `(1 − media_w)` 를 곱한다. 혼합 시드({movie, tv})면 페널티가
        자연히 0 이다. 진단(2,600쌍): 영화 시드에 TV 적합률 0.764 vs 영화 0.877,
        TV 시드에 영화 0.726 vs TV 0.920 — 양방향으로 나쁘다. coh_pixar(top-50 의
        56%가 TV, 그 적합률 0.32)가 최악 표현형. 하드 필터는 매체를 넘나드는 취향
        (mix2_anime_film 0.82 · coh_sageuk 1.00)을 해쳐서 소프트로 간다.
        """
        df = scored.copy()
        r = df["row"].to_numpy()
        base = 1.0 + self.rating_norm[r] * self.rating_boost
        if self.vote_boost:                      # 구설정 재현 경로
            base = base + self.vote_pct[r] * self.vote_boost
        if self.media_w and seed_medias:
            mis = ~np.isin(self._media[r], list(seed_medias))
            base = base * (1.0 - self.media_w * mis)
        if self.align_w and seed_pct is not None:
            # 거리 페널티. 계수가 1 을 넘지 않도록 잘라 음수 점수를 막는다.
            gap = np.abs(self.vote_pct[r] - float(seed_pct))
            base = base * np.clip(1.0 - self.align_w * gap, 0.05, None)
        df["final_score"] = df["seed_similarity"].to_numpy() * base
        if servable_mask is not None:
            df = df[servable_mask[df["row"].to_numpy()]]
        if exclude_rows:
            df = df[~df["row"].isin(set(int(x) for x in exclude_rows))]
        df = df.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df
