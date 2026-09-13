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
                 vote_boost: float = 0.0, media_w: float = 0.0,
                 genre_w: float = 0.0, vote_w: float = 0.0, kw_w: float = 0.0):
        d = artifact_dir(artifacts)
        idx = pd.read_parquet(d / "corpus_index.parquet").sort_values("embedding_row")
        ds = pd.read_parquet(d / "dataset.parquet").set_index("item_id")
        self.dataset = ds.loc[idx["item_id"].to_numpy()].reset_index()
        self.dataset["row"] = np.arange(len(self.dataset))
        self.rating_boost = rating_boost
        self.align_w = align_w
        self.vote_boost = vote_boost   # 구설정 재현용. 신규 설계에서는 0 이다
        # D-60. **D-31 의 전제가 TMDB 에서는 틀렸다.** D-31 은 Steam D-6("리뷰 수의 어떤
        # 단조 변환으로도 대작 취향과 저리뷰 취향을 동시에 만족시킬 수 없다")을 TMDB 에서
        # 재보지 않고 이식해 vote_boost 를 지웠다. 실측하니 **저인기 축(17개, 시드 백분위
        # 0.559)에서도 log 투표수 잔차상관 +0.120 (13/17 양수)** 로 방향이 안 뒤집힌다.
        # 대안이던 align_w 의 gap 은 저인기 축에서 +0.040 으로 **기대와 반대**였다.
        # 식은 Steam 확정식과 동일하게 쓴다 — 자유 파라미터를 하나 줄인다.
        self.vote_w = vote_w
        self.media_w = media_w         # D-42. 시드 매체 집합 밖이면 (1−media_w) 를 곱한다
        self._media = self.dataset["media"].to_numpy()
        self.genre_w = genre_w         # D-44. 시드 개별 최대 피복률로 장르 정합을 밀어준다
        self._genres = [frozenset(g.tolist() if hasattr(g, "tolist") else (g or []))
                        for g in self.dataset["genres"]] if genre_w else None
        # X-18. 장르피복과 같은 형태의 키워드 피복. 장르는 중앙 2개라 거칠고
        # 키워드는 중앙 5개라 더 곱다. 장르피복을 통제한 뒤에도 잔차상관 +0.078
        # (프로필 부호일치 42/52)로 독립 정보를 갖는다.
        self.kw_w = kw_w
        self._kw = [frozenset(k.tolist() if hasattr(k, "tolist") else (k or []))
                    for k in self.dataset["keywords"]] if kw_w else None
        vc = self.dataset["vote_count"].astype(float)
        self.vote_pct = vc.rank(pct=True).to_numpy()          # 결측 없음 → 눕힐 것이 없다
        self.vote_q = np.clip(np.log10(1.0 + vc.to_numpy()) / 5.0, 0.0, 1.0)   # D-60
        va = self.dataset["vote_average"].astype(float)
        self.rating_norm = ((va - 5.0) / 5.0).clip(-1, 1).to_numpy()   # 5점=0, 10점=1

    def rank(self, scored: pd.DataFrame, exclude_rows=None, top_n: int = 300,
             servable_mask: np.ndarray | None = None,
             seed_pct: float | None = None, seed_medias=None,
             seed_genres=None, seed_kws=None) -> pd.DataFrame:
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
        if self.vote_w:                          # D-60
            base = base + self.vote_q[r] * self.vote_w
        if self.genre_w and seed_genres:
            # 시드 **개별** 최대 피복률 |A∩S|/|S| (D-44). 합집합을 쓰면 시드가 많을수록
            # 느슨해져 twenty_library(시드 20개 · 장르 15종)에서 무의미해진다.
            # 분모에 후보 A 를 넣지 않는다 — 자카드(D-43)는 후보가 장르를 하나 더
            # 달았다는 이유로 벌점을 줘서 시드가 균질한 프로필(five_horror)의 랭킹을
            # 품질이 아닌 태그 일치도로 무너뜨렸다.
            gf = np.array([max((len(self._genres[i] & sg) / len(sg) if sg else 0.0)
                               for sg in seed_genres)
                           for i in r], dtype=np.float32)
            base = base * (1.0 + self.genre_w * gf)
        if self.kw_w and seed_kws:
            # 장르피복과 동일: 시드 **개별** 최대 피복률, 분모에 후보를 넣지 않는다(자카드 금지).
            kf = np.array([max((len(self._kw[i] & sk) / len(sk) if sk else 0.0)
                               for sk in seed_kws)
                           for i in r], dtype=np.float32)
            base = base * (1.0 + self.kw_w * kf)
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
