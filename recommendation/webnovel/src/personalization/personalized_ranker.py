from pathlib import Path

import pandas as pd

from src.config import artifact_dir


class PersonalizedRanker:
    def __init__(self, pop_boost: float = 0.03, artifacts: str | Path | None = None):
        self.artifacts = artifact_dir(artifacts)
        self.dataset = pd.read_parquet(self.artifacts / "dataset.parquet")
        self.pop_boost = pop_boost
        self.dataset = self.dataset.set_index("item_id")

    def rank(
        self,
        candidates: pd.DataFrame,
        exclude_ids: set[int] | None = None,
        top_n: int = 300,
    ) -> pd.DataFrame:
        result = candidates.copy()

        # **결측을 0으로 눕히지 않는다.** `rank()` 는 NaN 을 건너뛰므로 관측된 값끼리만
        # 순위가 매겨지고, 관측이 없는 작품은 아래 `fillna(0.0)` 으로 바닥에 놓인다.
        #
        # 왜 고치는가 — Steam 쪽에서 `fillna(0).rank(pct=True)` 가 결함으로 확인됐다.
        # 코퍼스 173,691건 중 리뷰 미보고가 87%라 전부 한 동점 그룹이 되고, pandas 가
        # 동점에 평균 순위를 줘서 **미보고가 pct 0.437**(최하위인데 중간 점수)을 받았다.
        # 남은 13%는 0.879~1.000 으로 압축돼 리뷰 101개와 520만개가 폭 0.12 안에 들어갔다.
        # 인기도 보정이 사실상 "미보고냐 아니냐" 이진 플래그로만 동작한 것이다.
        #
        # 웹소설은 지금 결측이 0건(값 0 이 378건 중 103건)이라 이 코드가 무동작이었다.
        # 실제 0 이 하위 27%에 동점으로 모여 0.138 을 받는 것 자체는 정상이다 — 정말
        # 바닥이니까. 지금 고치는 이유는 **결측과 0 을 구분해 두는 것**이다: 관심 수 크롤
        # 셀렉터가 이미 한 번 깨졌던 이력이 있어(README 의 백엔드 크롤러 문제) 결측이
        # 생기면 그 순간 실제 0 과 조용히 합쳐진다. 코퍼스가 파일럿 378건에서 목표
        # 5.8만으로 커지기 전, 이 위에 아무 상수도 튜닝되지 않은 지금이 가장 싸다.
        #
        # Steam 은 반대로 **고치지 않고 두었다.** 그쪽은 `rec_boost` 등 세 상수가 이 결함
        # 위에서 맞춰져 있어 고치면 저리뷰 취향이 깨진다(ten_jrpg 0.80→0.68 ·
        # coh_mmo 0.80→0.64). boost 를 0.08~0.15 로 스윕해도 전부 그렇다 — 리뷰 수의
        # 어떤 단조 변환으로도 대작 취향과 저리뷰 취향을 동시에 만족시킬 수 없다.
        # 이 도메인에서 `pop_boost` 를 맞출 때 그 함정을 처음부터 피해야 한다.
        pct = self.dataset["interest_count"].rank(pct=True, ascending=True)

        result["interest_percentile"] = (
            result["item_id"].map(pct).astype("float64").fillna(0.0)
        )
        result["final_score"] = result["seed_similarity"] * (
            1 + result["interest_percentile"] * self.pop_boost
        )

        if exclude_ids:
            result = result[~result["item_id"].isin(exclude_ids)]

        result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        result = result.head(top_n)
        return result
