"""시드×후보 유사도 행렬을 후보당 점수 하나로 접는다.

Steam 이 이미 기각한 것 — **다시 하지 말 것**:
  · "공통 축을 강제하는" 집계 (top2_mean − α·(top1−top2), α=0.5 는 min).
    밀집 벡터의 교집합은 장르의 교집합이 아니다. 두 시드의 **부수적** 공통점을
    정확히 집어낸다. Steam two_rhythm_arcade 에서 0.50 → 0.15 로 무너졌다.
  · 크로스도메인 h66·h67 이 코퍼스 전체 min 검색으로 재확인했다:
    전문가(한 시드에 3점·다른 시드에 0점)를 24.1% → 1.7% 로 제거하는 데는 성공하지만,
    그 자리를 채우는 것은 "모두에게 좋은"이 아니라 **"모두에게 무난한"**(13.0% → 43.3%)이다.

`dominant_seed` 는 점수와 무관하게 항상 채운다 — 후처리의 시드 인터리빙이 이걸로 묶는다.
"""
import numpy as np, pandas as pd


class ScoreAggregator:
    def aggregate_all(self, sim_matrix, seed_embeddings, corpus_df, strategies=None):
        if strategies is None: strategies = ["max", "mean", "top2_mean"]
        seed_rows = list(seed_embeddings.keys())
        n = sim_matrix.shape[0]
        # argmax 결과를 **먼저 파이썬 정수로** 바꾼 뒤 훑는다 — 59,780번의 np.int64 박싱과
        # `__index__` 왕복이 사라진다 (2026-09-19 서빙 지연). 같은 리스트를 만들 뿐이라
        # 값도 dtype 도 같다. 시드 키를 numpy 배열로 바꾸지 **않는** 이유는 크로스도메인
        # (`xseed.build_seed_dict`)이 'steam:730' 같은 문자열 키를 섞어 쓰기 때문이다 —
        # 이 함수의 계약은 "키는 아무 타입이나"이다.
        dominant = [seed_rows[i] for i in sim_matrix.argmax(axis=0).tolist()]
        out = {}
        if "max" in strategies:
            out["max"] = self._frame(corpus_df, sim_matrix.max(axis=0), "MAX", dominant)
        if "mean" in strategies:
            out["mean"] = self._frame(corpus_df, sim_matrix.mean(axis=0), "MEAN", dominant)
        if "top2_mean" in strategies:
            s = sim_matrix.mean(axis=0) if n <= 2 else np.sort(sim_matrix, axis=0)[-2:, :].mean(axis=0)
            out["top2_mean"] = self._frame(corpus_df, s, "TOP2_MEAN", dominant)
        return out

    @staticmethod
    def _frame(corpus_df, scores, strategy, dominant):
        df = corpus_df.copy()
        df["seed_similarity"] = scores
        df["dominant_seed"] = dominant
        df["strategy"] = strategy
        return df
