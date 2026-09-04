"""표현 진단 게이트 — 임베딩이 **변별력**을 갖는지 랭커 이전에 잰다.

세 플랫폼 모두 이 게이트를 통과한 뒤에야 랭커를 손댔다. 웹툰도 같다.
측정:
  1) 장르 내 vs 장르 간 평균 유사도 차 — 클수록 장르를 가른다
  2) 유사도 분포의 평탄도 — 1위와 50위 차가 작으면 꼬리에 신호가 없다(Steam 의 문제)
  3) 허브 — 모든 질의의 상위에 나오는 작품 수
    PYTHONPATH=. python -m src.diagnose_representation [--artifacts DIR]
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from src.config import artifact_dir


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifacts", default=None)
    a = ap.parse_args(); d = artifact_dir(a.artifacts)
    E = np.load(d / "corpus_embeddings.npy").astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ds = pd.read_parquet(d / "dataset.parquet").reset_index(drop=True)
    print(f"코퍼스 {len(ds):,}편 · {E.shape[1]}차원")

    g = ds["genres"].apply(lambda x: (list(x)[0] if x is not None and len(x) else "미상"))
    rng = np.random.default_rng(0); idx = rng.choice(len(ds), size=min(1200, len(ds)), replace=False)
    S = E[idx] @ E[idx].T; gg = g.iloc[idx].to_numpy()
    same = (gg[:, None] == gg[None, :]); np.fill_diagonal(same, False)
    off = ~same; np.fill_diagonal(off, False)
    print(f"  장르 내 {S[same].mean():.4f} · 장르 간 {S[off].mean():.4f} · 차 **{S[same].mean()-S[off].mean():+.4f}**")

    q = rng.choice(len(ds), size=200, replace=False)
    sim = E[q] @ E.T
    for r in (1, 10, 20, 50):
        v = np.sort(sim, axis=1)[:, -r-1]
        print(f"  {r:>2}위 유사도 중앙 {np.median(v):.4f}")
    top = np.argsort(-sim, axis=1)[:, 1:51]
    cnt = pd.Series(top.ravel()).value_counts()
    print(f"  허브: 상위 50 에 20회 이상 등장 {int((cnt>=20).sum())}편 · 최다 {int(cnt.iloc[0])}회 "
          f"({ds.iloc[int(cnt.index[0])]['name']})")


if __name__ == "__main__":
    main()
