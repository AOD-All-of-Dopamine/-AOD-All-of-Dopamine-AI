# src/eval_ranking.py
"""S1 랭킹 변형(R1/R2) 비교.

pool 계약: **비교 대상 전 변형의 Top-k union 이 pool 이고, pool 안은 전부 판정한다.**
예전에는 판정이 없는 후보를 relevance 0 으로 채웠는데(`gains.append(0)`), 그러면 커버리지가
낮은 변형이 조용히 불리해진다. 지금은 `assert_pool_coverage` 로 먼저 실패시킨다 —
새 변형을 추가하면 "pool 재생성 필요"로 멈추는 것이 정상 동작이다.
"""
import numpy as np
import pandas as pd

from src.config import load_config
from src.metrics import DEFAULT_GAIN, assert_pool_coverage, dcg, precision_at_k

SRC = "artifacts/s1_v2/ranking_eval_pilot.xlsx"
VARIANTS = [
    ("R1-final", "artifacts/s1_v2/ranked_top100_r1_final.parquet"),
    ("R2-1%", "artifacts/s1_v2/ranked_top100_r2_meta001.parquet"),
    ("R2-2%", "artifacts/s1_v2/ranked_top100_r2_meta002.parquet"),
    ("R2-3%", "artifacts/s1_v2/ranked_top100_r2_meta003.parquet"),
]
KEY_COLS = ["anchor_steam_appid", "candidate_steam_appid"]


def compute_idcgs(pooled: pd.DataFrame, anchors, k: int = 10, mode: str = DEFAULT_GAIN) -> dict:
    """앵커별 IDCG — 분모는 그 앵커의 판정 풀 전체에서 나온다."""
    idcgs = {}
    for aid in anchors:
        judged = pooled[pooled["anchor_steam_appid"] == aid]
        idcgs[aid] = dcg(sorted(judged["relevance"].astype(float), reverse=True)[:k], mode)
    return idcgs


def _eval_variant(
    pooled: pd.DataFrame,
    df: pd.DataFrame,
    anchors,
    idcgs: dict,
    k: int = 10,
    mode: str = DEFAULT_GAIN,
) -> pd.DataFrame:
    lookup = pooled.set_index(KEY_COLS)[["relevance", "recommendation_confidence"]]
    rows = []
    for aid in anchors:
        aname = pooled[pooled["anchor_steam_appid"] == aid]["anchor_name"].iloc[0]
        top = df[(df["anchor_steam_appid"] == aid) & (df["rank"] <= k)].sort_values("rank")
        judged = lookup.loc[
            [(aid, int(c)) for c in top["candidate_steam_appid"]]
        ]
        gains = judged["relevance"].astype(float).tolist()
        confs = judged["recommendation_confidence"].astype(float).tolist()
        idcg = idcgs.get(aid, 0.0)
        rows.append({
            "anchor": aname,
            "P@10": precision_at_k(gains, k),
            "NDCG": dcg(gains, mode) / idcg if idcg else 0.0,
            "Conf": float(np.mean(confs)) if confs else 0.0,
            "gains": [int(g) for g in gains],
        })
    return pd.DataFrame(rows)


def main():
    mode = load_config()["evaluation"]["ndcg_gain"]
    pooled = pd.read_excel(SRC)
    anchors = pooled["anchor_steam_appid"].unique()
    idcgs = compute_idcgs(pooled, anchors, mode=mode)
    print(f"ndcg_gain = {mode} | pool: {len(pooled)}쌍 / {len(anchors)}앵커\n")

    results = {}
    for label, src in VARIANTS:
        df = pd.read_parquet(src)
        df = df[df["anchor_steam_appid"].isin(anchors)]
        n = assert_pool_coverage(pooled, df, KEY_COLS, k=10, label=label)
        print(f"  {label}: Top-10 {n}쌍 전부 판정됨")
        results[label] = _eval_variant(pooled, df, anchors, idcgs, mode=mode)
    print()

    header = f"{'Anchor':25s}" + "".join(
        f"  {label + ' P@10':>12s}  {label + ' NDCG':>12s}  {label + ' Conf':>12s}"
        for label, _ in VARIANTS
    )
    print(header)
    print("-" * len(header))
    for i in range(len(results[VARIANTS[0][0]])):
        line = f"{results[VARIANTS[0][0]].iloc[i]['anchor'][:24]:25s}"
        for label, _ in VARIANTS:
            r = results[label].iloc[i]
            line += f"  {r['P@10']:12.2f}  {r['NDCG']:12.3f}  {r['Conf']:12.2f}"
        print(line)

    print("-" * len(header))
    line = f"{'MEAN':25s}"
    for label, _ in VARIANTS:
        r = results[label]
        line += f"  {r['P@10'].mean():12.2f}  {r['NDCG'].mean():12.3f}  {r['Conf'].mean():12.2f}"
    print(line)

    for label, _ in VARIANTS:
        print(f"\n=== {label} Top-10 gains ===")
        for _, r in results[label].iterrows():
            print(f"  {r['anchor'][:22]:22s}  {r['gains']}")


if __name__ == "__main__":
    main()
