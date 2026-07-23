import numpy as np
import pandas as pd

SRC = "artifacts/s1_v2/ranking_eval_pilot.xlsx"
VARIANTS = [
    ("R1-final", "artifacts/s1_v2/ranked_top100_r1_final.parquet"),
    ("R2-1%",   "artifacts/s1_v2/ranked_top100_r2_meta001.parquet"),
    ("R2-2%",   "artifacts/s1_v2/ranked_top100_r2_meta002.parquet"),
    ("R2-3%",   "artifacts/s1_v2/ranked_top100_r2_meta003.parquet"),
]


def dcg_at_k(gains, k=10):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(gains, idcg, k=10):
    dcg = dcg_at_k(gains, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_idcgs(pooled, anchors, k=10):
    idcgs = {}
    for aid in anchors:
        judged = pooled[pooled["anchor_steam_appid"] == aid]
        all_gains = sorted(judged["relevance"].astype(int), reverse=True)[:k]
        idcgs[aid] = dcg_at_k(all_gains, k)
    return idcgs


def precision_at_k(gains, k=10, threshold=2):
    return sum(1 for g in gains[:k] if g >= threshold) / k


def _eval_variant(pooled, df, anchors, idcgs=None):
    rows = []
    for aid in anchors:
        aname = pooled[pooled["anchor_steam_appid"] == aid]["anchor_name"].iloc[0]
        idcg = idcgs.get(aid) if idcgs else None
        t10 = df[(df["anchor_steam_appid"] == aid) & (df["rank"] <= 10)].copy()
        gains, confs = [], []
        for _, cr in t10.iterrows():
            cid = cr["candidate_steam_appid"]
            match = pooled[(pooled["anchor_steam_appid"] == aid) & (pooled["candidate_steam_appid"] == cid)]
            if len(match):
                gains.append(int(match["relevance"].iloc[0]))
                confs.append(int(match["recommendation_confidence"].iloc[0]))
            else:
                gains.append(0)
                confs.append(0)
        ndcg = ndcg_at_k(gains, idcg) if idcg else 0.0
        rows.append({
            "anchor": aname,
            "P@10": precision_at_k(gains, 10),
            "NDCG": ndcg,
            "Conf": np.mean(confs),
            "gains": gains,
        })
    return pd.DataFrame(rows)


def main():
    pooled = pd.read_excel(SRC)
    anchors = pooled["anchor_steam_appid"].unique()

    idcgs = compute_idcgs(pooled, anchors)

    results = {}
    for label, src in VARIANTS:
        df = pd.read_parquet(src)
        results[label] = _eval_variant(pooled, df, anchors, idcgs)

    # Print header
    header = f"{'Anchor':25s}"
    for label, _, in VARIANTS:
        header += f"  {label + ' P@10':>10s}  {label + ' NDCG':>10s}  {label + ' Conf':>10s}"
    print(header)
    print("-" * len(header))
    for i in range(len(results[VARIANTS[0][0]])):
        line = f"{results[VARIANTS[0][0]].iloc[i]['anchor']:25s}"
        for label, _, in VARIANTS:
            r = results[label].iloc[i]
            line += f"  {r['P@10']:10.2f}  {r['NDCG']:10.3f}  {r['Conf']:10.2f}"
        print(line)

    print("-" * len(header))
    line = f"{'MEAN':25s}"
    for label, _, in VARIANTS:
        r = results[label]
        line += f"  {r['P@10'].mean():10.2f}  {r['NDCG'].mean():10.3f}  {r['Conf'].mean():10.2f}"
    print(line)

    # Detailed gains
    for label, _, in VARIANTS:
        print(f"\n=== {label} Top-10 gains ===")
        for _, r in results[label].iterrows():
            print(f"  {r['anchor'][:22]:22s}  {r['gains']}")


if __name__ == "__main__":
    main()
