# src/evaluate.py
import json
import math
import sys

import pandas as pd

from src.config import ensure_artifacts_dir


def dcg(rels: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels: list[float], k: int = 10) -> float:
    rels = [float(r) for r in rels[:k]]
    idcg = dcg(sorted(rels, reverse=True))
    return dcg(rels) / idcg if idcg else 0.0


def precision_at_k(rels: list[float], k: int = 10, threshold: int = 2) -> float:
    rels = rels[:k]
    return sum(1 for r in rels if r >= threshold) / len(rels) if rels else 0.0


def per_anchor_metrics(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    rows = []
    for appid, grp in df.groupby("anchor_steam_appid"):
        rels = grp.sort_values("rank")["relevance"].astype(int).tolist()
        rows.append({
            "anchor_steam_appid": appid,
            "precision": precision_at_k(rels, k),
            "ndcg": ndcg_at_k(rels, k),
            "avg_relevance": sum(rels) / len(rels),
            "avg_confidence": grp["recommendation_confidence"].astype(int).mean(),
        })
    return pd.DataFrame(rows)


def compute_summary(per_anchor: pd.DataFrame) -> dict:
    return {
        "precision_at_10": float(per_anchor["precision"].mean()),
        "ndcg_at_10": float(per_anchor["ndcg"].mean()),
        "avg_relevance": float(per_anchor["avg_relevance"].mean()),
        "avg_confidence": float(per_anchor["avg_confidence"].mean()),
    }


def main():
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "artifacts/s1/evaluation.xlsx"
    out = ensure_artifacts_dir()
    judgments = pd.read_excel(xlsx, sheet_name="Judgments")
    anchors = pd.read_parquet(out / "anchors_40.parquet")

    summaries, seg_rows, genre_rows, per_all = {}, [], [], []
    for exp_id, grp in judgments.groupby("experiment_id"):
        per = per_anchor_metrics(grp)
        per["experiment_id"] = exp_id
        per = per.merge(
            anchors[["steam_appid", "bucket", "primary_stratum"]],
            left_on="anchor_steam_appid", right_on="steam_appid",
        )
        summaries[exp_id] = compute_summary(per)
        per_all.append(per)
        for bucket, g in per.groupby("bucket"):
            seg_rows.append({"experiment_id": exp_id, "bucket": bucket, **compute_summary(g)})
        for stratum, g in per.groupby("primary_stratum"):
            genre_rows.append({"experiment_id": exp_id, "stratum": stratum, **compute_summary(g)})

    t, q = summaries.get("steam_s1_tfidf_v1"), summaries.get("steam_s1_qwen_v1")
    result = {"experiments": summaries}
    if t and q:
        result["comparison"] = {
            "precision_delta": q["precision_at_10"] - t["precision_at_10"],
            "ndcg_delta": q["ndcg_at_10"] - t["ndcg_at_10"],
            "qwen_beats_tfidf": (
                q["precision_at_10"] > t["precision_at_10"]
                and q["ndcg_at_10"] > t["ndcg_at_10"]
            ),
            "quality_gate_passed": q["precision_at_10"] >= 0.70,
        }
    suffix = "_pilot" if "pilot" in xlsx else ""
    with open(out / f"metrics{suffix}.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(seg_rows).to_csv(out / f"metrics_by_segment{suffix}.csv", index=False)
    pd.DataFrame(genre_rows).to_csv(out / f"metrics_by_genre{suffix}.csv", index=False)
    pd.concat(per_all).to_csv(out / f"metrics_by_anchor{suffix}.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
