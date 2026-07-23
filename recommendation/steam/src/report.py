# src/report.py
import sys

import pandas as pd

from src.config import ensure_artifacts_dir


def error_distribution(df: pd.DataFrame) -> pd.DataFrame:
    low = df[df["relevance"].astype(int) <= 1]
    rows = []
    for exp_id, grp in low.groupby("experiment_id"):
        vc = grp["error_tag"].value_counts()
        total = vc.sum()
        for tag, n in vc.items():
            rows.append({
                "experiment_id": exp_id, "error_tag": tag,
                "count": int(n), "ratio": round(n / total, 4) if total else 0.0,
            })
    return pd.DataFrame(rows)


def failure_examples(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    low = df[df["relevance"].astype(int) <= 1]
    cols = [
        "experiment_id", "anchor_name", "candidate_name", "similarity",
        "rank", "relevance", "error_tag", "candidate_genres",
    ]
    return low.sort_values("similarity", ascending=False).head(n)[cols]


def main():
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "artifacts/s1/evaluation.xlsx"
    out = ensure_artifacts_dir()
    df = pd.read_excel(xlsx, sheet_name="Judgments")
    suffix = "_pilot" if "pilot" in xlsx else ""

    dist = error_distribution(df)
    dist.to_csv(out / f"error_distribution{suffix}.csv", index=False)
    failure_examples(df).to_csv(out / f"failure_examples{suffix}.csv", index=False)

    anchor_metrics = pd.read_csv(out / f"metrics_by_anchor{suffix}.csv")
    worst = anchor_metrics.sort_values("precision").groupby("experiment_id").head(10)
    worst.to_csv(out / f"worst_anchors{suffix}.csv", index=False)

    print("=== error distribution ===")
    print(dist.to_string(index=False))
    if len(dist):
        top = dist.loc[dist["ratio"].idxmax()]
        print(f"\n가장 큰 error category: {top['error_tag']} ({top['ratio']:.0%}) — experiment {top['experiment_id']}")
        print("→ experiment_decision.md에 다음 가설 1개를 기록할 것")


if __name__ == "__main__":
    main()
