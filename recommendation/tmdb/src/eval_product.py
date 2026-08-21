"""변형 하나를 평가한다 — 프로필별 P@k + 축별 분해."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.eval_harness import load_bank, variant_recs, score, P1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="{}"); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="dev"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    prof = pd.read_parquet(P1 / "profiles.parquet")
    if a.split != "all": prof = prof[prof["split"] == a.split]
    recs, _ = variant_recs(json.loads(a.variant), a.k, prof)
    bank = load_bank()
    df, ung = score(recs, bank, a.k)
    df = df.merge(prof[["profile_id", "n_seeds", "source", "seed_cohesion"]], on="profile_id")
    df["axis"] = df["profile_id"].str.split("_").str[0]
    print(f"변형 {a.variant} · split={a.split} · k={a.k}")
    print(f"  프로필 {len(df)} · 채점 {int(df['n'].sum())} · 미채점 {ung}")
    print(f"  **적합률 {df['fit'].mean():.4f}**  · 평균 등급 {df['mean_grade'].mean():.3f} "
          f"· 0.8 미만 {int((df['fit'] < 0.8).sum())}개")
    if not a.quiet:
        print("\n  축별:"); 
        for ax, g in df.groupby("axis"):
            print(f"    {ax:<10} n={len(g):>2} 적합 {g['fit'].mean():.3f}")
        print("\n  하위 8개:")
        for r in df.nsmallest(8, "fit").itertuples(index=False):
            print(f"    {r.profile_id:<24} {r.fit:.2f} (시드 {r.n_seeds}, 응집 {r.seed_cohesion:.2f})")
    return df

if __name__ == "__main__": main()
