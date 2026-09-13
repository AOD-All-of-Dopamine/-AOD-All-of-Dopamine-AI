import pandas as pd

SRC = "artifacts/s1_v2/ranking_eval_validation_30.xlsx"


def main():
    df = pd.read_excel(SRC)
    errors = []

    rel_filled = df["relevance"].notna().sum()
    conf_filled = df["recommendation_confidence"].notna().sum()
    if rel_filled != len(df):
        errors.append(f"relevance: {rel_filled}/{len(df)} filled")
    if conf_filled != len(df):
        errors.append(f"recommendation_confidence: {conf_filled}/{len(df)} filled")

    invalids = df[~df["relevance"].isin([0, 1, 2, 3]) & df["relevance"].notna()]
    if len(invalids):
        errors.append(f"{len(invalids)} relevance values not in {{0,1,2,3}}")

    invalids = df[~df["recommendation_confidence"].isin([0, 1, 2, 3]) & df["recommendation_confidence"].notna()]
    if len(invalids):
        errors.append(f"{len(invalids)} confidence values not in {{0,1,2,3}}")

    df["pair_key"] = df["anchor_steam_appid"].astype(str) + ":" + df["candidate_steam_appid"].astype(str)
    dupes = df["pair_key"].duplicated().sum()
    if dupes:
        errors.append(f"{dupes} duplicate pair_key entries")

    anchors = df["anchor_steam_appid"].nunique()
    if anchors != 30:
        errors.append(f"Expected 30 validation anchors, got {anchors}")

    if errors:
        print("INVALID:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print(f"VALID: {len(df)} rows, {anchors} anchors, all scored (0-3)")


if __name__ == "__main__":
    main()
