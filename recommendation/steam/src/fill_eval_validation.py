import pandas as pd
import sys

SRC_XLSX = "artifacts/s1_v2/ranking_eval_validation_30.xlsx"
SRC_CSV = "artifacts/s1_v2/validation_judgments.csv"


def main():
    judgments = pd.read_csv(SRC_CSV)
    missing = judgments["relevance"].isna() | judgments["recommendation_confidence"].isna()
    if missing.any():
        print(f"WARNING: {missing.sum()} rows still lack relevance/confidence in CSV")
        print(f"Run 'python -m src.fill_eval_validation' again after filling")
        sys.exit(1)

    df = pd.read_excel(SRC_XLSX)
    df["pair_key"] = df["anchor_steam_appid"].astype(str) + ":" + df["candidate_steam_appid"].astype(str)

    lookup = judgments.set_index("pair_key")[["relevance", "recommendation_confidence"]]
    df["relevance"] = df["pair_key"].map(lookup["relevance"])
    df["recommendation_confidence"] = df["pair_key"].map(lookup["recommendation_confidence"])

    df.drop(columns=["pair_key"], inplace=True)
    df.to_excel(SRC_XLSX, index=False, engine="openpyxl")

    filled_rel = df["relevance"].notna().sum()
    filled_conf = df["recommendation_confidence"].notna().sum()
    print(f"Filled: {filled_rel}/{len(df)} relevance, {filled_conf}/{len(df)} confidence -> {SRC_XLSX}")


if __name__ == "__main__":
    main()
