import sys

import pandas as pd

VALID_ERROR_TAGS = {
    "IRRELEVANT", "GENRE_ONLY", "KEYWORD_MATCH", "MODE_MISMATCH",
    "FRANCHISE_OR_VARIANT", "DESCRIPTION_SPARSE", "NICHE_NOISE", "OTHER",
    "THEME_MATCH_BUT_GAMEPLAY_MISMATCH", "GENERIC_SANDBOX_SIMILARITY",
}

JUDGE_COLS = ["relevance", "recommendation_confidence", "error_tag"]


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    errors = []
    for i, row in df.iterrows():
        rel, conf = row["relevance"], row["recommendation_confidence"]
        for name, val in (("relevance", rel), ("recommendation_confidence", conf)):
            if pd.isna(val):
                errors.append(f"row {i}: {name} empty")
            elif int(val) not in (0, 1, 2, 3):
                errors.append(f"row {i}: {name}={val} invalid (0~3)")
        tag = str(row["error_tag"]).strip() if not pd.isna(row["error_tag"]) else ""
        if tag and tag not in VALID_ERROR_TAGS:
            errors.append(f"row {i}: error_tag '{tag}' invalid")
        if not pd.isna(rel) and int(rel) <= 1 and not tag:
            errors.append(f"row {i}: relevance<=1 but error_tag empty")
    return errors


def fill_duplicate_judgments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for _, grp in df.groupby("pair_key"):
        judged = grp[grp["relevance"].notna()]
        if len(judged) and grp["relevance"].isna().any():
            src = judged.iloc[0]
            df.loc[grp.index[grp["relevance"].isna()], JUDGE_COLS] = src[JUDGE_COLS].values
    return df


def main():
    mode, path = sys.argv[1], sys.argv[2]
    df = pd.read_excel(path, sheet_name="Judgments")
    if mode == "fill":
        df = fill_duplicate_judgments(df)
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Judgments", index=False)
        print(f"filled duplicates in {path}")
    elif mode == "check":
        errors = validate_dataframe(df)
        if errors:
            print(f"INVALID ({len(errors)} errors):")
            print("\n".join(errors[:50]))
            raise SystemExit(1)
        print(f"OK: {len(df)} rows valid")
    else:
        raise SystemExit("usage: python -m src.validate_evaluation [check|fill] <xlsx>")


if __name__ == "__main__":
    main()
