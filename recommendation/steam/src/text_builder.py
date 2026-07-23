# src/text_builder.py
import pandas as pd

from src.config import ARTIFACTS_DIR, ensure_artifacts_dir


def build_semantic_text(
    short_description: str,
    genres: list[str],
    categories: list[str] | None = None,
    steam_rank: int | None = None,
) -> str:
    text = f"Description: {short_description}"
    if len(genres):
        text += f"\nGenres: {', '.join(genres)}"
    if categories is not None and len(categories):
        text += f"\nCategories: {', '.join(categories)}"
    if steam_rank is not None:
        text += f"\nSteam Top Sellers: #{steam_rank}"
    return text


def add_semantic_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_cat = "categories" in df.columns
    has_rank = "steam_rank" in df.columns
    df["semantic_text"] = [
        build_semantic_text(
            d, g,
            categories=r["categories"] if has_cat else None,
            steam_rank=int(r["steam_rank"]) if has_rank and pd.notna(r["steam_rank"]) else None,
        )
        for d, g, r in zip(df["short_description"], df["genres"], df.to_dict("records"))
    ]
    return df


def main():
    out = ensure_artifacts_dir()
    df = pd.read_parquet(out / "dataset.parquet")
    df = add_semantic_text(df)
    df.to_parquet(out / "dataset.parquet", index=False)
    print(f"rows={len(df)}")
    print(df.iloc[0]["semantic_text"])


if __name__ == "__main__":
    main()
