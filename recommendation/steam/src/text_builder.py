# src/text_builder.py
import pandas as pd

from src.config import ensure_artifacts_dir


def build_semantic_text(
    short_description: str,
    genres: list[str],
    categories: list[str] | None = None,
) -> str:
    """임베딩에 넣을 의미 텍스트.

    판매 순위(`steam_rank`)는 **일부러 넣지 않는다.** 인기도는 의미 신호가 아니라 랭킹
    신호이고, R1 이 이미 `recommendations_percentile` 로 랭킹 단계에서 다룬다. 텍스트에
    섞으면 인기 게임끼리 가까워져 의미 공간이 오염되고 부스트와 이중 계산된다.
    인기도 반영을 키우려면 rank.py 의 부스트 계수가 올바른 손잡이다.
    """
    text = f"Description: {short_description}"
    if len(genres):
        text += f"\nGenres: {', '.join(genres)}"
    if categories is not None and len(categories):
        text += f"\nCategories: {', '.join(categories)}"
    return text


def add_semantic_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_cat = "categories" in df.columns
    df["semantic_text"] = [
        build_semantic_text(d, g, categories=c if has_cat else None)
        for d, g, c in zip(
            df["short_description"],
            df["genres"],
            df["categories"] if has_cat else [None] * len(df),
        )
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
