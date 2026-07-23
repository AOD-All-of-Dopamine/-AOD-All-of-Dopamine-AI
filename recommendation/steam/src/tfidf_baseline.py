# src/tfidf_baseline.py
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import ensure_artifacts_dir, load_config
from src.retrieve import assemble_top100

EXPERIMENT_ID = "steam_s1_tfidf_v2"


def build_tfidf_matrix(texts: list[str], cfg: dict):
    t = cfg["tfidf"]
    vec = TfidfVectorizer(
        ngram_range=(t["ngram_min"], t["ngram_max"]),
        min_df=t["min_df"],
        max_df=t["max_df"],
        max_features=t["max_features"],
    )
    return vec, vec.fit_transform(texts)


def main():
    cfg = load_config()
    out = ensure_artifacts_dir()
    df = pd.read_parquet(out / "dataset.parquet")
    anchors = pd.read_parquet(out / "anchors_40.parquet")

    _, mat = build_tfidf_matrix(df["semantic_text"].tolist(), cfg)
    appid_to_row = {a: i for i, a in enumerate(df["steam_appid"])}
    positions = [appid_to_row[a] for a in anchors["steam_appid"]]

    sims = (mat[positions] @ mat.T).toarray()
    top = assemble_top100(
        df, anchors, np.asarray(sims), positions,
        k=cfg["retrieval"]["candidate_k"], experiment_id=EXPERIMENT_ID,
    )
    top.to_parquet(out / "tfidf_top100.parquet", index=False)
    print(f"wrote {out / 'tfidf_top100.parquet'} rows={len(top)}")


if __name__ == "__main__":
    main()
