# src/retrieve.py
import numpy as np
import pandas as pd


def assemble_top100(
    df: pd.DataFrame,
    anchors: pd.DataFrame,
    sims: np.ndarray,
    anchor_positions: list[int],
    k: int,
    experiment_id: str,
) -> pd.DataFrame:
    records = []
    for i, (_, anchor) in enumerate(anchors.iterrows()):
        self_row = anchor_positions[i]
        order = np.argsort(-sims[i], kind="stable")
        rank = 0
        for j in order:
            if j == self_row:
                continue
            rank += 1
            if rank > k:
                break
            cand = df.iloc[j]
            records.append({
                "experiment_id": experiment_id,
                "anchor_steam_appid": int(anchor["steam_appid"]),
                "anchor_name": anchor["name"],
                "anchor_genres": anchor["genres"],
                "rank": rank,
                "candidate_steam_appid": int(cand["steam_appid"]),
                "candidate_name": cand["name"],
                "candidate_genres": cand["genres"],
                "similarity": float(sims[i][j]),
                "has_metacritic": bool(cand["has_metacritic"]),
                "metacritic_score": cand["metacritic_score"],
                "has_recommendations": bool(cand["has_recommendations"]),
                "recommendations_total": cand["recommendations_total"],
            })
    return pd.DataFrame.from_records(records)


from src.config import ensure_artifacts_dir, load_config

QWEN_EXPERIMENT_ID = "steam_s1_qwen_v2"


def main():
    cfg = load_config()
    out = ensure_artifacts_dir()
    df = pd.read_parquet(out / "dataset.parquet")
    anchors = pd.read_parquet(out / "anchors_40.parquet")
    idx = pd.read_parquet(out / "corpus_index.parquet")

    corpus = np.load(out / "corpus_embeddings.npy")
    queries = np.load(out / "anchor_embeddings.npy")

    # corpus row 순서에 맞춘 metadata
    df_meta = idx[["steam_appid"]].merge(df, on="steam_appid", how="left")
    pos_map = {a: i for i, a in enumerate(idx["steam_appid"])}
    positions = [pos_map[a] for a in anchors["steam_appid"]]

    sims = queries @ corpus.T
    top = assemble_top100(
        df_meta, anchors, sims, positions,
        k=cfg["retrieval"]["candidate_k"], experiment_id=QWEN_EXPERIMENT_ID,
    )
    top.to_parquet(out / "qwen_top100.parquet", index=False)
    print(f"wrote {out / 'qwen_top100.parquet'} rows={len(top)}")


if __name__ == "__main__":
    main()
