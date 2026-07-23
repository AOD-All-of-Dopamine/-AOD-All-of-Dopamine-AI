import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DATASET = "artifacts/s1_v2/dataset.parquet"
STRATEGIES = ["max", "mean", "top2_mean"]
TOP_N = 300


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATASET)


def get_game_info(dataset: pd.DataFrame, appids: list[int]) -> dict[int, dict]:
    subset = dataset[dataset["steam_appid"].isin(appids)]
    result = {}
    for _, r in subset.iterrows():
        genres = ", ".join(r["genres"]) if isinstance(r["genres"], (list, np.ndarray)) else str(r["genres"])
        desc = str(r.get("short_description", ""))[:200]
        result[int(r["steam_appid"])] = {
            "name": r["name"],
            "genres": genres,
            "short_description": desc,
        }
    return result


def run_profile_rankings(liked_appids: list[int]) -> dict[str, pd.DataFrame]:
    from personalized_retrieve import run_multi
    return run_multi(
        liked_appids=liked_appids,
        strategies=STRATEGIES,
        top_n=TOP_N,
        rec_boost=0.03,
    )


def export_pooled_eval(
    profiles_df: pd.DataFrame,
    output_blind_xlsx: str,
    output_mapping_parquet: str,
    split: str = "dev",
):
    dataset = load_dataset()
    profiles = profiles_df[profiles_df["split"] == split]

    blind_rows = []
    mapping_rows = []

    for _, profile in profiles.iterrows():
        liked = list(profile["liked_appids"])
        profile_id = profile["profile_id"]

        liked_info = get_game_info(dataset, liked)
        liked_detail = {}
        for i, aid in enumerate(liked):
            info = liked_info.get(aid, {"name": str(aid), "genres": "", "short_description": ""})
            liked_detail[f"liked_game_{i+1}_name"] = info["name"]
            liked_detail[f"liked_game_{i+1}_appid"] = aid
            liked_detail[f"liked_game_{i+1}_genres"] = info["genres"]
            liked_detail[f"liked_game_{i+1}_desc"] = info["short_description"]

        results = run_profile_rankings(liked)

        pool = {}
        for strategy, df in results.items():
            top10 = df.head(10)
            for _, row in top10.iterrows():
                aid = int(row["steam_appid"])
                if aid not in pool:
                    pool[aid] = {
                        "candidate_name": row["name"],
                    }
                pool[aid][f"{strategy}_score"] = round(row["final_score"], 4)
                pool[aid][f"{strategy}_rank"] = int(row["rank"])
                pool[aid][f"{strategy}_sim"] = round(row["seed_similarity"], 4)

        for aid, info in pool.items():
            cand_info = get_game_info(dataset, [aid]).get(aid, {"name": str(aid), "genres": "", "short_description": ""})

            pair_key = f"{profile_id}__{aid}"

            blind_rows.append({
                "pair_key": pair_key,
                "profile_id": profile_id,
                **liked_detail,
                "candidate_appid": aid,
                "candidate_name": cand_info["name"],
                "candidate_genres": cand_info["genres"],
                "candidate_short_description": cand_info["short_description"],
                "relevance": "",
                "recommendation_confidence": "",
                "evaluation_type": "LLM_PROXY",
                "evaluator": "",
                "notes": "",
            })

            mapping_rows.append({
                "pair_key": pair_key,
                "profile_id": profile_id,
                "candidate_appid": aid,
                "max_score": info.get("max_score"),
                "max_rank": info.get("max_rank"),
                "max_seed_similarity": info.get("max_sim"),
                "mean_score": info.get("mean_score"),
                "mean_rank": info.get("mean_rank"),
                "mean_seed_similarity": info.get("mean_sim"),
                "top2_mean_score": info.get("top2_mean_score"),
                "top2_mean_rank": info.get("top2_mean_rank"),
                "top2_mean_seed_similarity": info.get("top2_mean_sim"),
            })

    blind_df = pd.DataFrame(blind_rows)
    blind_df = blind_df.sort_values(["profile_id", "candidate_name"]).reset_index(drop=True)

    mapping_df = pd.DataFrame(mapping_rows)

    with pd.ExcelWriter(output_blind_xlsx, engine="openpyxl") as writer:
        blind_df.to_excel(writer, sheet_name="blind_eval", index=False)

    mapping_df.to_parquet(output_mapping_parquet, index=False)

    print(f"Exported {len(blind_df)} blind rows to {output_blind_xlsx}")
    print(f"Exported {len(mapping_df)} mapping rows to {output_mapping_parquet}")
    for pid in blind_df["profile_id"].unique():
        subset = blind_df[blind_df["profile_id"] == pid]
        print(f"  {pid}: {len(subset)} candidates")
    return blind_df, mapping_df


def main():
    import sys
    sys.path.insert(0, "src")
    from config import PROJECT_ROOT

    profiles_path = PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet"
    if not profiles_path.exists():
        print(f"Error: profiles not found at {profiles_path}. Run build_p1_profiles.py first.")
        sys.exit(1)

    profiles_df = pd.read_parquet(profiles_path)

    out_dir = PROJECT_ROOT / "artifacts" / "p1_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["dev", "val"]:
        blind_path = out_dir / f"p1_eval_{split}_blind.xlsx"
        mapping_path = out_dir / f"p1_eval_{split}_mapping.parquet"
        export_pooled_eval(profiles_df, str(blind_path), str(mapping_path), split=split)


if __name__ == "__main__":
    main()
