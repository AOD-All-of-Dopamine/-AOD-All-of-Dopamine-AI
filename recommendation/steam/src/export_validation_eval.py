import pandas as pd

BASE = "artifacts/s1_v2"
DEV_SRC = f"{BASE}/ranking_eval_pilot.xlsx"
R0_SRC = f"{BASE}/qwen_top100.parquet"
R1_SRC = f"{BASE}/ranked_top100_r1_final.parquet"
R2_SRC = f"{BASE}/ranked_top100_r2_final.parquet"
DATASET_SRC = f"{BASE}/dataset.parquet"
OUT = f"{BASE}/ranking_eval_validation_30.xlsx"


def main():
    dev = pd.read_excel(DEV_SRC)
    dev_aids = set(dev["anchor_steam_appid"].unique())
    r0 = pd.read_parquet(R0_SRC)
    r1 = pd.read_parquet(R1_SRC)
    r2 = pd.read_parquet(R2_SRC)
    ds = pd.read_parquet(DATASET_SRC)

    all_aids = set(r0["anchor_steam_appid"].unique())
    val_aids = all_aids - dev_aids
    assert len(val_aids) == 30, f"Expected 30 validation anchors, got {len(val_aids)}"
    assert len(val_aids & dev_aids) == 0, "Validation overlaps with dev!"

    print(f"Validation anchors: {len(val_aids)}")

    # Build genre/description lookup from dataset
    info = ds.set_index("steam_appid")[["name", "genres", "short_description"]].to_dict("index")

    rows = []
    seen = set()

    for aid in sorted(val_aids):
        aname = r0[r0["anchor_steam_appid"] == aid]["anchor_name"].iloc[0]
        ainfo = info.get(aid, {})
        anchor_genres = ainfo.get("genres", [])
        anchor_desc = ainfo.get("short_description", "")

        for label, df in [("R0", r0), ("R1", r1), ("R2", r2)]:
            t10 = df[(df["anchor_steam_appid"] == aid) & (df["rank"] <= 10)]
            for _, cr in t10.iterrows():
                cid = cr["candidate_steam_appid"]
                key = (aid, cid)
                if key in seen:
                    continue
                seen.add(key)
                cinfo = info.get(cid, {})
                candidate_genres = cinfo.get("genres", [])
                candidate_desc = cinfo.get("short_description", "")
                rows.append({
                    "anchor_steam_appid": aid,
                    "anchor_name": aname,
                    "anchor_genres": ", ".join(anchor_genres) if isinstance(anchor_genres, list) else str(anchor_genres),
                    "anchor_short_description": anchor_desc,
                    "candidate_steam_appid": cid,
                    "candidate_name": cr["candidate_name"],
                    "candidate_genres": ", ".join(candidate_genres) if isinstance(candidate_genres, list) else str(candidate_genres),
                    "candidate_short_description": candidate_desc,
                    "r0_rank": 0,
                    "r1_rank": 0,
                    "r2_rank": 0,
                    "relevance": None,
                    "recommendation_confidence": None,
                })

    # Fill ranks
    df_out = pd.DataFrame(rows)
    for label, df in [("R0", r0), ("R1", r1), ("R2", r2)]:
        col = f"{label.lower()}_rank"
        for i, r in df_out.iterrows():
            match = df[(df["anchor_steam_appid"] == r["anchor_steam_appid"]) &
                       (df["candidate_steam_appid"] == r["candidate_steam_appid"])]
            if len(match):
                df_out.at[i, col] = int(match["rank"].iloc[0])
            else:
                df_out.at[i, col] = None

    # Sort by anchor then R2 rank
    df_out = df_out.sort_values(["anchor_steam_appid", "r2_rank"]).reset_index(drop=True)

    # Column order
    cols = [
        "anchor_steam_appid", "anchor_name", "anchor_genres", "anchor_short_description",
        "candidate_steam_appid", "candidate_name", "candidate_genres", "candidate_short_description",
        "r0_rank", "r1_rank", "r2_rank",
        "relevance", "recommendation_confidence",
    ]
    df_out = df_out[cols]
    df_out.to_excel(OUT, index=False, engine="openpyxl")
    print(f"Wrote {OUT}: {len(df_out)} rows, {df_out['anchor_steam_appid'].nunique()} anchors")


if __name__ == "__main__":
    main()
