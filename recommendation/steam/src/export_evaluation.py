import sys
from pathlib import Path

import pandas as pd

from src.config import ensure_artifacts_dir, load_config

JUDGMENT_COLUMNS = [
    "experiment_id", "pair_key",
    "anchor_steam_appid", "anchor_name", "anchor_genres",
    "rank",
    "candidate_steam_appid", "candidate_name", "candidate_genres",
    "similarity",
    "has_metacritic", "metacritic_score",
    "has_recommendations", "recommendations_total",
    "relevance", "recommendation_confidence", "error_tag",
    "evaluator", "notes",
]

EXPECTED_PAIRS: list[tuple[int, int, str]] = [
    (526870, 427520, "Satisfactory → Factorio (factory building)"),
    (427520, 526870, "Factorio → Satisfactory (factory building)"),
    (4000, 730, "Garry's Mod → CS2 (Source engine)"),
    (4000, 252490, "Garry's Mod → Rust (multiplayer sandbox)"),
    (264710, 648800, "Subnautica → Raft (ocean survival)"),
    (648800, 264710, "Raft → Subnautica (ocean survival)"),
    (105600, 219740, "Terraria → Don't Starve (2D survival)"),
    (219740, 105600, "Don't Starve → Terraria (2D survival)"),
    (294100, 108600, "RimWorld → Project Zomboid (deep survival sim)"),
    (108600, 294100, "Project Zomboid → RimWorld (deep survival sim)"),
    (646570, 262060, "Slay the Spire → Darkest Dungeon (roguelike)"),
    (413150, 391540, "Stardew Valley → Undertale (pixel-art indie)"),
    (292030, 489830, "Witcher 3 → Skyrim (open-world RPG)"),
    (489830, 292030, "Skyrim → Witcher 3 (open-world RPG)"),
    (739630, 550, "Phasmophobia → L4D2 (co-op horror)"),
    (550, 739630, "L4D2 → Phasmophobia (co-op horror)"),
]


def _find_rank(df: pd.DataFrame | None, anchor_aid: int, candidate_aid: int) -> int | None:
    if df is None:
        return None
    row = df[(df["anchor_steam_appid"] == anchor_aid) & (df["candidate_steam_appid"] == candidate_aid)]
    if len(row) == 0:
        return None
    return int(row["rank"].iloc[0])


def build_judgments(tops: list[pd.DataFrame], k: int) -> pd.DataFrame:
    parts = [t[t["rank"] <= k].copy() for t in tops]
    out = pd.concat(parts, ignore_index=True)
    out["pair_key"] = (
        out["anchor_steam_appid"].astype(str) + ":" + out["candidate_steam_appid"].astype(str)
    )
    for col in ("anchor_genres", "candidate_genres"):
        out[col] = out[col].apply(
            lambda gs: ", ".join(gs) if hasattr(gs, "__iter__") and not isinstance(gs, str) else ""
        )
    out["relevance"] = pd.NA
    out["recommendation_confidence"] = pd.NA
    out["error_tag"] = ""
    out["evaluator"] = ""
    out["notes"] = ""
    return out[JUDGMENT_COLUMNS]


def main():
    pilot = "--pilot" in sys.argv
    cfg = load_config()
    out = ensure_artifacts_dir()
    k = cfg["retrieval"]["evaluation_k"]

    tops = [
        pd.read_parquet(out / "tfidf_top100.parquet"),
        pd.read_parquet(out / "qwen_top100.parquet"),
    ]
    if pilot:
        first10 = pd.read_parquet(out / "anchors_40.parquet")["steam_appid"].tolist()[:10]
        tops = [t[t["anchor_steam_appid"].isin(first10)] for t in tops]

    judgments = build_judgments(tops, k)

    v1_dir = Path("artifacts/s1")
    top100_qwen_v2 = tops[1]
    top100_qwen_v1 = pd.read_parquet(v1_dir / "qwen_top100.parquet") if (v1_dir / "qwen_top100.parquet").exists() else None

    # Expected Candidates sheet
    exp_rows = []
    for aid_a, aid_b, reason in EXPECTED_PAIRS:
        v2_rank = _find_rank(top100_qwen_v2, aid_a, aid_b)
        v1_rank = _find_rank(top100_qwen_v1, aid_a, aid_b)
        delta = (v1_rank - v2_rank) if (v1_rank is not None and v2_rank is not None) else None
        exp_rows.append({
            "pair": reason,
            "anchor_steam_appid": aid_a,
            "candidate_steam_appid": aid_b,
            "v1_rank": v1_rank,
            "v2_rank": v2_rank,
            "delta_v1_minus_v2": delta,
        })
    expected_df = pd.DataFrame(exp_rows)

    name = "evaluation_pilot.xlsx" if pilot else "evaluation.xlsx"
    with pd.ExcelWriter(out / name, engine="openpyxl") as w:
        judgments.to_excel(w, sheet_name="Judgments", index=False)
        top100_qwen_v2.to_excel(w, sheet_name="Top100", index=False)
        expected_df.to_excel(w, sheet_name="Expected_Candidates", index=False)
    dup = judgments["pair_key"].duplicated().sum()
    print(f"wrote {out / name} rows={len(judgments)} duplicated_pair_keys={dup}")
    print(f"  Top100 sheet: {len(top100_qwen_v2)} rows")
    print(f"  Expected_Candidates: {len(expected_df)} pairs")
    found = expected_df["v2_rank"].notna().sum()
    print(f"    found in v2: {found}/{len(expected_df)}")


if __name__ == "__main__":
    main()
