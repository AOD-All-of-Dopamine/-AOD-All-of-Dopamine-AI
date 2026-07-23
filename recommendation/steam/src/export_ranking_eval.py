"""R0 vs R1 pooled blind evaluation xlsx for Pilot 10 anchors."""

from pathlib import Path

import pandas as pd

from src.config import ensure_artifacts_dir

ARTIFACTS = Path("artifacts/s1_v2")

PILOT_COLUMNS = [
    "experiment", "pair_key",
    "anchor_steam_appid", "anchor_name", "anchor_genres",
    "candidate_steam_appid", "candidate_name", "candidate_genres",
    "r0_rank", "r1_rank",
    "relevance", "recommendation_confidence", "evaluator", "notes",
]

R1_BONUS_COLUMNS = [
    "boost_rec",
]


def build_pooled(anchors: pd.DataFrame, r0: pd.DataFrame, r1: pd.DataFrame, k: int = 10):
    a_ids = anchors["steam_appid"].tolist()[:10]
    seen = set()
    rows = []
    for aid in a_ids:
        r0_t10 = r0[(r0["anchor_steam_appid"] == aid) & (r0["rank"] <= k)]
        r1_t10 = r1[(r1["anchor_steam_appid"] == aid) & (r1["rank"] <= k)]
        r0_map = dict(zip(r0_t10["candidate_steam_appid"], r0_t10["rank"]))
        r1_map = dict(zip(r1_t10["candidate_steam_appid"], r1_t10["rank"]))
        union = set(r0_map.keys()) | set(r1_map.keys())
        # joint: R0 Top-10 + R1 Top-10 union → sort by min rank
        combined = []
        for cid in union:
            r0r = r0_map.get(cid, 999)
            r1r = r1_map.get(cid, 999)
            combined.append((cid, min(r0r, r1r), r0r, r1r))
        combined.sort(key=lambda x: x[1])  # sort by best rank
        for cid, best_rank, r0r, r1r in combined:
            c_row = r0[(r0["anchor_steam_appid"] == aid) & (r0["candidate_steam_appid"] == cid)]
            if len(c_row) == 0:
                c_row = r1[(r1["anchor_steam_appid"] == aid) & (r1["candidate_steam_appid"] == cid)]
            c = c_row.iloc[0]
            pk = f"{aid}:{cid}"
            if pk in seen:
                continue
            seen.add(pk)
            rows.append({
                "experiment": "R0+R1_pooled",
                "pair_key": pk,
                "anchor_steam_appid": aid,
                "anchor_name": c["anchor_name"],
                "anchor_genres": c["anchor_genres"],
                "candidate_steam_appid": cid,
                "candidate_name": c["candidate_name"],
                "candidate_genres": c["candidate_genres"],
                "r0_rank": r0r if r0r <= k else None,
                "r1_rank": r1r if r1r <= k else None,
                "relevance": pd.NA,
                "recommendation_confidence": pd.NA,
                "evaluator": "",
                "notes": "",
            })
    out = pd.DataFrame(rows)
    for col in ("anchor_genres", "candidate_genres"):
        out[col] = out[col].apply(
            lambda gs: ", ".join(gs) if hasattr(gs, "__iter__") and not isinstance(gs, str) else str(gs or "")
        )
    return out


def main():
    out = ensure_artifacts_dir()
    anchors = pd.read_parquet(ARTIFACTS / "anchors_40.parquet")
    r0 = pd.read_parquet(ARTIFACTS / "qwen_top100.parquet")
    r1 = pd.read_parquet(ARTIFACTS / "ranked_top100.parquet")
    pooled = build_pooled(anchors, r0, r1)
    xlsx = out / "ranking_eval_pilot.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pooled.to_excel(w, sheet_name="R0_R1_Pooled", index=False)
    print(f"wrote {xlsx}")
    print(f"  anchors: {pooled['anchor_steam_appid'].nunique()} (first 10)")
    print(f"  total pairs: {len(pooled)}")
    r0_only = (pooled["r1_rank"].isna()).sum()
    r1_only = (pooled["r0_rank"].isna()).sum()
    both = ((pooled["r0_rank"].notna()) & (pooled["r1_rank"].notna())).sum()
    print(f"  R0 only: {r0_only}  R1 only: {r1_only}  both: {both}")


if __name__ == "__main__":
    main()
