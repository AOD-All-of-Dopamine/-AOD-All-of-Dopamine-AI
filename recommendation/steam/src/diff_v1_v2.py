import sys
from pathlib import Path

import pandas as pd

V1_DIR = Path("artifacts/s1")
V2_DIR = Path("artifacts/s1_v2")
V3_DIR = Path("artifacts/s1_v3")

MODELS = {
    "qwen": ("qwen_top100.parquet", "steam_s1_qwen_v1", "steam_s1_qwen_v2", "steam_s1_qwen_v3"),
    "tfidf": ("tfidf_top100.parquet", "steam_s1_tfidf_v1", "steam_s1_tfidf_v2", "steam_s1_tfidf_v3"),
}

VERSIONS = {"v1": V1_DIR, "v2": V2_DIR, "v3": V3_DIR}


def load(name: str, dir: Path, eid: str) -> pd.DataFrame | None:
    fname = MODELS[name][0]
    p = dir / fname
    if not p.exists():
        return None
    return pd.read_parquet(p).query("experiment_id == @eid")


def topk_overlap(a: pd.DataFrame, b: pd.DataFrame, k: int) -> float:
    anchors = sorted(set(a["anchor_steam_appid"].unique()) & set(b["anchor_steam_appid"].unique()))
    if not anchors:
        return 0.0
    overlaps = []
    for aid in anchors:
        s1 = set(a[(a["anchor_steam_appid"] == aid) & (a["rank"] <= k)]["candidate_steam_appid"])
        s2 = set(b[(b["anchor_steam_appid"] == aid) & (b["rank"] <= k)]["candidate_steam_appid"])
        if s1:
            overlaps.append(len(s1 & s2) / len(s1))
    return pd.Series(overlaps).mean()


def rank_change(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    anchors = sorted(set(a["anchor_steam_appid"].unique()) & set(b["anchor_steam_appid"].unique()))
    rows = []
    for aid in anchors:
        d1 = a[a["anchor_steam_appid"] == aid][["candidate_steam_appid", "rank"]]
        d2 = b[b["anchor_steam_appid"] == aid][["candidate_steam_appid", "rank"]]
        m = d1.merge(d2, on="candidate_steam_appid", suffixes=("_a", "_b"), how="outer")
        m["rank_change"] = m["rank_a"] - m["rank_b"]
        rows.append({
            "anchor_steam_appid": aid,
            "candidates_common": m["rank_change"].notna().sum(),
            "candidates_only_a": m["rank_b"].isna().sum(),
            "candidates_only_b": m["rank_a"].isna().sum(),
            "mean_rank_change": m["rank_change"].mean(),
            "median_rank_change": m["rank_change"].median(),
        })
    return pd.DataFrame(rows)


def compare(label_a: str, label_b: str):
    dir_a, dir_b = VERSIONS[label_a], VERSIONS[label_b]
    print(f"\n{'='*60}")
    print(f"  {label_a} vs {label_b}")
    print(f"{'='*60}")
    for name in MODELS:
        fname, *eids = MODELS[name]
        eid_a, eid_b = eids[{"v1": 0, "v2": 1, "v3": 2}[label_a]], eids[{"v1": 0, "v2": 1, "v3": 2}[label_b]]
        a, b = load(name, dir_a, eid_a), load(name, dir_b, eid_b)
        if a is None:
            print(f"\n  [{name}] {label_a} missing — skip")
            continue
        if b is None:
            print(f"\n  [{name}] {label_b} missing — skip")
            continue
        print(f"\n  [{name}]")
        for k in (10, 100):
            ov = topk_overlap(a, b, k)
            print(f"    Top-{k} overlap (mean): {ov:.1%}")
        rc = rank_change(a, b)
        print(f"    Rank change (shared candidates):")
        print(f"      mean={rc['mean_rank_change'].mean():+.1f}  median={rc['median_rank_change'].median():+.1f}")
        print(f"      only_{label_a} mean={rc['candidates_only_a'].mean():.1f}  only_{label_b} mean={rc['candidates_only_b'].mean():.1f}")


def main():
    pair = "v2-vs-v3" if "--v2-vs-v3" in sys.argv else "v1-vs-v2"
    label_a, label_b = pair.split("-vs-")
    compare(label_a, label_b)

    # Expected candidates lookup for v2 vs v3
    if pair == "v2-vs-v3" and "--expected" in sys.argv:
        print(f"\n  --- Expected Candidates (v2 vs v3) ---")
        pairs = [
            (526870, 427520, "Satisfactory→Factorio"),
            (427520, 526870, "Factorio→Satisfactory"),
            (4000, 730, "Garrys Mod→CS2"),
            (4000, 252490, "Garrys Mod→Rust"),
            (264710, 648800, "Subnautica→Raft"),
            (648800, 264710, "Raft→Subnautica"),
            (105600, 219740, "Terraria→Don't Starve"),
            (219740, 105600, "Don't Starve→Terraria"),
            (294100, 108600, "RimWorld→Project Zomboid"),
            (108600, 294100, "Project Zomboid→RimWorld"),
            (646570, 262060, "Slay the Spire→Darkest Dungeon"),
            (413150, 391540, "Stardew→Undertale"),
            (292030, 489830, "Witcher 3→Skyrim"),
            (489830, 292030, "Skyrim→Witcher 3"),
            (739630, 550, "Phasmophobia→L4D2"),
            (550, 739630, "L4D2→Phasmophobia"),
        ]
        for name in MODELS:
            fn, *eids = MODELS[name]
            a = load(name, V2_DIR, eids[1])
            b = load(name, V3_DIR, eids[2])
            if a is None or b is None:
                continue
            print(f"  [{name}]")
            for a_aid, c_aid, label in pairs:
                v2r = a[(a["anchor_steam_appid"] == a_aid) & (a["candidate_steam_appid"] == c_aid)]
                v3r = b[(b["anchor_steam_appid"] == a_aid) & (b["candidate_steam_appid"] == c_aid)]
                r2 = int(v2r["rank"].iloc[0]) if len(v2r) > 0 else None
                r3 = int(v3r["rank"].iloc[0]) if len(v3r) > 0 else None
                delta = f"{r2 - r3:+d}" if (r2 is not None and r3 is not None) else "-"
                print(f"    {label:30s}  v2={r2}  v3={r3}  Δ={delta}")

    if "--per-anchor" in sys.argv:
        label_a, label_b = "v1", "v2"
        if "--v2-vs-v3" in sys.argv:
            label_a, label_b = "v2", "v3"
        dir_a, dir_b = VERSIONS[label_a], VERSIONS[label_b]
        for name in MODELS:
            fn, *eids = MODELS[name]
            eid_a, eid_b = eids[{"v1": 0, "v2": 1, "v3": 2}[label_a]], eids[{"v1": 0, "v2": 1, "v3": 2}[label_b]]
            a, b = load(name, dir_a, eid_a), load(name, dir_b, eid_b)
            if a is None or b is None:
                continue
            print(f"\n  [{name}] per-anchor ({label_a} vs {label_b})")
            rc = rank_change(a, b)
            for _, r in rc.iterrows():
                print(f"    anchor {int(r['anchor_steam_appid'])}: common={r['candidates_common']:.0f} "
                      f"only_{label_a}={r['candidates_only_a']:.0f} only_{label_b}={r['candidates_only_b']:.0f} "
                      f"mean_delta={r['mean_rank_change']:+.1f}")


if __name__ == "__main__":
    main()
