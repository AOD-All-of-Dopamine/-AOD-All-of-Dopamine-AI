import numpy as np
import pandas as pd

COHERENT_PROFILES = [
    {
        "profile_id": "coh_fps",
        "liked_appids": [730, 578080, 359550],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_survival_craft",
        "liked_appids": [108600, 219740, 242760],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_openworld_survival",
        "liked_appids": [264710, 305620, 275850],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_strategy",
        "liked_appids": [289070, 268500, 281990],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_grand_strategy",
        "liked_appids": [394360, 236850, 8930],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_arpg",
        "liked_appids": [374320, 292030, 489830],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_indie_platformer",
        "liked_appids": [367520, 588650, 504230],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_cozy",
        "liked_appids": [413150, 433340, 648800],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_vehicle_sim",
        "liked_appids": [227300, 284160, 244210],
        "profile_type": "coherent",
    },
    {
        "profile_id": "coh_classic_multi",
        "liked_appids": [4000, 440, 550],
        "profile_type": "coherent",
    },
]

MIXED_PROFILES = [
    {
        "profile_id": "mix_fps_cozy",
        "liked_appids": [730, 578080, 413150],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_survival_strategy",
        "liked_appids": [242760, 251570, 289070],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_rpg_racing",
        "liked_appids": [292030, 489830, 284160],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_indie_multi",
        "liked_appids": [367520, 588650, 4000],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_openworld_cozy",
        "liked_appids": [275850, 264710, 433340],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_vehicle_fps",
        "liked_appids": [227300, 244210, 550],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_arpg_survival",
        "liked_appids": [374320, 377160, 242760],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_grand_casual",
        "liked_appids": [8930, 236850, 477160],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_multi_indie",
        "liked_appids": [440, 550, 105600],
        "profile_type": "mixed",
    },
    {
        "profile_id": "mix_cozy_fps",
        "liked_appids": [648800, 413150, 730],
        "profile_type": "mixed",
    },
]


def validate_split(dev_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    issues = []
    warnings = []

    dev_ids = set(dev_df["profile_id"])
    val_ids = set(val_df["profile_id"])
    overlap = dev_ids & val_ids
    if overlap:
        issues.append(f"Profile ID overlap: {overlap}")

    dev_seed_sets = [set(p) for p in dev_df["liked_appids"]]
    val_seed_sets = [set(p) for p in val_df["liked_appids"]]

    identical_overlap = 0
    for ds in dev_seed_sets:
        for vs in val_seed_sets:
            if ds == vs:
                identical_overlap += 1
    if identical_overlap > 0:
        issues.append(f"Identical seed-set overlap count: {identical_overlap}")

    max_shared = 0
    for ds in dev_seed_sets:
        for vs in val_seed_sets:
            shared = len(ds & vs)
            max_shared = max(max_shared, shared)
    if max_shared > 2:
        issues.append(f"Max shared seeds between Dev/Val profile pair: {max_shared} (>2)")
    elif max_shared > 1:
        warnings.append(f"Max shared seeds: {max_shared} (≤2, acceptable)")

    dev_seeds_all = set()
    for p in dev_df["liked_appids"]:
        dev_seeds_all.update(p)
    val_seeds_all = set()
    for p in val_df["liked_appids"]:
        val_seeds_all.update(p)

    new_in_val = val_seeds_all - dev_seeds_all
    val_total = len(val_seeds_all)
    new_ratio = len(new_in_val) / val_total if val_total > 0 else 0

    if new_ratio < 0.35:
        issues.append(f"New seeds in Val: {len(new_in_val)}/{val_total} ({new_ratio:.0%}) < 35%")

    print(f"  Unique seeds: Dev={len(dev_seeds_all)}, Val={len(val_seeds_all)}")
    print(f"  New seeds in Val (not in Dev): {len(new_in_val)} / {val_total} ({new_ratio:.0%})")
    print(f"  Max shared seeds per profile pair: {max_shared}")

    if warnings:
        for w in warnings:
            print(f"  (ok) {w}")
    if issues:
        msg = "\n".join(issues)
        raise ValueError(f"Split validation failed:\n{msg}")


def build_profiles_df(max_retries: int = 100) -> pd.DataFrame:
    all_profiles = COHERENT_PROFILES + MIXED_PROFILES

    for attempt in range(max_retries):
        rng = np.random.default_rng(42 + attempt)
        shuffled = all_profiles.copy()
        rng.shuffle(shuffled)

        rows = []
        for i, p in enumerate(shuffled):
            rows.append({
                "profile_id": p["profile_id"],
                "liked_appids": p["liked_appids"],
                "profile_type": p["profile_type"],
                "profile_order": i,
            })

        df = pd.DataFrame(rows)
        df["split"] = "dev"
        dev_count = len(df) // 3 * 2
        df.iloc[dev_count:, df.columns.get_loc("split")] = "val"

        dev_df = df[df["split"] == "dev"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)

        try:
            validate_split(dev_df, val_df)
            print(f"  Valid split found after {attempt+1} attempt(s)")
            return df
        except ValueError:
            continue

    raise RuntimeError(f"Could not find valid split after {max_retries} retries")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from config import ensure_artifacts_dir

    out_dir = ensure_artifacts_dir()
    p1_dir = out_dir.parent / "p1"
    p1_dir.mkdir(parents=True, exist_ok=True)

    df = build_profiles_df()
    path = p1_dir / "profiles.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} profiles to {path}")
    print(f"  Dev: {(df['split']=='dev').sum()}, Val: {(df['split']=='val').sum()}")
    for _, r in df.iterrows():
        print(f"  {r['profile_id']:20s} [{r['split']}] liked={r['liked_appids']}")
