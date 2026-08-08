# src/build_p1_profiles.py
"""평가 프로필 정의 — 판정 414쌍이 이 파일 위에 서 있다.

**절대 어기면 안 되는 불변식**

판정은 `(profile_id, candidate_appid)` 로만 키가 잡히는데 **관련성은 시드에 의존한다.**
따라서 기존 프로필의 `liked` 를 바꾸면 조인은 성공하는데 의미가 달라진다 —
가장 위험한 종류의 오염이다. `profile_id` 를 바꾸면 조인이 조용히 실패해 미판정이 된다.

  · 이미 판정된 프로필의 `liked` · `profile_id` 를 바꾸지 않는다.
  · `split` 은 라벨이라 바꿔도 판정은 안전하다. 단 **dev → val 이동은 금지** —
    튜닝에 쓴 프로필은 홀드아웃이 될 수 없다. val → dev 한 방향만 허용한다.

**왜 split 탐색을 없앴나**

예전 구현은 rng 로 섞어가며 통과하는 split 을 찾았다(`max_retries=100`). 그 방식은
(a) 이미 튜닝에 쓴 dev 프로필을 val 로 보낼 수 있고 (b) 판정 이력과 무관하게 재배치한다.
게다가 검증 기준이 `max_shared > 2` 라 **3개 중 2개 공유가 통과**했다. 실측 결과:

    val 8개 중 4개가 dev 프로필과 시드를 2개씩 공유
    val(누수 4개) P@10 0.800  vs  val(깨끗 4개) P@10 0.650  (dev 0.750)

"홀드아웃이 버텼다"는 결론이 그 누수분이 만든 것이었다. 지금은 split 을 정의에 명시하고
검증만 한다 — `max_shared >= 2` 면 실패한다.

**니치 축**

기존 34개 시드가 전부 코퍼스 상위 0.22%(최저 TF2 47,572 리뷰)라 니치 취향이 하나도 없었다.
그래서 "시드 인기도에 따라 인기도 부스트를 조절한다"는 실험이 모든 프로필에서 같은 값이라
검증 불가로 끝났다. `niche_*` 6개는 **리뷰 1,000~20,000 구간에서만** 뽑았고,
시드 개수도 1·3·5·10 으로 섞어 고정 3개 가정을 깬다.
"""
import hashlib

import numpy as np
import pandas as pd

DEV, VAL = "dev", "val"

# 기존 20개 — liked 는 절대 수정 금지 (판정 414쌍이 여기에 매여 있다).
# split 만 조정했다: 아래 4개가 dev 프로필과 시드를 2개씩 공유해 홀드아웃 자격이 없어
# val → dev 로 옮겼다 (mix_multi_indie / coh_openworld_survival / mix_indie_multi / coh_vehicle_sim).
LEGACY_PROFILES = [
    # --- 선언상 coherent ---
    {"profile_id": "coh_fps",                "liked": [730, 578080, 359550],    "declared": "coherent", "split": DEV},
    {"profile_id": "coh_survival_craft",     "liked": [108600, 219740, 242760], "declared": "coherent", "split": VAL},
    {"profile_id": "coh_openworld_survival", "liked": [264710, 305620, 275850], "declared": "coherent", "split": DEV},   # ← 누수
    {"profile_id": "coh_strategy",           "liked": [289070, 268500, 281990], "declared": "coherent", "split": DEV},
    {"profile_id": "coh_grand_strategy",     "liked": [394360, 236850, 8930],   "declared": "coherent", "split": VAL},
    {"profile_id": "coh_arpg",               "liked": [374320, 292030, 489830], "declared": "coherent", "split": DEV},
    {"profile_id": "coh_indie_platformer",   "liked": [367520, 588650, 504230], "declared": "coherent", "split": DEV},
    {"profile_id": "coh_cozy",               "liked": [413150, 433340, 648800], "declared": "coherent", "split": DEV},
    {"profile_id": "coh_vehicle_sim",        "liked": [227300, 284160, 244210], "declared": "coherent", "split": DEV},   # ← 누수
    {"profile_id": "coh_classic_multi",      "liked": [4000, 440, 550],         "declared": "coherent", "split": DEV},
    # --- 선언상 mixed ---
    {"profile_id": "mix_fps_cozy",           "liked": [730, 578080, 413150],    "declared": "mixed", "split": DEV},
    {"profile_id": "mix_survival_strategy",  "liked": [242760, 251570, 289070], "declared": "mixed", "split": VAL},
    {"profile_id": "mix_rpg_racing",         "liked": [292030, 489830, 284160], "declared": "mixed", "split": DEV},
    {"profile_id": "mix_indie_multi",        "liked": [367520, 588650, 4000],   "declared": "mixed", "split": DEV},      # ← 누수
    {"profile_id": "mix_openworld_cozy",     "liked": [275850, 264710, 433340], "declared": "mixed", "split": DEV},
    {"profile_id": "mix_vehicle_fps",        "liked": [227300, 244210, 550],    "declared": "mixed", "split": DEV},
    {"profile_id": "mix_arpg_survival",      "liked": [374320, 377160, 242760], "declared": "mixed", "split": DEV},
    {"profile_id": "mix_grand_casual",       "liked": [8930, 236850, 477160],   "declared": "mixed", "split": VAL},
    {"profile_id": "mix_multi_indie",        "liked": [440, 550, 105600],       "declared": "mixed", "split": DEV},      # ← 누수
    {"profile_id": "mix_cozy_fps",           "liked": [648800, 413150, 730],    "declared": "mixed", "split": DEV},
]

# 니치 축 — 시드는 전부 리뷰 1,000~20,000. 시드 개수도 1·3·5·10 으로 섞는다.
NICHE_PROFILES = [
    {"profile_id": "niche_puzzle_solo", "split": VAL, "declared": "single",
     "liked": [558990]},                                                    # Opus Magnum 5,856
    {"profile_id": "niche_soulslike_solo", "split": VAL, "declared": "single",
     "liked": [283640]},                                                    # Salt and Sanctuary 19,987
    {"profile_id": "niche_tactics", "split": DEV, "declared": "coherent",
     "liked": [590380, 1102190, 287980]},                                   # Into the Breach / Monster Train / Mini Metro
    {"profile_id": "niche_cozy_casual", "split": DEV, "declared": "coherent",
     "liked": [1055540, 1046030, 557600]},                                  # A Short Hike / ISLANDERS / Gorogoa
    {"profile_id": "niche_sim", "split": VAL, "declared": "mixed",
     "liked": [1161580, 365450, 1677770, 1435790, 2162800]},                # Shipbreaker / Hacknet / Golden Idol / 방탈출 / shapez 2
    {"profile_id": "niche_roguelite", "split": DEV, "declared": "coherent",
     "liked": [894020, 330020, 692890, 418530, 242680,
               512900, 788100, 1123770, 1253920, 1740720]},                 # 10개
]

ALL_PROFILES = LEGACY_PROFILES + NICHE_PROFILES

NICHE_REVIEW_BAND = (1000, 20000)


# ------------------------------------------------------------------ 검증

def seeds_fingerprint(profiles: list[dict]) -> str:
    """`(profile_id, liked)` 의 지문. 기존 프로필이 조용히 바뀌었는지 잡는다."""
    h = hashlib.sha256()
    for p in sorted(profiles, key=lambda x: x["profile_id"]):
        h.update(f"{p['profile_id']}:{','.join(map(str, p['liked']))}|".encode())
    return h.hexdigest()[:12]


def validate_split(profiles: list[dict], max_shared: int = 1) -> dict:
    """dev/val 이 진짜로 분리돼 있는지 검증한다. 탐색하지 않고 주어진 것을 본다.

    `max_shared=1` — 시드 3개 중 2개를 공유하면 사실상 같은 프로필이다.
    예전 기준(`> 2` 일 때만 실패)은 그것을 통과시켰고, 그래서 홀드아웃이 오염됐다.
    """
    dev = [p for p in profiles if p["split"] == DEV]
    val = [p for p in profiles if p["split"] == VAL]
    if not dev or not val:
        raise ValueError(f"split 이 한쪽으로 쏠렸습니다: dev {len(dev)} / val {len(val)}")

    ids = [p["profile_id"] for p in profiles]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"profile_id 중복: {dup}")

    worst, leaks = 0, []
    for v in val:
        sv = set(v["liked"])
        for d in dev:
            shared = len(sv & set(d["liked"]))
            worst = max(worst, shared)
            if shared > max_shared:
                leaks.append(f"{v['profile_id']} ~ {d['profile_id']} ({shared}개 공유)")
    if leaks:
        raise ValueError(
            f"dev–val 시드 누수 {len(leaks)}건 (허용 {max_shared}개 이하):\n  "
            + "\n  ".join(leaks)
        )

    dev_seeds = {a for p in dev for a in p["liked"]}
    return {
        "dev": len(dev), "val": len(val),
        "max_shared_dev_val": worst,
        "val_seeds": len({a for p in val for a in p["liked"]}),
        "val_seeds_also_in_dev": len({a for p in val for a in p["liked"]} & dev_seeds),
        "unique_seeds": len({a for p in profiles for a in p["liked"]}),
        "seed_slots": sum(len(p["liked"]) for p in profiles),
    }


def validate_niche(profiles: list[dict], dataset: pd.DataFrame) -> dict:
    """니치 프로필의 시드가 정말 니치 구간인지 확인한다.

    이것을 안 재면 "니치 축을 추가했다"고 적어놓고 실제로는 또 대작만 들어 있을 수 있다.
    """
    rec = dataset.set_index("steam_appid")["recommendations_total"].astype("float")
    lo, hi = NICHE_REVIEW_BAND
    bad = []
    for p in profiles:
        if not p["profile_id"].startswith("niche_"):
            continue
        for a in p["liked"]:
            r = rec.get(a)
            if pd.isna(r) or not (lo <= r <= hi):
                bad.append(f"{p['profile_id']}/{a} 리뷰 {r}")
    if bad:
        raise ValueError(f"니치 구간({lo:,}~{hi:,}) 밖의 시드:\n  " + "\n  ".join(bad))

    all_seeds = [a for p in profiles for a in p["liked"]]
    revs = rec.reindex(all_seeds).fillna(0)
    pct = rec.fillna(0).rank(pct=True).reindex(all_seeds)
    return {
        "seed_reviews_min": int(revs.min()), "seed_reviews_max": int(revs.max()),
        "seed_percentile_min": round(float(pct.min()), 4),
        "seed_percentile_max": round(float(pct.max()), 4),
    }


# ------------------------------------------------------- 라벨을 측정으로 대체

def compute_seed_cohesion(profiles: list[dict], artifacts: str | None = None) -> dict[str, float]:
    """시드 쌍 코사인의 평균. 시드 1개면 NaN (쌍이 없다).

    손으로 붙인 `coherent`/`mixed` 라벨이 데이터로 확인되지 않았기 때문에 넣는다 —
    실측하면 coherent 평균 0.581 / mixed 0.528 로 구간이 완전히 겹쳤고
    `coh_arpg`(0.478) 가 `mix_multi_indie`(0.643) 보다 덜 일관적이었다.
    """
    from src.personalization.seed_loader import SeedLoader

    loader = SeedLoader(artifacts)
    out = {}
    for p in profiles:
        liked = p["liked"]
        if len(liked) < 2:
            out[p["profile_id"]] = float("nan")
            continue
        vecs = list(loader.load(liked).values())
        sims = [float(np.dot(a, b)) for i, a in enumerate(vecs) for b in vecs[i + 1:]]
        out[p["profile_id"]] = float(np.mean(sims))
    return out


def derive_profile_type(cohesion: dict[str, float]) -> dict[str, str]:
    """실측 응집도의 중앙값으로 coherent/mixed 를 가른다. 시드 1개는 `single`."""
    vals = [v for v in cohesion.values() if not np.isnan(v)]
    med = float(np.median(vals)) if vals else 0.0
    return {
        pid: "single" if np.isnan(v) else ("coherent" if v >= med else "mixed")
        for pid, v in cohesion.items()
    }


# ------------------------------------------------------------------ 빌드

def build_profiles_df(artifacts: str | None = None, dataset: pd.DataFrame | None = None,
                      profiles: list[dict] | None = None) -> pd.DataFrame:
    profiles = profiles or ALL_PROFILES
    stats = validate_split(profiles)
    if dataset is not None:
        stats |= validate_niche(profiles, dataset)

    cohesion = compute_seed_cohesion(profiles, artifacts)
    derived = derive_profile_type(cohesion)

    df = pd.DataFrame([{
        "profile_id": p["profile_id"],
        "liked_appids": list(p["liked"]),
        "n_seeds": len(p["liked"]),
        "profile_type": derived[p["profile_id"]],
        "profile_type_declared": p["declared"],
        "seed_cohesion": cohesion[p["profile_id"]],
        "split": p["split"],
        "profile_order": i,
    } for i, p in enumerate(profiles)])
    df.attrs["stats"] = stats
    return df


def main():
    import argparse
    import json

    from src.config import ARTIFACTS_DIR, artifact_dir

    # 기본 artifacts 는 s1_v2(트렌드 트랙)라 개인화 코퍼스가 아니다.
    # 니치 구간 검증과 응집도 계산은 전체 코퍼스 임베딩이 필요하다.
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/full_v1")
    args = ap.parse_args()

    art = artifact_dir(args.artifacts)
    ds_path = art / "dataset.parquet"
    dataset = pd.read_parquet(ds_path) if ds_path.exists() else None

    df = build_profiles_df(artifacts=args.artifacts, dataset=dataset)
    p1_dir = ARTIFACTS_DIR.parent / "p1"
    p1_dir.mkdir(parents=True, exist_ok=True)
    path = p1_dir / "profiles.parquet"
    df.to_parquet(path, index=False)

    print(f"저장: {path}  ({len(df)}행)")
    print(json.dumps(df.attrs["stats"], ensure_ascii=False, indent=2))
    print(f"시드 지문: {seeds_fingerprint(ALL_PROFILES)}  (기존 프로필이 바뀌면 달라진다)")
    flipped = df[(df.profile_type != df.profile_type_declared)]
    print(f"\n선언 라벨과 실측이 다른 프로필 {len(flipped)}/{len(df)}:")
    for _, r in flipped.iterrows():
        print(f"  {r.profile_id:24s} {r.profile_type_declared} → {r.profile_type} "
              f"(응집도 {r.seed_cohesion:.3f})")
    print()
    for _, r in df.iterrows():
        print(f"  {r.profile_id:24s} [{r.split}] seeds={r.n_seeds:2d} "
              f"cohesion={r.seed_cohesion:.3f} {r.profile_type}")


if __name__ == "__main__":
    main()
