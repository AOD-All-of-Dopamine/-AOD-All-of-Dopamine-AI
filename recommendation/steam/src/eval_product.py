# src/eval_product.py
"""제품 경로(`next_page`) 평가 — 판정 풀 검사와 신뢰구간을 강제한다.

이 모듈이 생긴 이유는 두 가지 사고 때문이다.

**1. NDCG 가 1을 넘고 있었다.**
`ndcg_at_k` 의 `ideal_pool` 에 "그 프로필에서 지금까지 판정된 것 전부"를 넘겼는데,
랭킹 상위 10개 중 일부가 아직 판정 전이면 pool ⊅ ranking 이 되어 DCG > IDCG 가 된다.
실측으로 P07 1.160 / P02 1.021 이 나왔다. `assert_pool_coverage` 라는 가드가
`src/metrics.py` 에 이미 있었지만 S1 트랙에서만 부르고 개인화 경로에서는 한 번도 안 불렀다.

**2. 판정을 늘리면 과거 숫자가 움직인다.**
같은 추천 목록인데 판정 풀이 127쌍 → 334쌍으로 늘자 평균 NDCG 가 0.887 → 0.630 으로
0.257 움직였다. 측정하려던 효과(0.013~0.138)의 2~20배다. 그래서 NDCG 를 registry 에
남길 때는 반드시 `pool_id` 와 `pool_size` 를 같이 박고, 다른 pool 의 값과 나란히 놓지 않는다.

**3. 8프로필 × 10칸 = 1칸이 0.0125 다.**
`compare()` 는 짝지은 부트스트랩 신뢰구간을 항상 함께 낸다. 유의하지 않은 차이를
"개선"이라고 적는 것을 막기 위해서다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.metrics import (
    DEFAULT_GAIN,
    POSITIVE_THRESHOLD,
    SLOT_DENOM,
    dcg,
    format_ci,
    paired_bootstrap_ci,
    precision_at_k,
)

PROFILES_PATH = PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet"
WORKBOOK = PROJECT_ROOT / "artifacts" / "human_eval" / "recommendation_review_claude.xlsx"
EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"


# ------------------------------------------------------------------ 판정 로드

def load_judgments() -> dict[tuple[str, int], int]:
    """엑셀 워크북(최초 127쌍) + judgments_claude.py(이후 추가분)를 합친다.

    판정이 두 군데로 갈라져 있는 것은 역사적 사고다 — 워크북은 블라인드 채점용이었고,
    실험이 돌면서 새로 진입한 후보는 코드에 적는 편이 diff 로 추적하기 쉬웠다.
    합칠 때 코드 쪽이 나중이므로 우선한다.
    """
    from src.eval_human import load_human

    d = load_human(str(WORKBOOK))
    d = d[d["relevance"].notna()]
    out = {(r["profile_id"], int(r["candidate_appid"])): int(r["relevance"]) for _, r in d.iterrows()}

    if str(EXPERIMENTS_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS_DIR))
    try:
        from judgments_claude import ALL

        out.update({k: v[0] for k, v in ALL.items()})
    except ImportError:
        pass
    return out


def judged_profiles(judgments: dict[tuple[str, int], int]) -> list[str]:
    return sorted({pid for pid, _ in judgments})


# ------------------------------------------------------------------ 평가

def evaluate_config(
    ranker,
    profile_ids: list[str] | None = None,
    judgments: dict[tuple[str, int], int] | None = None,
    k: int = 10,
    mode: str = DEFAULT_GAIN,
    label: str = "config",
    strict: bool = True,
) -> pd.DataFrame:
    """`ranker(profile_id, liked_appids) -> DataFrame[steam_appid]` 를 받아 프로필별 지표를 낸다.

    `strict=True` 면 Top-k 중 미판정이 하나라도 있을 때 예외를 낸다. 조용히 0점을 주면
    새 후보를 많이 끌어오는 설정이 자동으로 불리해진다 — 실제로 구/신 코퍼스 비교에서
    구 쪽 14칸이 미판정이던 동안 결론이 반대로 나왔다가, 채우고 나서 뒤집힌 적이 있다.
    """
    judgments = judgments if judgments is not None else load_judgments()
    profiles = pd.read_parquet(PROFILES_PATH).set_index("profile_id")
    profile_ids = profile_ids or judged_profiles(judgments)

    rows = []
    for pid in profile_ids:
        ranked = ranker(pid, list(profiles.loc[pid, "liked_appids"]))
        appids = [int(a) for a in ranked["steam_appid"][:k]]
        missing = [a for a in appids if (pid, a) not in judgments]
        if missing and strict:
            raise ValueError(
                f"{label}/{pid}: Top-{k} 중 {len(missing)}쌍이 미판정입니다 {missing[:5]}. "
                "판정을 채운 뒤 다시 실행하세요 — 미판정을 0점으로 두면 비교가 무효입니다."
            )
        rels = [judgments[(pid, a)] for a in appids if (pid, a) in judgments]
        pool = [v for (q, _), v in judgments.items() if q == pid]
        idcg = dcg(sorted(pool, reverse=True)[:k], mode)
        actual = dcg(rels, mode)
        if actual > idcg + 1e-9:
            raise ValueError(
                f"{label}/{pid}: NDCG > 1 — 판정 풀이 랭킹을 못 덮습니다 "
                f"(pool {len(pool)}, top-{k} {len(rels)})."
            )
        rows.append({
            "profile_id": pid,
            "unjudged": len(missing),
            # 분모는 판정 개수가 아니라 칸 수 k — 미판정 칸은 실패로 센다.
            # (strict=False 로 열어둔 경우에도 정밀도가 부풀려지지 않게 한다)
            "p_at_k": precision_at_k(rels, k, POSITIVE_THRESHOLD, denominator=SLOT_DENOM),
            "ndcg_at_k": actual / idcg if idcg else 0.0,
            "mean_relevance": sum(rels) / len(rels) if rels else 0.0,
            "zeros": sum(1 for r in rels if r == 0),
            "pool_size": len(pool),
        })
    return pd.DataFrame(rows)


def summarize(per_profile: pd.DataFrame) -> dict:
    return {
        "profiles": int(len(per_profile)),
        "unjudged": int(per_profile["unjudged"].sum()),
        "P@10": round(float(per_profile["p_at_k"].mean()), 4),
        "NDCG@10": round(float(per_profile["ndcg_at_k"].mean()), 4),
        "mean_relevance": round(float(per_profile["mean_relevance"].mean()), 4),
        "zeros_per_profile": round(float(per_profile["zeros"].mean()), 4),
        "pool_size_total": int(per_profile["pool_size"].sum()),
    }


# ------------------------------------------------------------------ 비교

def compare(
    results: dict[str, pd.DataFrame],
    baseline: str | None = None,
    metric: str = "p_at_k",
    seed: int = 0,
) -> pd.DataFrame:
    """설정들을 기준선 대비 짝지은 신뢰구간과 함께 비교한다.

    `results` 는 `{라벨: evaluate_config 결과}`. 모든 결과가 같은 프로필 집합이어야 한다.
    """
    labels = list(results)
    baseline = baseline or labels[0]
    order = results[baseline]["profile_id"].tolist()
    for lab, df in results.items():
        if df["profile_id"].tolist() != order:
            raise ValueError(f"{lab}: 프로필 집합/순서가 기준선과 다릅니다 — 짝지을 수 없습니다.")

    base = results[baseline][metric].to_numpy()
    rows = []
    for lab in labels:
        s = summarize(results[lab])
        row = {"설정": lab, **{kk: s[kk] for kk in ("P@10", "NDCG@10", "unjudged")}}
        if lab == baseline:
            row |= {"Δ": 0.0, "95%CI": "(기준선)", "유의": ""}
        else:
            ci = paired_bootstrap_ci(base, results[lab][metric].to_numpy(), seed=seed)
            row |= {
                "Δ": round(ci["delta"], 4),
                "95%CI": f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]",
                "유의": "예" if ci["significant"] else "아니오",
            }
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 제품 설정

def product_ranker(artifacts: str = "artifacts/full_v1", **overrides):
    """지금 제품이 쓰는 설정 그대로의 ranker 를 만든다."""
    from src.personalized_retrieve import build_components, next_page

    comp = build_components(overrides.pop("rec_boost", 0.15), artifacts=artifacts)

    def rank(profile_id: str, liked: list[int]) -> pd.DataFrame:
        return next_page(liked, page_size=overrides.get("page_size", 10), components=comp,
                         **{kk: vv for kk, vv in overrides.items() if kk != "page_size"})

    return rank


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/full_v1")
    ap.add_argument("--profiles", default="judged", help="'judged' | 'dev' | 'val' | 'all'")
    ap.add_argument("--allow-unjudged", action="store_true")
    args = ap.parse_args()

    j = load_judgments()
    if args.profiles == "judged":
        pids = judged_profiles(j)
    else:
        p = pd.read_parquet(PROFILES_PATH)
        pids = sorted(p["profile_id"] if args.profiles == "all"
                      else p[p["split"] == args.profiles]["profile_id"])

    per = evaluate_config(product_ranker(args.artifacts), pids, j,
                          label="product", strict=not args.allow_unjudged)
    print(per.to_string(index=False, float_format="%.3f"))
    print()
    print(json.dumps(summarize(per), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
