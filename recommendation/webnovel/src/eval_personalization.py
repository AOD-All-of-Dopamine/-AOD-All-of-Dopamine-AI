# src/eval_personalization.py
"""P1 개인화 평가.

두 종류의 지표를 낸다. 성격이 완전히 다르므로 섞어 쓰면 안 된다.

1. **판정 기반** (`evaluate_from_judgments`) — 전략 선택의 근거.
   프로필별로 계산한 뒤 macro-average 한다. 예전에는 `profile_id` groupby 없이 split 전체를
   한 풀로 놓고 `nlargest(10)` 을 해서, 절대 유사도가 높은 한두 프로필이 지표를 독식했다
   (MEAN/TOP2_MEAN 이 P@10 = 1.0 으로 계산됐다).

2. **leave-one-out** (`evaluate_leave_one_out`) — 판정 없이 돌릴 수 있는 **보조 회귀 지표**.
   "이미 좋아한다고 밝힌 게임"을 맞히는 과제라 참신성을 전혀 측정하지 않는다.
   전략 선택의 근거로 쓰지 말 것.
"""
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.metrics import DEFAULT_GAIN, dcg, precision_at_k

STRATEGIES = ["max", "mean", "top2_mean"]
JUDGMENT_SHEET = "blind_eval"


# ---------------------------------------------------------------- 판정 기반

def load_judged_split(split: str) -> pd.DataFrame:
    """블라인드 판정지 + 전략 매핑을 pair_key 로 조인한다."""
    judged = pd.read_excel(
        PROJECT_ROOT / "artifacts" / "p1_review" / f"p1_eval_{split}_scored_llm_proxy.xlsx",
        sheet_name=JUDGMENT_SHEET,
    )
    mapping = pd.read_parquet(
        PROJECT_ROOT / "artifacts" / "p1_v2" / f"p1_eval_{split}_mapping.parquet"
    )
    return judged.merge(
        mapping.drop(columns=["profile_id", "candidate_appid"]), on="pair_key", how="inner"
    )


def evaluate_from_judgments(
    joined: pd.DataFrame,
    strategies: list[str] | None = None,
    k: int = 10,
    mode: str = DEFAULT_GAIN,
    score_col: str = "relevance",
    conf_col: str = "recommendation_confidence",
) -> pd.DataFrame:
    """프로필별 NDCG@k / P@k / Conf@k 를 계산하고 macro-average 한다.

    IDCG 는 **해당 프로필의 판정 풀 전체**에서 뽑는다 — 그래야 세 전략이 같은 분모를 쓴다.
    Conf@k 는 `recommendation_confidence` 의 평균이다 — 예전 구현은 여기에 `relevance` 를
    넣어 P@k 와 중복된 값을 Conf 라는 이름으로 보고했다.
    """
    if strategies is None:
        strategies = STRATEGIES

    rows = []
    for strategy in strategies:
        score_key = f"{strategy}_score"
        per_profile = []
        for _, grp in joined.groupby("profile_id"):
            ranked = grp[grp[score_key].notna()]
            if ranked.empty:
                continue
            top = ranked.nlargest(k, score_key)
            rels = [float(r) for r in top[score_col] if pd.notna(r)]
            pool = [float(r) for r in grp[score_col] if pd.notna(r)]
            idcg = dcg(sorted(pool, reverse=True)[:k], mode)
            confs = (
                [float(c) for c in top[conf_col] if pd.notna(c)]
                if conf_col in top.columns
                else []
            )
            per_profile.append({
                "ndcg": dcg(rels, mode) / idcg if idcg else 0.0,
                "precision": precision_at_k(rels, k),
                "conf": sum(confs) / len(confs) if confs else 0.0,
                "n": len(rels),
            })
        if not per_profile:
            continue
        per = pd.DataFrame(per_profile)
        rows.append({
            "strategy": strategy.upper(),
            "k": k,
            "profiles": len(per),
            "NDCG": round(per["ndcg"].mean(), 4),
            f"P@{k}": round(per["precision"].mean(), 4),
            f"Conf@{k}": round(per["conf"].mean(), 4),
            "n_judged": int(per["n"].sum()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ leave-one-out

def evaluate_leave_one_out(
    profiles_df: pd.DataFrame,
    ks: tuple[int, ...] = (10, 50, 100, 300),
    strategies: list[str] | None = None,
    pop_boost: float = 0.03,
) -> pd.DataFrame:
    """seed 를 하나 빼고 나머지로 추천한 뒤, **빠진 seed 의 순위**를 측정한다.

    핵심은 제외 목록에서도 hold-out 을 빼는 것이다 — 예전 구현은 랭커에 `liked` 전체를
    넘겨 hold-out 을 후보에서 지워버렸고, `recall = len(top) / len(liked)` 라는
    (k 와 seed 개수만으로 결정되는) 상수를 반환했다.

    **보조 회귀 지표다.** 참신성을 측정하지 않으므로 전략 선택의 근거로 쓰지 말 것.
    """
    from src.personalized_retrieve import build_components, run_multi

    if strategies is None:
        strategies = STRATEGIES
    components = build_components(pop_boost)

    rows = []
    for _, profile in profiles_df.iterrows():
        liked = list(profile["liked_ids"])
        if len(liked) < 2:
            continue
        for held_out in liked:
            remaining = [a for a in liked if a != held_out]
            results = run_multi(
                liked_ids=remaining,
                strategies=strategies,
                top_n=max(ks),
                pop_boost=pop_boost,
                components=components,
            )
            for strategy in strategies:
                ranked = results[strategy]
                hit = ranked.index[ranked["item_id"] == held_out]
                rank = int(ranked.loc[hit[0], "rank"]) if len(hit) else None
                rows.append({
                    "profile_id": profile["profile_id"],
                    "strategy": strategy.upper(),
                    "held_out": int(held_out),
                    "rank": rank,
                    "rr": 1.0 / rank if rank else 0.0,
                    **{f"hit@{k}": int(rank is not None and rank <= k) for k in ks},
                })
    return pd.DataFrame(rows)


def summarize_leave_one_out(
    loo: pd.DataFrame, ks: tuple[int, ...] = (10, 50, 100, 300)
) -> pd.DataFrame:
    agg = {"MRR": ("rr", "mean"), "n": ("rr", "size")}
    agg.update({f"HitRate@{k}": (f"hit@{k}", "mean") for k in ks})
    out = loo.groupby("strategy").agg(**agg).reset_index()
    for c in out.columns:
        if c not in ("strategy", "n"):
            out[c] = out[c].round(4)
    return out


# --------------------------------------------------------------- seed 독점

def seed_dominance_from_rankings(profiles_df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """MAX 전략에서 한 seed 가 Top-k 를 얼마나 독식하는지."""
    from src.personalized_retrieve import build_components, run_multi

    components = build_components()
    rows = []
    for _, profile in profiles_df.iterrows():
        liked = list(profile["liked_ids"])
        results = run_multi(
            liked_ids=liked, strategies=["max"], top_n=k, components=components
        )
        counts = results["max"]["dominant_seed"].value_counts()
        for seed_aid, count in counts.items():
            rows.append({
                "profile_id": profile["profile_id"],
                "dominant_seed": int(seed_aid),
                "top_k_count": int(count),
                "top_k_share": round(count / k, 2),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- main

def main():
    mode = load_config()["evaluation"]["ndcg_gain"]
    print(f"ndcg_gain = {mode}\n")

    for split in ("dev", "val"):
        joined = load_judged_split(split)
        print(f"=== {split.upper()} — 판정 기반 (프로필별 macro-average) ===")
        print(evaluate_from_judgments(joined, mode=mode).to_string(index=False))
        print()

    profiles_path = PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet"
    if not profiles_path.exists():
        print("profiles.parquet 없음 — build_p1_profiles.py 를 먼저 실행하세요.")
        return
    profiles_df = pd.read_parquet(profiles_path)

    print("=== leave-one-out (보조 회귀 지표 — 전략 선택 근거 아님) ===")
    loo = evaluate_leave_one_out(profiles_df)
    print(summarize_leave_one_out(loo).to_string(index=False))


if __name__ == "__main__":
    main()
