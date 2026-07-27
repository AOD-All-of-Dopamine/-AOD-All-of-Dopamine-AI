# src/eval_human.py
"""사람 채점 결과를 분석한다.

세 가지를 한 번에 낸다:
  1. 추천 품질 — 프로필별 macro-average NDCG@10 / P@10 (evaluate_from_judgments 와 동일 규칙)
  2. MAX vs TOP2_MEAN 블라인드 비교 — 동결 베이스라인이 맞았는지
  3. **LLM proxy 일치도** — 지금까지 모든 P1 결론의 근거였던 LLM 판정을 믿어도 되는지 (§8 B5)
"""
import sys

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.eval_personalization import evaluate_from_judgments
from src.export_human_eval import BASELINE, CHALLENGER, OUT_DIR

SCORE_COL = "적합도"
TAG_COL = "이유_태그"
LLM_SPLITS = ("dev", "val")


def load_human(xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name="추천평가")
    mapping = pd.read_parquet(OUT_DIR / "recommendation_mapping.parquet")
    joined = df.merge(mapping, on="pair_key", how="inner")
    joined["relevance"] = pd.to_numeric(joined[SCORE_COL], errors="coerce")
    return joined


def load_llm_judgments() -> pd.DataFrame:
    frames = []
    for split in LLM_SPLITS:
        p = PROJECT_ROOT / "artifacts" / "p1_review" / f"p1_eval_{split}_scored_llm_proxy.xlsx"
        d = pd.read_excel(p, sheet_name="blind_eval")
        frames.append(d[["profile_id", "candidate_appid", "relevance", "recommendation_confidence"]])
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns={
        "relevance": "llm_relevance", "recommendation_confidence": "llm_confidence"
    })


def report_progress(joined: pd.DataFrame) -> pd.DataFrame:
    scored = joined[joined["relevance"].notna()]
    print("=" * 68)
    print("1. 채점 진행률")
    print("=" * 68)
    for pri, grp in joined.groupby("우선순위"):
        done = grp["relevance"].notna().sum()
        print(f"  우선순위 {pri}: {done}/{len(grp)}쌍 ({done / len(grp):.0%})")
    if scored.empty:
        raise SystemExit("\n채점된 행이 없습니다. 적합도 컬럼(0~3)을 채운 뒤 다시 실행하세요.")
    bad = scored[~scored["relevance"].isin([0, 1, 2, 3])]
    if len(bad):
        print(f"\n  ⚠ 0~3 범위를 벗어난 값 {len(bad)}건: {sorted(bad['relevance'].unique())}")
    print()
    return scored


def report_quality(scored: pd.DataFrame, mode: str):
    print("=" * 68)
    print("2. 추천 품질 (프로필별 macro-average)")
    print("=" * 68)
    # evaluate_from_judgments 는 profile_id 로 묶는다. 익명 ID 로 대체해도 그룹은 같다.
    frame = scored.copy()
    frame["profile_id"] = frame["anon_profile_id"]
    res = evaluate_from_judgments(frame, strategies=[BASELINE, CHALLENGER], k=10, mode=mode)
    print(res[["strategy", "profiles", "NDCG", "P@10", "n_judged"]].to_string(index=False))

    b = res[res["strategy"] == BASELINE.upper()]
    c = res[res["strategy"] == CHALLENGER.upper()]
    if len(b) and len(c):
        d_ndcg = float(b["NDCG"].iloc[0] - c["NDCG"].iloc[0])
        d_p = float(b["P@10"].iloc[0] - c["P@10"].iloc[0])
        winner = BASELINE.upper() if d_ndcg > 0 else CHALLENGER.upper()
        print(f"\n  NDCG 차이 (MAX - TOP2) = {d_ndcg:+.4f}, P@10 차이 = {d_p:+.4f} → {winner} 우세")
        if abs(d_ndcg) < 0.03:
            print("  ※ 차이가 0.03 미만이면 이 표본에서는 '무승부'로 보는 것이 안전합니다.")
    print()


def report_llm_agreement(scored: pd.DataFrame, evaluator: str = "사람"):
    is_human = evaluator == "사람"
    print("=" * 68)
    if is_human:
        print("3. LLM proxy 일치도 — 지금까지의 P1 결론을 믿어도 되는가")
    else:
        print(f"3. 판정자 간 일치도 ({evaluator} vs GPT-5.6)")
        print("   ※ 둘 다 LLM 이므로 이것은 사람 검증이 아니다. LLM 판정의 '안정성'만 잰다.")
    print("=" * 68)
    llm = load_llm_judgments()
    both = scored.merge(llm, on=["profile_id", "candidate_appid"], how="inner")
    both = both[both["llm_relevance"].notna()]
    if len(both) < 10:
        print(f"  겹치는 판정이 {len(both)}쌍뿐이라 일치도를 계산하지 않습니다.")
        return

    h, l = both["relevance"].astype(float), both["llm_relevance"].astype(float)
    exact = float((h == l).mean())
    within1 = float((abs(h - l) <= 1).mean())
    spearman = float(pd.Series(h).corr(pd.Series(l), method="spearman"))
    bias = float((h - l).mean())

    from sklearn.metrics import cohen_kappa_score

    kappa = float(cohen_kappa_score(h.astype(int), l.astype(int), weights="linear"))

    gap = both["llm_relevance"].astype(float) - both["relevance"].astype(float)
    higher, same, lower = int((gap > 0).sum()), int((gap == 0).sum()), int((gap < 0).sum())

    print(f"  비교 가능한 쌍: {len(both)}")
    print(f"  정확 일치      : {exact:.1%}")
    print(f"  ±1 이내        : {within1:.1%}")
    print(f"  Spearman ρ     : {spearman:.3f}")
    print(f"  가중 Cohen's κ : {kappa:.3f}")
    print(f"  편향({evaluator}-GPT): {bias:+.2f}  "
          f"({evaluator + '이 더 후함' if bias > 0 else 'GPT-5.6 이 더 후함'})")
    print(f"  방향성         : GPT 높음 {higher} / 동일 {same} / {evaluator} 높음 {lower}")

    # 불일치가 한쪽으로만 쏠리면 '노이즈'가 아니라 '기준 차이'다 — 처방이 달라진다.
    one_sided = min(higher, lower) == 0 and max(higher, lower) > 0
    if one_sided:
        print("    → 불일치가 **한 방향으로만** 발생한다. 랜덤 노이즈가 아니라 계통적인")
        print("      기준(캘리브레이션) 차이다. 순위 비교는 살아남지만 절대 수치는 못 믿는다.")

    print("\n  판정:")
    if kappa >= 0.6:
        print(f"    κ >= 0.6 — 판정이 안정적이다.")
        if is_human:
            print("    → 표본 확대(B6)를 LLM 으로 진행해도 됩니다.")
    elif kappa >= 0.4:
        print("    0.4 <= κ < 0.6 — 중간 정도 일치. 방향성은 믿되 0.05 미만의 지표 차이로는")
        print("    결론을 내리지 마세요.")
        if not is_human:
            print("    → 절대 수치(P@10 = 0.8 같은)를 성과 지표로 인용하지 마세요.")
    else:
        print("    κ < 0.4 — 판정을 신뢰하기 어렵습니다.")
        print("    → 지금까지의 P1 결론(MAX 동결 포함)을 다시 세워야 합니다.")

    print(f"\n  구간별 불일치 ({evaluator} 점수 기준):")
    for score in [0, 1, 2, 3]:
        sub = both[both["relevance"] == score]
        if len(sub):
            print(f"    {evaluator} {score}점 ({len(sub):3d}쌍) → GPT-5.6 평균 {sub['llm_relevance'].mean():.2f}")
    print()


def report_tags(scored: pd.DataFrame):
    low = scored[scored["relevance"] <= 1]
    print("=" * 68)
    print(f"4. 낮은 점수의 원인 ({len(low)}쌍, 전체의 {len(low) / len(scored):.0%})")
    print("=" * 68)
    if low.empty:
        print("  0~1점 판정이 없습니다.")
        print()
        return
    tags = low[TAG_COL].fillna("(미기재)").astype(str).str.strip().replace("", "(미기재)")
    vc = tags.value_counts()
    for tag, n in vc.items():
        print(f"  {tag:24s} {n:3d}  ({n / len(low):.0%})")
    top = vc.index[0]
    if top != "(미기재)":
        print(f"\n  가장 큰 실패 유형: {top} — 다음 개선의 1순위 후보입니다.")
        hint = {
            "FRANCHISE_OR_VARIANT": "→ C3 다양성 후처리(프랜차이즈 dedup)가 직접 해결합니다.",
            "GENRE_ONLY": "→ C6 표현 개선(장르 외 신호 추가)이 필요합니다.",
            "KEYWORD_MATCH": "→ C6 표현 개선. 설명 텍스트가 짧아 단어 겹침에 의존하고 있습니다.",
            "MODE_MISMATCH": "→ categories(싱글/멀티) 를 랭킹 피처로 올리는 것을 검토하세요.",
            "TOO_NICHE": "→ C1 인기도 부스트를 키우거나 hard filter 를 검토하세요.",
        }.get(top)
        if hint:
            print(f"  {hint}")
    print()


def report_worst_profiles(scored: pd.DataFrame):
    print("=" * 68)
    print("5. 프로필별 결과 (낮은 순)")
    print("=" * 68)
    per = (
        scored.groupby(["anon_profile_id", "profile_type"])["relevance"]
        .agg(평균="mean", 채점수="size")
        .reset_index()
        .sort_values("평균")
    )
    per["평균"] = per["평균"].round(2)
    print(per.to_string(index=False))
    coh = per[per["profile_type"] == "coherent"]["평균"].mean()
    mix = per[per["profile_type"] == "mixed"]["평균"].mean()
    print(f"\n  coherent 평균 {coh:.2f}  vs  mixed 평균 {mix:.2f}")
    if abs(coh - mix) > 0.3:
        worse = "취향이 섞인" if mix < coh else "취향이 일관된"
        print(f"  → {worse} 프로필에서 눈에 띄게 약합니다. 집계 전략(C4)을 다시 볼 근거입니다.")
    print()


def _evaluator_label() -> str:
    """--evaluator 로 판정자를 밝힌다. 사람이 아니면 리포트 문구가 달라진다."""
    for i, a in enumerate(sys.argv):
        if a == "--evaluator" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "사람"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    xlsx = args[0] if args else str(OUT_DIR / "recommendation_review.xlsx")
    evaluator = _evaluator_label()
    mode = load_config()["evaluation"]["ndcg_gain"]
    joined = load_human(xlsx)
    scored = report_progress(joined)
    report_quality(scored, mode)
    report_llm_agreement(scored, evaluator)
    report_tags(scored)
    report_worst_profiles(scored)


if __name__ == "__main__":
    main()
