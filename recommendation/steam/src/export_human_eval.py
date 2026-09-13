# src/export_human_eval.py
"""사람이 직접 채점할 추천 평가지를 만든다.

기존 `export_personalization_eval.py` 와 목적이 다르다:
  - 그쪽은 LLM proxy 판정용. profile_id 가 `coh_fps` 처럼 의미를 노출해서
    summary.json 스스로 "불투명 ID 를 썼어야 했다"고 지적했다.
  - 이쪽은 **사람 블라인드 판정용**이라 profile_id 를 P01~P20 으로 치환하고,
    전략 정보(어느 전략이 몇 위로 뽑았는지)를 평가지에서 완전히 제거한다.
    복원용 매핑은 별도 parquet 으로 나간다.

출력:
  artifacts/human_eval/recommendation_review.xlsx   ← 사람이 채점하는 파일
  artifacts/human_eval/recommendation_mapping.parquet ← 채점 후 조인용 (열어보지 말 것)
  artifacts/human_eval/final_recommendations.parquet  ← 최종 추천 원본(전략/순위 포함)
"""
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.personalized_retrieve import build_components, run_multi

OUT_DIR = PROJECT_ROOT / "artifacts" / "human_eval"
DATASET = PROJECT_ROOT / "artifacts" / "s1_v2" / "dataset.parquet"
PROFILES = PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet"

# 동결 베이스라인과 챌린저. 평가지에서는 둘 다 익명이다.
BASELINE = "max"
CHALLENGER = "top2_mean"
TOP_K = 10
REC_BOOST = 0.03
PRIORITY_PROFILES = 8  # 앞의 N개는 우선 채점 대상으로 표시
SEED = 42


def _game_info(dataset: pd.DataFrame) -> dict[int, dict]:
    info = {}
    for r in dataset.itertuples():
        genres = ", ".join(r.genres) if isinstance(r.genres, (list, np.ndarray)) else ""
        info[int(r.steam_appid)] = {
            "name": r.name,
            "genres": genres,
            "desc": str(r.short_description)[:300],
        }
    return info


def generate_final_recommendations(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """최종 설정(MAX + rec_boost 3%)과 챌린저(TOP2_MEAN)로 프로필별 Top-K 를 만든다."""
    components = build_components(REC_BOOST)
    rows = []
    for _, profile in profiles_df.iterrows():
        liked = list(profile["liked_appids"])
        results = run_multi(
            liked_appids=liked,
            strategies=[BASELINE, CHALLENGER],
            top_n=TOP_K,
            rec_boost=REC_BOOST,
            components=components,
        )
        for strategy, ranked in results.items():
            for r in ranked.itertuples():
                rows.append({
                    "profile_id": profile["profile_id"],
                    "profile_type": profile["profile_type"],
                    "split": profile["split"],
                    "liked_appids": list(liked),
                    "strategy": strategy,
                    "rank": int(r.rank),
                    "candidate_appid": int(r.steam_appid),
                    "candidate_name": r.name,
                    "seed_similarity": round(float(r.seed_similarity), 4),
                    "final_score": round(float(r.final_score), 4),
                })
    return pd.DataFrame(rows)


def build_blind_sheet(recs: pd.DataFrame, dataset: pd.DataFrame, rng: np.random.Generator):
    """전략/순위를 지운 블라인드 평가지 + 복원 매핑."""
    info = _game_info(dataset)
    profile_ids = list(dict.fromkeys(recs["profile_id"]))
    anon = {pid: f"P{i + 1:02d}" for i, pid in enumerate(profile_ids)}

    blind_rows, map_rows = [], []
    for pid in profile_ids:
        grp = recs[recs["profile_id"] == pid]
        liked = list(grp.iloc[0]["liked_appids"])
        anon_id = anon[pid]
        priority = 1 if profile_ids.index(pid) < PRIORITY_PROFILES else 2

        liked_cols = {}
        for i, aid in enumerate(liked, start=1):
            g = info.get(aid, {"name": str(aid), "genres": "", "desc": ""})
            liked_cols[f"좋아하는_게임{i}"] = g["name"]
            liked_cols[f"좋아하는_게임{i}_장르"] = g["genres"]

        # 후보 풀 = 두 전략의 Top-K 합집합. 순서는 섞어서 순위 힌트를 없앤다.
        pool = sorted(set(grp["candidate_appid"]))
        rng.shuffle(pool)
        for aid in pool:
            g = info.get(aid, {"name": str(aid), "genres": "", "desc": ""})
            pair_key = f"{anon_id}__{aid}"
            blind_rows.append({
                "우선순위": priority,
                "pair_key": pair_key,
                "프로필": anon_id,
                **liked_cols,
                "추천_게임": g["name"],
                "추천_게임_장르": g["genres"],
                "추천_게임_설명": g["desc"],
                "적합도": "",       # 0~3 — 아래 가이드 참고
                "이유_태그": "",     # 0~1 점일 때만
                "메모": "",
            })
            sub = grp[grp["candidate_appid"] == aid]
            map_rows.append({
                "pair_key": pair_key,
                "profile_id": pid,
                "anon_profile_id": anon_id,
                "profile_type": grp.iloc[0]["profile_type"],
                "split": grp.iloc[0]["split"],
                "candidate_appid": aid,
                **{
                    f"{s}_rank": (
                        int(sub[sub["strategy"] == s]["rank"].iloc[0])
                        if (sub["strategy"] == s).any() else None
                    )
                    for s in (BASELINE, CHALLENGER)
                },
                **{
                    f"{s}_score": (
                        float(sub[sub["strategy"] == s]["final_score"].iloc[0])
                        if (sub["strategy"] == s).any() else None
                    )
                    for s in (BASELINE, CHALLENGER)
                },
            })

    blind = pd.DataFrame(blind_rows).sort_values(
        ["우선순위", "프로필", "추천_게임"]
    ).reset_index(drop=True)
    return blind, pd.DataFrame(map_rows)


GUIDE = [
    ("무엇을 판단하나", "'이 사람이 왼쪽 3개 게임을 좋아한다'는 사실만 보고, 오른쪽 추천이 타당한지 봅니다. 게임을 몰라도 장르와 설명만으로 판단하세요."),
    ("", ""),
    ("적합도 3", "매우 타당. 이 사람이 실제로 좋아할 가능성이 높다."),
    ("적합도 2", "타당. 추천에 있어도 이상하지 않다."),
    ("적합도 1", "약함. 장르나 키워드만 겹치고 실제로는 안 맞을 것 같다."),
    ("적합도 0", "부적절. 왜 나왔는지 모르겠다."),
    ("", ""),
    ("이유_태그 (0~1점일 때만)", "왜 나쁜지 한 단어로. 아래 중 하나를 골라 적으세요."),
    ("  GENRE_ONLY", "장르만 같고 실제 플레이 경험이 다름"),
    ("  KEYWORD_MATCH", "설명의 단어만 겹침 (예: '좀비'만 같음)"),
    ("  MODE_MISMATCH", "플레이 방식이 안 맞음 (싱글 vs 멀티, 캐주얼 vs 하드코어)"),
    ("  FRANCHISE_OR_VARIANT", "이미 좋아하는 게임의 시리즈/DLC/유사 변형"),
    ("  TOO_NICHE", "너무 마이너해서 추천으로 부적절"),
    ("  IRRELEVANT", "완전히 무관"),
    ("", ""),
    ("얼마나 채점하나", "우선순위 1(앞쪽 8개 프로필)만 채워도 유효한 결론이 나옵니다. 시간이 되면 우선순위 2까지."),
    ("주의", "행 순서에 의미가 없습니다. 추천 순위와 무관하게 섞여 있으니 순서를 힌트로 쓰지 마세요."),
    ("", ""),
    ("채점 후", "python -m src.eval_human artifacts/human_eval/recommendation_review.xlsx"),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    dataset = pd.read_parquet(DATASET)
    profiles_df = pd.read_parquet(PROFILES)

    print(f"프로필 {len(profiles_df)}개로 최종 추천 생성 중 "
          f"({BASELINE.upper()} 베이스라인 + {CHALLENGER.upper()} 챌린저, rec_boost {REC_BOOST:.0%})...")
    recs = generate_final_recommendations(profiles_df)
    recs.to_parquet(OUT_DIR / "final_recommendations.parquet", index=False)

    blind, mapping = build_blind_sheet(recs, dataset, rng)
    mapping.to_parquet(OUT_DIR / "recommendation_mapping.parquet", index=False)

    xlsx = OUT_DIR / "recommendation_review.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame(GUIDE, columns=["항목", "설명"]).to_excel(w, sheet_name="가이드", index=False)
        blind.to_excel(w, sheet_name="추천평가", index=False)
        _autosize(w.book["가이드"], [28, 95])
        _autosize(w.book["추천평가"], [8, 14, 8] + [22, 20] * 3 + [26, 22, 60, 8, 20, 24])

    n_pri = int((blind["우선순위"] == 1).sum())
    print(f"\n최종 추천: {len(recs)}행 → {OUT_DIR / 'final_recommendations.parquet'}")
    print(f"평가지:   {len(blind)}쌍 ({blind['프로필'].nunique()}개 프로필) → {xlsx}")
    print(f"  우선순위 1: {n_pri}쌍  /  우선순위 2: {len(blind) - n_pri}쌍")
    print(f"매핑:     {OUT_DIR / 'recommendation_mapping.parquet'} (채점 전에는 열지 마세요)")

    overlap = _overlap_stats(recs)
    print(f"\n{BASELINE.upper()} vs {CHALLENGER.upper()} Top-{TOP_K} 겹침: "
          f"평균 {overlap:.1f}/{TOP_K} — 두 전략이 다른 후보를 내는 만큼만 블라인드 비교가 의미 있습니다.")


def _overlap_stats(recs: pd.DataFrame) -> float:
    vals = []
    for _, grp in recs.groupby("profile_id"):
        a = set(grp[grp["strategy"] == BASELINE]["candidate_appid"])
        b = set(grp[grp["strategy"] == CHALLENGER]["candidate_appid"])
        vals.append(len(a & b))
    return float(np.mean(vals))


def _autosize(ws, widths):
    from openpyxl.utils import get_column_letter

    for i, wdt in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = wdt
    ws.freeze_panes = "A2"


if __name__ == "__main__":
    main()
