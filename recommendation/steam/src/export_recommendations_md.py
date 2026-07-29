# src/export_recommendations_md.py
"""현재 설정으로 뽑은 추천 목록을 마크다운으로 저장한다.

**어떤 상태에서 나온 결과인지**를 문서 머리에 함께 박는다 — 임베딩 아티팩트 지문,
git sha, 후처리 설정, 판정자. 나중에 코퍼스나 표현을 바꿔 다시 뽑았을 때 무엇이 달라졌는지
비교하려면 이 정보가 없으면 안 된다.

    python -m src.export_recommendations_md [--artifacts artifacts/rep_v2] [--out docs/...]
"""
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, artifact_dir
from src.experiment import file_fingerprint
from src.personalized_retrieve import REFRESH_REC_BOOST, build_components, next_page

SCORE_LABEL = {3: "★★★", 2: "★★", 1: "★", 0: "✗"}
LEGEND = "★★★ 매우 타당 · ★★ 타당 · ★ 약함 · ✗ 부적절 · – 미채점"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def load_scores() -> dict:
    """지금까지의 Claude 판정을 모은다 (엑셀 + 실험별 추가분)."""
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "artifacts" / "experiments"))
    from src.eval_human import load_human

    d = load_human(str(PROJECT_ROOT / "artifacts" / "human_eval" / "recommendation_review_claude.xlsx"))
    d = d[d["relevance"].notna()]
    scores = {(r["profile_id"], int(r["candidate_appid"])): int(r["relevance"]) for _, r in d.iterrows()}
    try:
        from judgments_claude import ALL

        scores.update({k: v[0] for k, v in ALL.items()})
    except ImportError:
        pass
    return scores, set(d["profile_id"])


def build_doc(artifacts: str, page_size: int = 10) -> str:
    art = artifact_dir(artifacts)
    ds = pd.read_parquet(art / "dataset.parquet").set_index("steam_appid")
    profiles = pd.read_parquet(PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet")
    scores, judged_pids = load_scores()
    comp = build_components(REFRESH_REC_BOOST, artifacts=artifacts)

    L = [
        "# 추천 결과 스냅샷",
        "",
        f"생성 {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M')} · git `{_git_sha()}`",
        "",
        "## 어떤 상태에서 나온 결과인가",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 코퍼스 | {len(ds):,}개 |",
        f"| 임베딩 아티팩트 | `{art.relative_to(PROJECT_ROOT)}` (지문 `{file_fingerprint(art / 'corpus_embeddings.npy')}`) |",
        "| 임베딩 모델 | Qwen/Qwen3-Embedding-0.6B, 1024차원, L2 정규화 |",
        "| 표현 | `Description` + `Genres` + `Modes`(게임플레이 모드만, 플랫폼 문구 제거) |",
        "| 집계 | MAX (시드별 최대 유사도) |",
        f"| 인기도 부스트 | {REFRESH_REC_BOOST:.0%} |",
        "| 후처리 | 시드 라운드로빈 인터리빙 · 시리즈 상한 1 · hard filter(성인/VR전용/미출시) |",
        "| 품질 하한 | 리뷰 수가 보고되는 게임만 (`has_recommendations`) |",
        "| 판정자 | Claude (Opus 5) — 사람 판정 아님 |",
        f"| 판정 범위 | 20개 프로필 중 {len(judged_pids)}개 |",
        "",
        f"평가 표기: {LEGEND}",
        "",
        "> 이 코퍼스는 Steam 전체(176,609개)의 11%다. 전체 크롤링이 진행 중이며, 완료 후 같은",
        "> 스크립트로 다시 뽑아 비교한다.",
        "",
        "---",
        "",
    ]

    judged_first = sorted(profiles.to_dict("records"),
                          key=lambda p: (p["profile_id"] not in judged_pids, p["profile_id"]))
    for p in judged_first:
        liked = list(p["liked_appids"])
        r = next_page(liked, page_size=page_size, components=comp)
        title = " + ".join(str(ds.loc[a, "name"])[:30] for a in liked)
        tag = "판정됨" if p["profile_id"] in judged_pids else "미판정"
        L += [f"### {title}", "", f"`{p['profile_type']}` · {tag}", "",
              "| # | 추천 게임 | 장르 | 평가 | 리뷰 |", "|---|---|---|---|---|"]
        for t in r.itertuples():
            g = ds.loc[int(t.steam_appid)]
            rev = g["recommendations_total"]
            s = scores.get((p["profile_id"], int(t.steam_appid)))
            L.append("| {} | {} | {} | {} | {} |".format(
                t.rank, str(t.name)[:40], ", ".join(g["genres"])[:26],
                SCORE_LABEL.get(s, "–"), f"{int(rev):,}" if pd.notna(rev) else "–"))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/rep_v2")
    ap.add_argument("--out", default="artifacts/human_eval/recommendations_snapshot.md")
    ap.add_argument("--page-size", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_doc(args.artifacts, args.page_size), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
