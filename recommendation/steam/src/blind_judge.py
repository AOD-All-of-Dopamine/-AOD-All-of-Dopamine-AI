# src/blind_judge.py
"""판정을 블라인드로 되돌린다.

**왜 필요한가** — 판정 514쌍 중 블라인드는 127쌍(25%)뿐이다. 나머지 387쌍은 후보를
"어느 설정에서 새로 진입했는지" 아는 상태로 채점했다. 즉 시험 중인 설정을 알면서 매긴
점수로 그 설정을 평가했다. 판정자가 시스템 설계자와 같은 사람(Claude)이므로 이 누수는
그냥 소음이 아니라 방향성 있는 편향이다.

**무엇을 가리는가**

    가림:  어느 설정/변형에서 나왔는가 · 몇 위인가 · 어느 집계 전략인가
           프로필의 실제 id · 프로필 순서 · 후보 등장 순서
    보임:  시드 게임(이름 · 장르) · 후보 게임(이름 · 장르 · 설명)

`show_reviews=False` 가 기본이다. 리뷰 수는 게임 메타데이터이기도 하지만 동시에
`min_reviews` 실험의 조작 변수라, 보이면 "리뷰 150개니까 하한 없는 설정에서 왔겠군"을
추론할 수 있다. 다만 이걸 가리면 기존 `TOO_NICHE`/`LOW_QUALITY` 태그와 판정 근거가
달라지므로, 과거 판정과 섞어 쓸 때는 그 사실을 알고 있어야 한다.

**행 순서를 반드시 섞는다.** 안 섞으면 프로필별로 뭉쳐 나와서 "이건 3페이지짜리 후보군"
같은 맥락이 그대로 보인다.

    python -m src.blind_judge make  --pairs pending.json --out artifacts/blind/round7
    (채점)
    python -m src.blind_judge merge --dir artifacts/blind/round7 --name ROUND7
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, artifact_dir

SHEET = "blind_sheet.csv"
MAPPING = "mapping.parquet"
SCORED = "blind_sheet_scored.csv"

SCORE_COL = "적합도"
TAG_COL = "이유태그"
NOTE_COL = "메모"

RUBRIC = (
    "3 = 매우 타당 (이 사람에게 확실히 맞다) · 2 = 타당 (방향은 맞다) · "
    "1 = 약함 (장르만 겹치고 경험이 다르다) · 0 = 부적절 (나오면 안 된다)"
)
TAGS = ("IRRELEVANT", "GENRE_ONLY", "KEYWORD_MATCH", "MODE_MISMATCH",
        "FRANCHISE_OR_VARIANT", "TOO_NICHE", "LOW_QUALITY")


def _game_info(dataset: pd.DataFrame) -> dict[int, dict]:
    d = dataset.set_index("steam_appid") if "steam_appid" in dataset.columns else dataset
    return {
        int(a): {
            "name": str(r["name"]),
            "genres": ", ".join(r["genres"]) if isinstance(r["genres"], (list, np.ndarray)) else "",
            "desc": " ".join(str(r["short_description"]).split())[:300],
        }
        for a, r in d.iterrows()
    }


def build_sheet(
    pairs: list[tuple[str, int]],
    profiles: pd.DataFrame,
    dataset: pd.DataFrame,
    seed: int = 42,
    show_reviews: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(프로필, 후보) 쌍 목록 → (블라인드 시트, 복원 매핑).

    같은 쌍이 여러 설정에서 나와도 한 번만 채점한다 — 판정은 (프로필, 게임)의 함수이지
    설정의 함수가 아니다.
    """
    info = _game_info(dataset)
    prof = profiles.set_index("profile_id")
    rng = np.random.default_rng(seed)

    uniq = list(dict.fromkeys((str(p), int(a)) for p, a in pairs))
    pids = sorted({p for p, _ in uniq})
    # 프로필 익명 id 도 섞어서 부여한다 — 정의 순서대로 주면 P01=coh_fps 를 유추할 수 있다.
    shuffled_pids = list(pids)
    rng.shuffle(shuffled_pids)
    anon = {pid: f"A{i + 1:02d}" for i, pid in enumerate(shuffled_pids)}

    rows, maps = [], []
    for pid, appid in uniq:
        liked = list(prof.loc[pid, "liked_appids"])
        g = info.get(appid, {"name": str(appid), "genres": "", "desc": ""})
        row = {
            "pair_key": "",                                   # 아래에서 섞은 뒤 부여
            "프로필": anon[pid],
            "좋아하는_게임": " / ".join(info.get(a, {}).get("name", str(a)) for a in liked),
            "좋아하는_게임_장르": " / ".join(info.get(a, {}).get("genres", "") for a in liked),
            "추천_게임": g["name"],
            "추천_게임_장르": g["genres"],
            "추천_게임_설명": g["desc"],
            SCORE_COL: "",
            TAG_COL: "",
            NOTE_COL: "",
        }
        if show_reviews:
            rec = dataset.set_index("steam_appid")["recommendations_total"]
            v = rec.get(appid)
            row["리뷰수"] = int(v) if pd.notna(v) else 0
        rows.append(row)
        maps.append({"profile_id": pid, "candidate_appid": appid, "anon_profile_id": anon[pid]})

    sheet = pd.DataFrame(rows)
    mapping = pd.DataFrame(maps)
    order = rng.permutation(len(sheet))            # 행 순서를 섞는다 — 프로필별 뭉침 제거
    sheet = sheet.iloc[order].reset_index(drop=True)
    mapping = mapping.iloc[order].reset_index(drop=True)
    keys = [f"B{i + 1:04d}" for i in range(len(sheet))]
    sheet["pair_key"] = keys
    mapping["pair_key"] = keys
    return sheet, mapping


def leaks(sheet: pd.DataFrame) -> list[str]:
    """시트에 남으면 안 되는 컬럼을 찾는다. 이 함수가 통과해야 시트를 내보낸다."""
    banned = {"profile_id", "candidate_appid", "rank", "strategy", "config", "variant",
              "split", "profile_type", "seed_similarity", "final_score", "exp_id",
              "dominant_seed", "profile_order"}
    return sorted(banned & set(sheet.columns))


def write(out_dir: Path, sheet: pd.DataFrame, mapping: pd.DataFrame) -> None:
    bad = leaks(sheet)
    if bad:
        raise ValueError(f"블라인드 시트에 정체를 드러내는 컬럼이 있습니다: {bad}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out_dir / SHEET, index=False)
    mapping.to_parquet(out_dir / MAPPING, index=False)


def merge(out_dir: Path, name: str = "BLIND") -> tuple[dict[tuple[str, int], tuple[int, str]], str]:
    """채점된 시트를 매핑과 조인해 판정 dict 와 judgments_claude.py 에 붙일 코드를 낸다."""
    scored = pd.read_csv(out_dir / SCORED)
    mapping = pd.read_parquet(out_dir / MAPPING)
    j = scored.merge(mapping, on="pair_key", how="inner", validate="one_to_one")

    unscored = j[j[SCORE_COL].isna()]
    if len(unscored):
        raise ValueError(f"미채점 {len(unscored)}행이 남아 있습니다 (예: {unscored['pair_key'].head(3).tolist()})")

    bad = j[~j[SCORE_COL].astype(int).isin([0, 1, 2, 3])]
    if len(bad):
        raise ValueError(f"0~3 밖의 점수: {bad['pair_key'].tolist()[:5]}")

    out, lines = {}, [f"{name} = {{"]
    for _, r in j.sort_values(["profile_id", "candidate_appid"]).iterrows():
        score = int(r[SCORE_COL])
        tag = "" if pd.isna(r.get(TAG_COL)) or score >= 2 else str(r[TAG_COL]).strip()
        out[(r["profile_id"], int(r["candidate_appid"]))] = (score, tag)
        lines.append(f'    ("{r["profile_id"]}", {int(r["candidate_appid"])}): '
                     f'({score}, "{tag}"),  # {str(r["추천_게임"])[:40]}')
    lines.append("}")
    return out, "\n".join(lines)


def pending_pairs(rankers: dict[str, object], profile_ids: list[str],
                  judgments: dict, k: int = 10) -> list[tuple[str, int]]:
    """비교할 설정들의 Top-k **합집합** 중 아직 판정 안 된 쌍.

    설정 하나만 판정하면 그 설정이 자동으로 이긴다 — 실측으로 기준선 미판정 3칸 vs
    대안 55~115칸이 나왔고, 미판정을 성공으로 세면 순위가 뒤집혔다. 합집합을 먼저 만들고
    전부 채우는 것이 비교를 성립시키는 유일한 방법이다.
    """
    profiles = pd.read_parquet(PROFILES_PATH := PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet")
    prof = profiles.set_index("profile_id")
    need = []
    for pid in profile_ids:
        liked = list(prof.loc[pid, "liked_appids"])
        for rank in rankers.values():
            for a in list(rank(pid, liked)["steam_appid"][:k]):
                if (pid, int(a)) not in judgments:
                    need.append((pid, int(a)))
    return list(dict.fromkeys(need))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "merge"])
    ap.add_argument("--pairs", help="make: [[profile_id, appid], ...] JSON")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--artifacts", default="artifacts/full_v1")
    ap.add_argument("--name", default="BLIND")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-reviews", action="store_true")
    args = ap.parse_args()

    out = Path(args.dir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    if args.cmd == "make":
        pairs = [(p, int(a)) for p, a in json.loads(Path(args.pairs).read_text())]
        profiles = pd.read_parquet(PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet")
        dataset = pd.read_parquet(artifact_dir(args.artifacts) / "dataset.parquet")
        sheet, mapping = build_sheet(pairs, profiles, dataset, args.seed, args.show_reviews)
        write(out, sheet, mapping)
        print(f"블라인드 시트 {len(sheet)}행 → {out / SHEET}")
        print(f"매핑(채점 전 열지 말 것)   → {out / MAPPING}")
        print(f"\n채점 규칙: {RUBRIC}")
        print(f"실패 태그: {', '.join(TAGS)}")
        print(f"\n채점 후 {SCORED} 로 저장하고 `merge` 를 실행하세요.")
    else:
        _, code = merge(out, args.name)
        print(code)


if __name__ == "__main__":
    main()
