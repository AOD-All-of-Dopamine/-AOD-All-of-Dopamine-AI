"""Steam 52프로필 × k=50 평가 — D-24 수정용.

**왜 이 파일이 필요한가.** 로그에 5회 인용된 "Steam 0.8488" 은 근거가 없었다
(D-24). 눈가림 라운드 20개 합계 3,643행이 있으나 어느 것도 52프로필 × k=50
단일 측정이 아니다. TMDB(0.8485) · 웹소설(0.9535) 과 같은 조건으로 다시 잰다.

등급 은행은 그 20개 라운드에서 복원했다 — `(profile_id, appid) → 0~3`.
중복 0건, 등급 불일치 0건이라 그대로 쓸 수 있다.

**미채점이 0 이 되기 전에는 어떤 수치도 발표하지 않는다.**
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.personalized_retrieve import build_components, run_multi  # noqa: E402

P1 = Path(__file__).resolve().parents[1] / "artifacts" / "p1"


def load_bank() -> dict:
    f = P1 / "grades.py"
    if not f.exists():
        return {}
    ns: dict = {}
    exec(f.read_text(encoding="utf-8"), ns)
    return {(str(p), int(a)): int(v) for (p, a), v in ns["BANK"].items()}


def save_bank(bank: dict) -> None:
    lines = ["# Steam 등급 은행 — (profile_id, appid) → 0~3", "BANK = {"]
    for (p, a), g in sorted(bank.items(), key=lambda t: (t[0][0], t[0][1])):
        lines.append(f'    ("{p}",{a}): {g},')
    lines.append("}")
    (P1 / "grades.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_profiles(split=None) -> pd.DataFrame:
    p = pd.read_parquet(P1 / "profiles.parquet")
    if split and split not in ("all", ""):
        p = p[p["split"] == split]
    return p.reset_index(drop=True)


def _appid_col(df: pd.DataFrame) -> str:
    for c in ("steam_appid", "appid", "candidate_appid"):
        if c in df.columns:
            return c
    return df.columns[0]


def variant_recs(variant: dict, k: int, profiles: pd.DataFrame, comps=None) -> dict:
    """`comps` 를 주지 않으면 variant 의 quality_w 로 새로 만든다."""
    strat = variant.get("strategy", "top2_mean")
    if comps is None:
        comps = build_components(quality_w=variant.get("quality_w", 0.0),
                                 quality_cap=variant.get("quality_cap", 5.0),
                                 quality_src=variant.get("quality_src", "dataset"))
    out = {}
    for _, p in profiles.iterrows():
        res = run_multi(list(p["liked_appids"]), strategies=[strat], top_n=k,
                        components=comps, postprocess=True,
                        postprocess_kwargs=variant.get("postprocess_kwargs"))
        df = list(res.values())[0]
        if isinstance(df, dict):
            df = df.get("recommendations", df)
            if isinstance(df, dict):
                df = pd.DataFrame(df)
        out[p["profile_id"]] = df.head(k).reset_index(drop=True)
    return out


def score(recs: dict, bank: dict, k: int):
    rows, ungraded, miss = [], 0, []
    for pid, df in recs.items():
        col = _appid_col(df)
        gs = []
        for a in df[col].head(k):
            g = bank.get((pid, int(a)))
            if g is None:
                ungraded += 1
                miss.append((pid, int(a)))
            else:
                gs.append(g)
        rows.append(dict(profile_id=pid, n=len(gs),
                         fit=float(np.mean([x >= 2 for x in gs])) if gs else np.nan,
                         mean_grade=float(np.mean(gs)) if gs else np.nan))
    return pd.DataFrame(rows), ungraded, miss
