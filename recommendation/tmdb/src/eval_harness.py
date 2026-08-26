"""평가 하네스 — 등급 은행 + 변형 비교.

Steam 과 같은 잣대: **52프로필 × k · 적합 = 2점 이상 · 눈가림 채점.**

**등급 은행**이 핵심이다. 변형(λ·boost·후처리)을 바꿔도 후보는 대부분 겹치므로,
`(profile_id, row) → 등급` 을 한 번 매겨 두면 이후 변형은 **새로 등장한 것만** 채점하면 된다.
Steam 이 52프로필 × k=50 을 여러 설정에서 비교할 수 있었던 이유가 이것이다.

등급 척도 (크로스도메인과 동일):
    3 아주 잘 맞음 · 2 맞음 · 1 애매 · 0 안 맞음      적합 = 2 이상
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "artifacts" / "p1"
BANK = P1 / "grades.py"
sys.path.insert(0, str(ROOT))

def load_bank() -> dict:
    if not BANK.exists(): return {}
    ns = {}; exec(BANK.read_text(encoding="utf-8"), ns)
    return {tuple(k.split("\t")): v for k, v in ns["G"].items()}

def save_bank(bank: dict):
    lines = ['"""TMDB 등급 은행 — (profile_id, row) → 0~3. 눈가림 채점."""', "G = {"]
    for (pid, row), g in sorted(bank.items()):
        lines.append(f'"{pid}\\t{row}": {g},')
    lines.append("}")
    BANK.write_text("\n".join(lines), encoding="utf-8")

def variant_recs(variant: dict, k: int, profiles: pd.DataFrame, components=None):
    """변형 하나로 모든 프로필의 top-k 를 만든다."""
    from src.personalized_retrieve import build_components, recommend
    comp = components or build_components(
        hub_lambda=variant.get("hub_lambda", 0.0),
        vote_boost=variant.get("vote_boost", 0.0),
        rating_boost=variant.get("rating_boost", 0.0),
        align_w=variant.get("align_w", 0.0),
        min_overview_len=variant.get("min_overview_len", 0),
        media_w=variant.get("media_w", 0.0),
        genre_w=variant.get("genre_w", 0.0))
    out = {}
    for r in profiles.itertuples(index=False):
        out[r.profile_id] = recommend(
            list(r.seed_rows), components=comp, top_n=k,
            strategy=variant.get("strategy", "top2_mean"),
            postprocess_on=variant.get("postprocess", True),
            postprocess_kwargs=dict(
                franchise_max=variant.get("franchise_max", 1),
                seed_franchise_max=variant.get("seed_franchise_max", 0),
                drop_seed_iter=variant.get("drop_seed_iter"),
                interleave=variant.get("interleave", True),
                tv_max_ratio=variant.get("tv_max_ratio")))
    return out, comp

def intra_list_similarity(vecs) -> float:
    """리스트 내부 유사도(ILS) — top-k 임베딩의 대각 제외 평균 쌍유사도.

    **P@k 하나로는 부족하다(D-32).** 적합률은 "같은 책 10권"을 만점으로 센다.
    웹소설 52프로필 실측에서 corr(적합률, ILS) = **+0.308** — 지표가 중복을
    보상한다. 적합률 1.00 인 rule_rf_none 은 ILS 0.704 인데, 적합률 0.80 인
    coh_talent 는 0.586 으로 **후자가 추천으로서 더 낫다.**

    그래서 적합률과 항상 같이 낸다. 낮을수록 다양하다.
    """
    if vecs is None or len(vecs) < 2:
        return float("nan")
    V = np.asarray(vecs, dtype=np.float32)
    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    S = V @ V.T
    n = len(V)
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))


def score(recs: dict, bank: dict, k: int, vec_of=None):
    """프로필별 P@k (적합률) — 미채점이 있으면 그 개수를 함께 돌려준다.

    `vec_of(rows) -> (n, d)` 를 주면 ILS 열도 채운다 (D-32).
    """
    rows, ungraded = [], 0
    for pid, df in recs.items():
        gs = []
        for row in df["row"].head(k):
            g = bank.get((pid, str(int(row))))
            if g is None: ungraded += 1
            else: gs.append(g)
        ils = np.nan
        if vec_of is not None:
            try: ils = intra_list_similarity(vec_of(list(df["row"].head(k))))
            except Exception: pass
        rows.append(dict(profile_id=pid, n=len(gs),
                         fit=float(np.mean([g >= 2 for g in gs])) if gs else np.nan,
                         mean_grade=float(np.mean(gs)) if gs else np.nan,
                         ils=ils))
    return pd.DataFrame(rows), ungraded
