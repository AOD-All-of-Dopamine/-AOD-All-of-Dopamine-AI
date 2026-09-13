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
    """변형 하나로 모든 프로필의 top-k 를 만든다.

    **후처리 기본값은 `PRODUCTION_POSTPROCESS` 에서 가져온다** (D-55). 여기서 각
    항목의 기본값을 따로 적으면 서빙과 갈라진다 — 실제로 D-46 을 확정한 직후
    하네스는 `drop_seed_iter` 를 끈 채로 재고 있었다(0.9346 vs 서빙 0.9385).
    같은 사고가 `align_w` 에서 한 번 있었다(D-43 verdict 부수 결함).
    변형이 명시한 키만 덮어쓴다.
    """
    from src.personalized_retrieve import build_components, recommend
    from src.config import PRODUCTION, PRODUCTION_POSTPROCESS
    # **미지정 축은 확정값이 들어간다** — 예전에는 여기 기본이 리터럴 "top2_mean" 이라
    # 확정값 "mean" 과 어긋났다. `--variant "{}"` 로 부르면(eval_product·export_eval 의
    # 기본값이다) 하네스가 서빙과 **다른 접기**로 재게 된다. 웹소설이 D-66 에서 고친 것과
    # 같은 종류이고, TMDB 에만 남아 있었다.
    # **정렬 축도 마찬가지다.** 위 독스트링이 인용한 D-55·D-46·D-43 의 교훈이
    # 후처리에만 적용되고 여기 정렬 축에는 적용되지 않은 채로 남아 있었다 —
    # 기본이 전부 0.0 이라 `variant_recs({"genre_w": 0.5})` 같은 부분 변형은
    # **나머지 확정 축(hub_lambda 0.35 · rating_boost 0.15 · media_w 0.20 · genre_w 0.40)을
    # 전부 끈 채로** 재고 있었다. Steam 은 D-55, 웹소설은 D-66 에서 고쳤고 TMDB 만 남아 있었다.
    def _ax(key, fallback=0.0):
        return variant.get(key, PRODUCTION.get(key, fallback))
    comp = components or build_components(
        hub_lambda=_ax("hub_lambda"),
        vote_boost=_ax("vote_boost"),
        rating_boost=_ax("rating_boost"),
        align_w=_ax("align_w"),
        min_overview_len=_ax("min_overview_len", 0),
        media_w=_ax("media_w"),
        genre_w=_ax("genre_w"),
        vote_w=_ax("vote_w"))
    out = {}
    for r in profiles.itertuples(index=False):
        out[r.profile_id] = recommend(
            list(r.seed_rows), components=comp, top_n=k,
            strategy=variant.get("strategy", PRODUCTION["strategy"]),
            postprocess_on=variant.get("postprocess", True),
            postprocess_kwargs={
                **PRODUCTION_POSTPROCESS,
                **{key: variant[key] for key in
                   ("franchise_max", "seed_franchise_max", "drop_seed_iter",
                    "interleave", "tv_max_ratio") if key in variant}})
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
