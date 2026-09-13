"""평가 하네스 — 눈가림 시트 만들기 · 채점 병합 · P@k + ILS.

**규율은 세 플랫폼과 같다**: 사전등록 → md5 봉인 → 눈가림 채점 → 미채점 0 → 판정.
`score()` 는 P@k 와 **ILS(목록 내 유사도)를 함께** 낸다 — P@k 단독은 중복을 보상한다(D-32).
"""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, pandas as pd
from src.config import PROJECT_ROOT
from src.personalized_retrieve import Engine

EVAL = PROJECT_ROOT / "eval"


def intra_list_similarity(vecs: np.ndarray) -> float:
    if vecs is None or len(vecs) < 2: return float("nan")
    M = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    S = M @ M.T
    iu = np.triu_indices(len(M), 1)
    return float(S[iu].mean())


def variant_recs(eng: Engine, profiles: list[dict], k: int, **kw) -> dict:
    return {p["pid"]: eng.recommend(p["seeds"], k=k, **kw) for p in profiles}


def export_blind(recs: dict, profiles: list[dict], eng: Engine, out: Path,
                 head: int = 10, tail_n: int = 10, seed: int = 0) -> list[dict]:
    """눈가림 시트. **셔플한다**(D-48) — 순서가 곧 변형 라벨이 되면 눈가림이 아니다."""
    names = {int(r["item_id"]): r["name"] for _, r in eng.ds.iterrows()}
    meta = eng.ds.set_index("item_id")
    rng = random.Random(seed); rows = []
    for p in profiles:
        df = recs[p["pid"]]
        idx = list(range(min(head, len(df))))
        tail = list(range(head, len(df)))
        idx += rng.sample(tail, min(tail_n, len(tail)))
        for i in idx:
            r = df.iloc[i]; m = meta.loc[int(r["item_id"])]
            m = m.iloc[0] if isinstance(m, pd.DataFrame) else m
            g = m.get("tags"); g = list(g)[:6] if g is not None and len(g) else []
            rows.append(dict(pid=p["pid"], item=int(r["item_id"]), pos=i + 1,
                             part=("top10" if i < head else "tail"),
                             seed=", ".join(names.get(int(s), str(s)) for s in p["seeds"]),
                             cand=str(r["name"]),
                             meta=f"웹툰 · {', '.join(map(str,g)) or '-'} | 관심 {int(m.get('favorite_count',0)):,} · {int(m.get('episode_count',0))}화"))
    rng.shuffle(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, out.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


def merge_grades(rows: list[dict], grade_files: list[Path]) -> tuple[dict, float]:
    """3역할 다수결. 적합 = 등급 ≥2 를 2역할 이상."""
    import collections
    votes = collections.defaultdict(list)
    for f in grade_files:
        for i, g in json.load(open(f, encoding="utf-8")).items():
            votes[i].append(int(g) >= 2)
    maj, unan = {}, 0
    for i, v in votes.items():
        if len(v) >= 3:
            maj[i] = sum(v) >= 2; unan += (len(set(v)) == 1)
    return maj, (unan / max(len(maj), 1))


def score(recs: dict, bank: dict, k: int, eng: Engine | None = None) -> dict:
    """P@k · 구간별 · 미채점 수 · ILS. **미채점이 남으면 비교하지 않는다.**"""
    hit = tot = miss = 0; head = tail = hn = tn = 0; ils = []
    for pid, df in recs.items():
        for i, r in df.head(k).iterrows():
            key = (pid, int(r["item_id"]))
            if key not in bank: miss += 1; continue
            ok = bool(bank[key]); tot += 1; hit += ok
            if i < 10: hn += 1; head += ok
            else: tn += 1; tail += ok
        if eng is not None and len(df):
            rows = [eng.row[int(x)] for x in df.head(k)["item_id"] if int(x) in eng.row]
            if len(rows) > 1: ils.append(intra_list_similarity(eng.emb[rows]))
    return dict(P=hit / tot if tot else float("nan"),
                P_top10=head / hn if hn else float("nan"),
                P_tail=tail / tn if tn else float("nan"),
                n=tot, unscored=miss,
                ILS=float(np.mean(ils)) if ils else float("nan"))
