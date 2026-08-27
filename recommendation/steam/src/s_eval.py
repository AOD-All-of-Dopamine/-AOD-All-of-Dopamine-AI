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
        # 지정하지 않은 축은 **확정값**(config.PRODUCTION)이 들어간다 (D-55).
        # 예전에는 0.0 이 기본이라 "한 축만 스윕"이 실은 "나머지 축을 전부 끈" 측정이었다.
        comps = build_components(quality_w=variant.get("quality_w"),
                                 quality_cap=variant.get("quality_cap"),
                                 quality_src=variant.get("quality_src"),
                                 tag_w=variant.get("tag_w"),
                                 mc_w=variant.get("mc_w"))
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
    """`vec_of(appids) -> (n, d)` 를 주면 프로필별 ILS 열도 채운다 (D-32)."""
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
        ils = np.nan
        if vec_of is not None:
            try: ils = intra_list_similarity(vec_of(list(df[col].head(k))))
            except Exception: pass
        rows.append(dict(profile_id=pid, n=len(gs),
                         fit=float(np.mean([x >= 2 for x in gs])) if gs else np.nan,
                         mean_grade=float(np.mean(gs)) if gs else np.nan,
                         ils=ils))
    return pd.DataFrame(rows), ungraded, miss


def export_blind(pairs, tag: str, profiles=None) -> int:
    """미채점 쌍을 눈가림 시트로 내보낸다 — `(pid, appid)` 목록을 받는다.

    D-24 · D-37 · D-38 에서 같은 코드를 세 번 다시 썼다. 네 번째부터는 여기를 쓴다.
    등급을 숨기고 **시드 · 장르 · 제목 · 소개문**만 보여 준다. 리뷰 수는 **넣지 않는다** —
    D-37/38 이 검증한 것이 리뷰 기반 신호라 시트에 노출하면 순환이 된다.
    **`tags` 도 넣지 않는다** — D-49 가 검증하는 것이 태그 정합이라 같은 이유다.

    **순서를 섞는다(D-48).** D-43/44 에서 프로필별·랭크순으로 냈더니 1,674쌍이
    채점자에게 순위를 흘렸고, 시트 순번 구간별 적합률이 1–5위 0.956 → 21–40위 0.918 로
    기울었다. 폭 0.038 은 판정 문턱(+0.03)과 같은 자릿수라 무시할 수 없다.
    `seed` 는 고정이라 같은 입력이면 같은 시트가 나온다(재현용).

    `artifacts/p1/{tag}_chunks.txt` 와 `{tag}_key.json` 을 쓰고 쌍 수를 돌려준다.
    """
    import json, random
    from src.config import artifact_dir
    profiles = load_profiles() if profiles is None else profiles
    ds = pd.read_parquet(artifact_dir() / "dataset.parquet")
    ds = ds.set_index(ds["steam_appid"].astype(int))
    seeds = {r["profile_id"]: list(r["liked_appids"]) for _, r in profiles.iterrows()}

    def nm(a):
        try:
            return str(ds.loc[int(a), "name"])
        except Exception:
            return f"appid{a}"

    ordered = sorted(pairs)
    random.Random(20260823).shuffle(ordered)      # D-48. 프로필 묶음·순위를 흘리지 않는다
    lines, key = [], []
    for j, (pid, appid) in enumerate(ordered):
        row = ds.loc[int(appid)]
        # **시드를 전부 보여준다 (D-58).** 예전에는 앞 3개만, 그것도 50자로 잘라
        # 보여줬다. `twenty_broad` 는 시드가 20개인데 Terraria/DST/Portal 셋만 보였고,
        # 숨은 시드에 다크 소울·폴아웃4·몬헌·XCOM·데스티니가 있는 줄 모른 채
        # ELDEN RING 을 g=1, Nioh 3 을 g=0 으로 매겼다. 세 플랫폼 모두 같은 편향이
        # 있었다 — 시드 수 구간별 적합률이 TMDB 0.948 → 0.924 → 0.880 → 0.640 으로
        # 단조 감소한다. **잘린 시드는 곧 없는 시드다.**
        sd = " / ".join(nm(x) for x in seeds[pid])
        lines.append(f"{tag}{j:04d} [{sd}] ({row.get('genres', '')}) "
                     f"{nm(appid)} | {str(row.get('short_description', ''))[:94]}")
        key.append(dict(id=f"{tag}{j:04d}", pid=pid, appid=int(appid)))
    (P1 / f"{tag}_chunks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump(key, open(P1 / f"{tag}_key.json", "w"))
    return len(key)


def merge_grades(tag: str, var: str) -> tuple[int, int]:
    """`grades_{tag}.py` 의 `{var}` 를 은행에 합친다. (추가, 충돌) 을 돌려준다."""
    import json
    ns: dict = {}
    exec((P1 / f"grades_{tag}.py").read_text(encoding="utf-8"), ns)
    g = ns[var]
    key = {r["id"]: (r["pid"], int(r["appid"])) for r in json.load(open(P1 / f"{tag}_key.json"))}
    bank = load_bank()
    add = conflict = 0
    for sid, v in g.items():
        k = key[sid]
        if k in bank:
            conflict += bank[k] != v
        else:
            bank[k] = int(v); add += 1
    save_bank(bank)
    return add, conflict
