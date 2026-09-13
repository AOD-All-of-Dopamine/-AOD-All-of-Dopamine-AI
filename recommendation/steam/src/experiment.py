# src/experiment.py
"""실험 기록 · 비교.

append-only JSONL 레지스트리 하나로 관리한다. 외부 서비스도 DB 도 쓰지 않는다 —
이 프로젝트는 오프라인 배치 실험이고 산출물이 전부 파일이라 그게 맞다.

**지표를 두 종류로 나눠 담는 것이 이 모듈의 핵심이다.**

  diagnostics : 판정이 필요 없다. 임베딩만 있으면 계산된다.
                표현을 바꿔도 **언제나 서로 비교할 수 있다.**
  judged      : 사람/LLM 판정이 필요하다. 추천 결과가 바뀌면 판정 풀도 바뀌므로
                **같은 풀에서 나온 것끼리만 비교할 수 있다.** compare() 가 이걸 검사한다.

    python -m src.experiment list
    python -m src.experiment show   <exp_id>
    python -m src.experiment compare <exp_id> <exp_id> ...
    python -m src.experiment diagnose <exp_id> --note "..."   # 현재 아티팩트로 기록 생성
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, artifact_dir, load_config

EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
REGISTRY = EXP_DIR / "registry.jsonl"

# 변별력을 추적할 기준 게임. 취향 축이 서로 다른 것으로 골랐다.
REFERENCE_ANCHORS = {
    367520: "Hollow Knight",     # 지금 가장 변별력이 낮은 사례
    504230: "Celeste",
    413150: "Stardew Valley",    # 지금 잘 되는 사례
    730: "Counter-Strike 2",
}
SIM_SAMPLE = 400  # 코퍼스 평균 유사도 추정 표본


# ------------------------------------------------------------- 레지스트리

def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def file_fingerprint(path: Path) -> str:
    """산출물이 바뀌었는지 확인용. 어떤 임베딩으로 낸 숫자인지 추적한다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()[:12]


def record(exp: dict) -> None:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    exp.setdefault("git_sha", _git_sha())
    with open(REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(exp, ensure_ascii=False) + "\n")
    print(f"기록됨: {exp['exp_id']} → {REGISTRY}")


def load_all() -> list[dict]:
    if not REGISTRY.exists():
        return []
    with open(REGISTRY, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get(exp_id: str) -> dict:
    for e in reversed(load_all()):  # 같은 id 가 여러 번이면 최신
        if e["exp_id"] == exp_id:
            return e
    raise SystemExit(f"실험 없음: {exp_id} (있는 것: {[e['exp_id'] for e in load_all()]})")


# --------------------------------------------------------------- 진단 지표

def compute_diagnostics(
    embeddings_path: Path, index_path: Path, seed: int = 42
) -> dict:
    """판정 없이 계산되는 지표. 표현을 바꿔도 서로 비교할 수 있다."""
    E = np.asarray(np.load(embeddings_path, mmap_mode="r"))
    idx = pd.read_parquet(index_path)
    row_of = dict(zip(idx["steam_appid"], idx["embedding_row"]))

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(E), size=min(SIM_SAMPLE, len(E)), replace=False)
    sims = E[picks] @ E.T
    # 자기 자신(=1.0) 제외
    for i, p in enumerate(picks):
        sims[i, p] = np.nan
    mean_sim = float(np.nanmean(sims))
    median_sim = float(np.nanmedian(sims))

    spread = {}
    for appid, label in REFERENCE_ANCHORS.items():
        if appid not in row_of:
            continue
        s = np.sort(E @ E[row_of[appid]])[::-1]
        spread[label] = {
            "top1": round(float(s[1]), 4),
            "top10": round(float(s[10]), 4),
            "top100": round(float(s[100]), 4),
            # 이 값이 클수록 "1등과 100등이 확실히 다르다" = 변별력이 있다
            "spread_1_to_100": round(float(s[1] - s[100]), 4),
        }

    return {
        "corpus_mean_similarity": round(mean_sim, 4),
        "corpus_median_similarity": round(median_sim, 4),
        "anchor_spread": spread,
        "n_corpus": int(len(E)),
        "dim": int(E.shape[1]),
    }


def compute_ranking_diagnostics(
    artifacts: str | Path | None = None,
    postprocess: bool = False,
    ks: tuple[int, ...] = (10, 30, 100),
    page_size: int = 10,
) -> dict:
    """추천 목록의 다양성 — 판정 불필요. 새로고침 제품의 핵심 지표다.

    `max_seed_share` : 한 시드가 먹은 비율. 사용자가 "내가 고른 3개 중 하나만 반영됐다"고
                       느끼는 정도. 시드 3개면 이론 하한 0.33.
    `effective_genres`: exp(장르 엔트로피) = "사실상 몇 개 장르에 걸쳐 있나".
                       max_genre_share 는 쓰지 않는다 — `인디`가 코퍼스의 71.5% 에 붙어 있어
                       우리 랭킹이 아니라 코퍼스 구성을 재는 지표가 되어버린다.
    `unique_series`   : 서로 다른 시리즈 비율.
    `page_turnover`   : 연속한 두 페이지의 시드 **구성** 차이(0~1).
                        ⚠ 단독으로 높/낮음을 판단하면 안 된다. 인터리빙을 걸면 모든 페이지가
                        같은 균형 구성(예: 3/3/4)이 되어 이 값이 **내려간다** — 그건 정상이다.
                        `max_seed_share@10`(페이지 내부 다양성)과 **함께** 읽어야 한다.
                        이상적인 상태는 seed_share 낮음 + turnover 낮음이다.
    """
    import numpy as np

    from src.personalized_retrieve import build_components, run_multi
    from src.postprocess import series_key

    profiles = pd.read_parquet(PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet")
    dataset = pd.read_parquet(artifact_dir(artifacts) / "dataset.parquet").set_index("steam_appid")
    comp = build_components(0.03, artifacts=artifacts)

    acc = {k: {"seed": [], "genres": [], "series": []} for k in ks}
    turnover = []
    for _, p in profiles.iterrows():
        r = run_multi(
            list(p["liked_appids"]), strategies=["max"], top_n=max(ks),
            components=comp, postprocess=postprocess,
        )["max"]
        for k in ks:
            t = r.head(k)
            if not len(t):
                continue
            acc[k]["seed"].append(t["dominant_seed"].value_counts().iloc[0] / len(t))
            counts = pd.Series(
                [g for a in t["steam_appid"] for g in dataset.loc[a, "genres"]]
            ).value_counts()
            probs = counts / counts.sum()
            acc[k]["genres"].append(float(np.exp(-(probs * np.log(probs)).sum())))
            acc[k]["series"].append(
                len({series_key(dataset.loc[a, "name"]) for a in t["steam_appid"]}) / len(t)
            )
        # 페이지 간 시드 구성 변화
        pages = [r.iloc[i:i + page_size] for i in range(0, min(len(r), max(ks)), page_size)]
        diffs = []
        for a, b in zip(pages, pages[1:]):
            sa = a["dominant_seed"].value_counts(normalize=True)
            sb = b["dominant_seed"].value_counts(normalize=True)
            idx = sa.index.union(sb.index)
            diffs.append(float(sa.reindex(idx, fill_value=0).sub(sb.reindex(idx, fill_value=0)).abs().sum() / 2))
        if diffs:
            turnover.append(float(np.mean(diffs)))

    out = {"page_turnover": round(float(np.mean(turnover)), 4) if turnover else None}
    for k in ks:
        out[f"max_seed_share@{k}"] = round(float(np.mean(acc[k]["seed"])), 4)
        out[f"effective_genres@{k}"] = round(float(np.mean(acc[k]["genres"])), 4)
        out[f"unique_series@{k}"] = round(float(np.mean(acc[k]["series"])), 4)
    return out


def compute_loo(rec_boost: float = 0.03) -> dict:
    """leave-one-out — 판정 없이 도는 회귀 지표. 표현 비교에 그대로 쓸 수 있다."""
    from src.eval_personalization import evaluate_leave_one_out, summarize_leave_one_out

    profiles = pd.read_parquet(PROJECT_ROOT / "artifacts" / "p1" / "profiles.parquet")
    loo = evaluate_leave_one_out(profiles, rec_boost=rec_boost)
    s = summarize_leave_one_out(loo)
    return {
        r["strategy"]: {k: float(r[k]) for k in s.columns if k not in ("strategy", "n")}
        for _, r in s.iterrows()
    }


# ------------------------------------------------------------------ 비교

def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, f"{key}."))
        else:
            out[key] = v
    return out


def compare(exp_ids: list[str]) -> None:
    exps = [get(e) for e in exp_ids]

    print("=" * 78)
    print("실험 개요")
    print("=" * 78)
    for e in exps:
        print(f"  {e['exp_id']:24s} {e.get('created_at','')[:16]}  git:{e.get('git_sha','')}")
        print(f"    {e.get('change','(설명 없음)')}")
    print()

    _section("진단 지표 (판정 불필요 — 항상 비교 가능)", exps, "diagnostics")

    # 판정 지표는 풀이 같을 때만 의미가 있다
    pools = {e["exp_id"]: (e.get("judged") or {}).get("pool_id") for e in exps}
    judged = [e for e in exps if e.get("judged")]
    print("=" * 78)
    print("판정 지표 (같은 판정 풀에서 나온 것끼리만 비교 가능)")
    print("=" * 78)
    if len(judged) < 2:
        print("  판정 기록이 2개 미만입니다.")
    else:
        distinct = {pools[e["exp_id"]] for e in judged}
        if len(distinct) > 1:
            print("  ⚠️  판정 풀이 서로 다릅니다 — 아래 숫자를 직접 비교하지 마세요.")
            for e in judged:
                print(f"      {e['exp_id']:24s} pool={pools[e['exp_id']]}")
            print("      표현을 바꾸면 Top-10 이 바뀌고 새 후보는 판정된 적이 없습니다.")
            print("      비교하려면 새 Top-10 의 미판정분을 채점해 풀을 확장해야 합니다.")
            print()
    _section("", judged, "judged")


def _section(title: str, exps: list[dict], key: str) -> None:
    if title:
        print("=" * 78)
        print(title)
        print("=" * 78)
    if not exps:
        return
    rows = {}
    for e in exps:
        for k, v in _flat(e.get(key) or {}).items():
            rows.setdefault(k, {})[e["exp_id"]] = v
    if not rows:
        print("  (없음)")
        print()
        return

    ids = [e["exp_id"] for e in exps]
    w = max(len(k) for k in rows) + 2
    print(f"  {'지표'.ljust(w)}" + "".join(f"{i:>18s}" for i in ids) + "      변화")
    print("  " + "-" * (w + 18 * len(ids) + 10))
    for k, vals in rows.items():
        line = f"  {k.ljust(w)}"
        nums = []
        for i in ids:
            v = vals.get(i)
            if isinstance(v, (int, float)):
                nums.append(float(v))
                line += f"{v:>18.4f}" if isinstance(v, float) else f"{v:>18d}"
            else:
                nums.append(None)
                line += f"{str(v)[:16]:>18s}"
        if len(nums) >= 2 and nums[0] is not None and nums[-1] is not None and nums[0] != 0:
            d = nums[-1] - nums[0]
            line += f"   {d:+.4f} ({d / abs(nums[0]) * 100:+.1f}%)"
        print(line)
    print()


# ------------------------------------------------------------------- CLI

def _arg(flag: str, default=None):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

    if cmd == "list":
        for e in load_all():
            j = "판정있음" if e.get("judged") else "-"
            print(f"  {e['exp_id']:24s} {e.get('created_at','')[:16]}  {j:8s} {e.get('change','')[:44]}")

    elif cmd == "show":
        print(json.dumps(get(sys.argv[2]), ensure_ascii=False, indent=2))

    elif cmd == "compare":
        ids = [a for a in sys.argv[2:] if not a.startswith("--")]
        compare(ids)

    elif cmd == "diagnose":
        exp_id = sys.argv[2]
        art = Path(_arg("--artifacts", str(ARTIFACTS_DIR)))
        if not art.is_absolute():
            art = PROJECT_ROOT / art
        emb, idx = art / "corpus_embeddings.npy", art / "corpus_index.parquet"
        cfg = load_config()
        print(f"진단 중: {emb}")
        diag = compute_diagnostics(emb, idx)
        exp = {
            "exp_id": exp_id,
            "created_at": _arg("--at", ""),
            "change": _arg("--note", ""),
            "config": {
                "representation": _arg("--representation", "description+genres+categories"),
                "model": cfg["embedding"]["model_name"],
                "max_seq_length": cfg["runtime"].get("max_seq_length"),
                "aggregation": "max",
                "rec_boost": 0.03,
            },
            "assets": {
                "corpus_embeddings": str(emb.relative_to(PROJECT_ROOT)),
                "fingerprint": file_fingerprint(emb),
            },
            "diagnostics": diag,
        }
        if "--with-loo" in sys.argv:
            print("LOO 계산 중 (60회 추천)...")
            exp["diagnostics"]["loo"] = compute_loo()
        record(exp)
        print(json.dumps(diag, ensure_ascii=False, indent=2))

    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
