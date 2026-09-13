# src/evaluate.py
import json
import sys

import pandas as pd

from src.config import ensure_artifacts_dir, load_config
from src.metrics import DEFAULT_GAIN, dcg, ndcg_at_k, precision_at_k


def per_anchor_metrics(df: pd.DataFrame, k: int = 10, mode: str = DEFAULT_GAIN) -> pd.DataFrame:
    rows = []
    for appid, grp in df.groupby("anchor_steam_appid"):
        rels = grp.sort_values("rank")["relevance"].astype(int).tolist()
        rows.append({
            "anchor_steam_appid": appid,
            "precision": precision_at_k(rels, k),
            "ndcg": ndcg_at_k(rels, k, mode=mode),
            "avg_relevance": sum(rels) / len(rels),
            "avg_confidence": grp["recommendation_confidence"].astype(int).mean(),
        })
    return pd.DataFrame(rows)


def compute_summary(per_anchor: pd.DataFrame) -> dict:
    return {
        "precision_at_10": float(per_anchor["precision"].mean()),
        "ndcg_at_10": float(per_anchor["ndcg"].mean()),
        "avg_relevance": float(per_anchor["avg_relevance"].mean()),
        "avg_confidence": float(per_anchor["avg_confidence"].mean()),
    }


def build_comparison(summaries: dict, cfg: dict) -> dict:
    """baseline 대비 treatment 델타 + 품질 게이트.

    실험 ID가 판정 데이터에 없으면 **실패한다.** 예전에는 `_v1` 리터럴을 찾다가 실험이 `_v2`로
    올라간 뒤 조용히 스킵됐고, 그 결과 품질 게이트가 한 번도 산출된 적이 없었다.
    """
    ev = cfg["evaluation"]
    base_id, treat_id = ev["baseline_experiment_id"], ev["treatment_experiment_id"]
    missing = [i for i in (base_id, treat_id) if i not in summaries]
    if missing:
        raise SystemExit(
            f"판정 데이터에 실험 ID가 없습니다: {missing}\n"
            f"  판정 파일에 있는 ID: {sorted(summaries)}\n"
            f"  configs 의 evaluation.baseline_experiment_id / treatment_experiment_id 를 확인하세요."
        )
    base, treat = summaries[base_id], summaries[treat_id]
    gate = float(ev["precision_gate"])
    return {
        "baseline_experiment_id": base_id,
        "treatment_experiment_id": treat_id,
        "precision_delta": treat["precision_at_10"] - base["precision_at_10"],
        "ndcg_delta": treat["ndcg_at_10"] - base["ndcg_at_10"],
        "treatment_beats_baseline": (
            treat["precision_at_10"] > base["precision_at_10"]
            and treat["ndcg_at_10"] > base["ndcg_at_10"]
        ),
        "precision_gate": gate,
        "quality_gate_passed": treat["precision_at_10"] >= gate,
    }


def assert_fully_judged(judgments: pd.DataFrame, xlsx: str) -> None:
    """미판정 행이 있으면 명확히 실패한다.

    예전에는 여기서 `astype(int)` 가 pandas 내부에서 IntCastingNaNError 로 터져
    "판정을 아직 안 했다"는 사실이 드러나지 않았다.
    """
    n_missing = int(judgments["relevance"].isna().sum())
    if not n_missing:
        return
    per_exp = (
        judgments.assign(_judged=judgments["relevance"].notna())
        .groupby("experiment_id")["_judged"]
        .agg(["size", "sum"])
    )
    detail = "\n".join(
        f"    {eid}: {int(r['sum'])}/{int(r['size'])} 판정됨" for eid, r in per_exp.iterrows()
    )
    raise SystemExit(
        f"{xlsx}: {n_missing}/{len(judgments)} 행이 미판정입니다.\n{detail}\n"
        f"  Judgments 시트의 relevance / recommendation_confidence 를 채운 뒤 다시 실행하세요."
    )


def main():
    xlsx = sys.argv[1] if len(sys.argv) > 1 else "artifacts/s1/evaluation.xlsx"
    cfg = load_config()
    mode = cfg["evaluation"]["ndcg_gain"]
    out = ensure_artifacts_dir()
    judgments = pd.read_excel(xlsx, sheet_name="Judgments")
    assert_fully_judged(judgments, xlsx)
    anchors = pd.read_parquet(out / "anchors_40.parquet")

    summaries, seg_rows, genre_rows, per_all = {}, [], [], []
    for exp_id, grp in judgments.groupby("experiment_id"):
        per = per_anchor_metrics(grp, mode=mode)
        per["experiment_id"] = exp_id
        per = per.merge(
            anchors[["steam_appid", "bucket", "primary_stratum"]],
            left_on="anchor_steam_appid", right_on="steam_appid",
        )
        summaries[exp_id] = compute_summary(per)
        per_all.append(per)
        for bucket, g in per.groupby("bucket"):
            seg_rows.append({"experiment_id": exp_id, "bucket": bucket, **compute_summary(g)})
        for stratum, g in per.groupby("primary_stratum"):
            genre_rows.append({"experiment_id": exp_id, "stratum": stratum, **compute_summary(g)})

    result = {
        "ndcg_gain": mode,
        "experiments": summaries,
        "comparison": build_comparison(summaries, cfg),
    }
    suffix = "_pilot" if "pilot" in xlsx else ""
    with open(out / f"metrics{suffix}.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(seg_rows).to_csv(out / f"metrics_by_segment{suffix}.csv", index=False)
    pd.DataFrame(genre_rows).to_csv(out / f"metrics_by_genre{suffix}.csv", index=False)
    pd.concat(per_all).to_csv(out / f"metrics_by_anchor{suffix}.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
