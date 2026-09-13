import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, load_config, resolve_path

DATASET = ARTIFACTS_DIR / "dataset.parquet"
TREND_DIR = PROJECT_ROOT / "artifacts" / "trend_v1"

FUTURE_RELEASE_DATE_PATTERNS: list[str] = ["출시 예정"]

_KR_DATE_RE = re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일")

FRESHNESS_WEIGHTS: dict[str, float] = {
    "0_90d": 1.0,
    "91_365d": 0.75,
    "1_3y": 0.35,
    "3y_plus": 0.0,
}


def parse_release_date(value) -> datetime | None:
    """출시일 파싱. 미출시면 None.

    두 형식을 모두 받는다:
      · dict  — `{"coming_soon": bool, "date": "2000년 11월 1일"}` (Steam appdetails 원형)
      · str   — `"2000년 11월 1일"` (구 jsonl 이 평탄화해둔 형태)

    dict 쪽이 정확하다. `coming_soon` 은 Steam 이 직접 주는 플래그라, 문자열에서
    `"출시 예정"` 접두어를 찾는 것보다 언어·표기 변화에 안전하다.
    """
    if isinstance(value, dict):
        if value.get("coming_soon"):
            return None
        value = value.get("date") or ""

    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    for pat in FUTURE_RELEASE_DATE_PATTERNS:
        if s.startswith(pat):
            return None
    m = _KR_DATE_RE.match(s)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def assign_age_bucket(age_days: int) -> str:
    if age_days <= 90:
        return "0_90d"
    if age_days <= 365:
        return "91_365d"
    if age_days <= 1095:
        return "1_3y"
    return "3y_plus"


def build_release_date_map() -> dict[int, object]:
    """appid → release_date 원본(dict 또는 str). 파싱은 parse_release_date 가 한다."""
    mapping: dict[int, object] = {}
    raw_data = resolve_path(load_config()["data"]["input_path"])
    with open(raw_data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = r.get("steam_appid")
            if isinstance(aid, int):
                mapping[aid] = r.get("release_date") or ""
    return mapping


def build_full_features(
    as_of_date: str = "2026-07-23",
    min_cohort_size: int = 50,
    dataset_path: Path | None = None,
) -> pd.DataFrame:
    """나이 코호트 보정 인기도. 절대 인기도가 아니라 **또래 대비 초과분**을 잰다.

    `dataset_path` 의 dataset 이 `release_date`/`coming_soon` 컬럼을 들고 있으면 그것을 쓴다.
    없으면 원본 jsonl 을 훑는다(`build_release_date_map`). 전체 코퍼스(173,691)를 다루면서
    dataset 이 직접 들고 있게 됐으므로, 이제 원본 jsonl 없이도 돈다 —
    구 구현은 jsonl 을 강제로 읽어서 이 환경에서 실행 자체가 안 됐다.
    """
    as_of = datetime.fromisoformat(as_of_date)

    dataset = pd.read_parquet(dataset_path or DATASET)

    df = dataset[["steam_appid", "name", "has_recommendations", "recommendations_total"]].copy()

    if "release_date" in dataset.columns and "coming_soon" in dataset.columns:
        raw = pd.Series(
            [{"coming_soon": bool(c), "date": d or ""}
             for c, d in zip(dataset["coming_soon"], dataset["release_date"])],
            index=dataset.index,
        )
    else:
        raw = df["steam_appid"].map(build_release_date_map())
    df["parsed_date"] = raw.apply(parse_release_date)
    # dict 는 parquet 에 그대로 못 넣는다 — 표시용 문자열과 플래그로 분리해 저장한다
    df["coming_soon"] = raw.apply(lambda v: bool(v.get("coming_soon")) if isinstance(v, dict) else
                                  (isinstance(v, str) and v.strip().startswith(tuple(FUTURE_RELEASE_DATE_PATTERNS))))
    df["raw_release_date"] = raw.apply(
        lambda v: (v.get("date") or "") if isinstance(v, dict) else (v if isinstance(v, str) else "")
    )
    df["has_valid_date"] = df["parsed_date"].notna()

    df["age_days"] = None
    df["age_bucket"] = None
    mask = df["has_valid_date"]
    df.loc[mask, "age_days"] = (as_of - df.loc[mask, "parsed_date"]).dt.days
    df.loc[mask, "age_bucket"] = df.loc[mask, "age_days"].apply(assign_age_bucket)

    known = df["has_recommendations"] & df["has_valid_date"]
    n_known = known.sum()

    df["global_rec_percentile"] = 0.0
    if n_known > 0:
        known_pct = df.loc[known, "recommendations_total"].rank(pct=True, ascending=True)
        df.loc[known, "global_rec_percentile"] = known_pct

    df["age_cohort_rec_percentile"] = np.nan
    df["cohort_known_count"] = 0

    for bucket in ["0_90d", "91_365d", "1_3y", "3y_plus"]:
        in_bucket = known & (df["age_bucket"] == bucket)
        n_bucket = in_bucket.sum()
        df.loc[known & (df["age_bucket"] == bucket), "cohort_known_count"] = n_bucket

        if n_bucket >= min_cohort_size:
            cohort_pct = df.loc[in_bucket, "recommendations_total"].rank(pct=True, ascending=True)
            df.loc[in_bucket, "age_cohort_rec_percentile"] = cohort_pct

    trend_excess = np.maximum(
        0.0,
        df["age_cohort_rec_percentile"].fillna(0.0)
        - df["global_rec_percentile"],
    )
    df["trend_excess"] = trend_excess

    df["freshness_weight"] = df["age_bucket"].map(FRESHNESS_WEIGHTS).fillna(0.0)

    df["trend_signal"] = df["trend_excess"] * df["freshness_weight"]

    return df


def build_output_features(full: pd.DataFrame) -> pd.DataFrame:
    out = full[
        [
            "steam_appid",
            "name",
            "raw_release_date",
            "coming_soon",
            "parsed_date",
            "age_days",
            "age_bucket",
            "has_recommendations",
            "recommendations_total",
            "global_rec_percentile",
            "age_cohort_rec_percentile",
            "cohort_known_count",
            "trend_excess",
            "freshness_weight",
            "trend_signal",
        ]
    ].copy()
    out["parsed_date"] = out["parsed_date"].astype(object).where(out["parsed_date"].notna(), None)
    return out


def print_coverage_audit(df: pd.DataFrame):
    total = len(df)
    print("=" * 60)
    print("T1-01: TREND DATA COVERAGE AUDIT")
    print("=" * 60)

    dates_ok = df["has_valid_date"].sum()
    recs_ok = df["has_recommendations"].sum()
    both_ok = (df["has_valid_date"] & df["has_recommendations"]).sum()

    print(f"  Total games in corpus:       {total:>6d}")
    print(f"  Valid release_date:          {dates_ok:>6d}  ({dates_ok/total*100:5.1f}%)")
    print(f"  Recommendations known:       {recs_ok:>6d}  ({recs_ok/total*100:5.1f}%)")
    print(f"  Both known:                  {both_ok:>6d}  ({both_ok/total*100:5.1f}%)")

    invalid = total - dates_ok
    future = int(df["coming_soon"].sum()) if "coming_soon" in df.columns else 0
    print(f"  Invalid / future date:       {invalid:>6d}  (미출시: {future})")

    print()
    print("--- Age cohort sizes (both-known) ---")
    for bucket in ["0_90d", "91_365d", "1_3y", "3y_plus"]:
        in_b = (df["has_valid_date"] & df["has_recommendations"] & (df["age_bucket"] == bucket)).sum()
        print(f"  {bucket:>12s}: {in_b:>6d}")
    print()


def print_signal_distribution(df: pd.DataFrame):
    print("=" * 60)
    print("T1-09: TREND SIGNAL DISTRIBUTION")
    print("=" * 60)

    nonzero = (df["trend_signal"] > 0).sum()
    zero = (df["trend_signal"] == 0).sum()
    print(f"  trend_signal == 0: {zero} ({zero/len(df)*100:.1f}%)")
    print(f"  trend_signal  > 0: {nonzero} ({nonzero/len(df)*100:.1f}%)")

    if nonzero > 0:
        signals = df[df["trend_signal"] > 0]["trend_signal"]
        print()
        print(f"  Percentiles (nonzero only):")
        for p in [50, 75, 90, 95, 99]:
            print(f"    P{p:>2d}: {signals.quantile(p/100):.4f}")
        print(f"    Max: {signals.max():.4f}")

    print()
    print("--- trend_signal > 0 by age_bucket ---")
    for bucket in ["0_90d", "91_365d", "1_3y", "3y_plus"]:
        cnt = (df["age_bucket"] == bucket).sum()
        nz = ((df["age_bucket"] == bucket) & (df["trend_signal"] > 0)).sum()
        print(f"  {bucket:>12s}: {nz:>6d} / {cnt:<6d} ({nz/cnt*100:.1f}% if cnt > 0 else 0)")

    print()
    print("--- Top-30 by trend_signal ---")
    top30 = df.nlargest(30, "trend_signal")[
        ["name", "raw_release_date", "recommendations_total",
         "global_rec_percentile", "age_cohort_rec_percentile",
         "trend_excess", "freshness_weight", "trend_signal"]
    ]
    for _, r in top30.iterrows():
        name = r["name"][:45] if r["name"] else "?"
        rd = r["raw_release_date"][:12] if isinstance(r["raw_release_date"], str) else "?"
        rec = r["recommendations_total"] if pd.notna(r["recommendations_total"]) else 0
        g = r["global_rec_percentile"]
        c = r["age_cohort_rec_percentile"] if pd.notna(r["age_cohort_rec_percentile"]) else 0
        e = r["trend_excess"]
        fw = r["freshness_weight"]
        ts = r["trend_signal"]
        print(f"  {name:45s}  {rd:12s}  rec={rec:<7d}  g={g:.2f}  c={c:.2f}  e={e:.3f}  w={fw:.2f}  ts={ts:.4f}")

    # Red flag checks
    print()
    print("--- Sanity checks ---")
    old_nz = ((df["age_bucket"] == "3y_plus") & (df["trend_signal"] > 0.01)).sum()
    if old_nz > 0:
        print(f"  ⚠️  3y_plus games with trend_signal > 0.01: {old_nz}")
    else:
        print(f"  ✅  No 3y_plus games with significant trend_signal")

    # Low volume check
    low_vol = df[df["recommendations_total"].notna() & (df["recommendations_total"] > 0)]
    low_small = low_vol[low_vol["recommendations_total"] < 50]
    nz_low = (low_small["trend_signal"] > 0).sum()
    if nz_low > 0:
        print(f"  ⚠️  Games with <50 recs and trend_signal>0: {nz_low}")
        worst = df[(df["trend_signal"] > 0) & (df["recommendations_total"].fillna(0) < 50)].nsmallest(5, "recommendations_total")
        for _, r in worst.iterrows():
            print(f"      {r['name'][:40]:40s}  recs={r['recommendations_total']}  ts={r['trend_signal']:.4f}")
    else:
        print(f"  ✅  No low-volume (<50 recs) games with positive trend_signal")

    print()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="artifacts/full_v1/dataset.parquet")
    ap.add_argument("--out", default="artifacts/trend_v1")
    ap.add_argument("--as-of", default="2026-08-10")
    args = ap.parse_args()

    as_of_date = args.as_of
    min_cohort_size = 50
    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = PROJECT_ROOT / ds_path

    full = build_full_features(as_of_date=as_of_date, min_cohort_size=min_cohort_size,
                               dataset_path=ds_path)
    out = build_output_features(full)

    print_coverage_audit(full)
    print_signal_distribution(full)

    trend_dir = Path(args.out)
    if not trend_dir.is_absolute():
        trend_dir = PROJECT_ROOT / trend_dir
    trend_dir.mkdir(parents=True, exist_ok=True)
    out_path = trend_dir / "trend_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path}")

    config = {
        "as_of_date": as_of_date,
        "min_cohort_size": min_cohort_size,
        "feature_version": "v1",
        "freshness_weights": FRESHNESS_WEIGHTS,
        "dataset": str(ds_path.relative_to(PROJECT_ROOT)),
        "total_games": len(full),
        "games_with_trend_signal_gt_0": int((full["trend_signal"] > 0).sum()),
    }
    config_path = trend_dir / "trend_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {config_path}")


if __name__ == "__main__":
    main()
