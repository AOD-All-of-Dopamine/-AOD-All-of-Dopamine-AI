import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


from src.trend.trend_features import (
    parse_release_date,
    assign_age_bucket,
    FRESHNESS_WEIGHTS,
)


def test_parse_valid_korean_date():
    dt = parse_release_date("2024년 2월 8일")
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 2
    assert dt.day == 8


def test_parse_single_digit_month_day():
    dt = parse_release_date("2000년 11월 1일")
    assert dt is not None
    assert dt.year == 2000
    assert dt.month == 11
    assert dt.day == 1


def test_parse_empty():
    assert parse_release_date("") is None
    assert parse_release_date(None) is None
    assert parse_release_date("   ") is None


def test_parse_coming_soon():
    assert parse_release_date("출시 예정") is None
    assert parse_release_date("출시 예정 2024") is None


def test_assign_bucket_0_90d():
    assert assign_age_bucket(0) == "0_90d"
    assert assign_age_bucket(45) == "0_90d"
    assert assign_age_bucket(90) == "0_90d"


def test_assign_bucket_91_365d():
    assert assign_age_bucket(91) == "91_365d"
    assert assign_age_bucket(200) == "91_365d"
    assert assign_age_bucket(365) == "91_365d"


def test_assign_bucket_1_3y():
    assert assign_age_bucket(366) == "1_3y"
    assert assign_age_bucket(730) == "1_3y"
    assert assign_age_bucket(1095) == "1_3y"


def test_assign_bucket_3y_plus():
    assert assign_age_bucket(1096) == "3y_plus"
    assert assign_age_bucket(3650) == "3y_plus"


def test_freshness_weights_structure():
    assert set(FRESHNESS_WEIGHTS.keys()) == {"0_90d", "91_365d", "1_3y", "3y_plus"}
    assert FRESHNESS_WEIGHTS["0_90d"] == 1.0
    assert FRESHNESS_WEIGHTS["3y_plus"] == 0.0
    assert FRESHNESS_WEIGHTS["91_365d"] > FRESHNESS_WEIGHTS["1_3y"]


def test_old_game_freshness_zero():
    assert FRESHNESS_WEIGHTS.get("3y_plus") == 0.0


def test_artifact_exists():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "trend_v1" / "trend_features.parquet"
    assert path.exists(), f"Trend feature artifact not found at {path}"


def test_artifact_schema():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "trend_v1" / "trend_features.parquet"
    df = pd.read_parquet(path)

    required = [
        "steam_appid", "trend_signal", "trend_excess",
        "global_rec_percentile", "age_cohort_rec_percentile",
        "freshness_weight", "age_bucket",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

    # trend_signal should be non-negative
    assert (df["trend_signal"] >= 0).all()
    # trend_excess should be non-negative
    assert (df["trend_excess"] >= 0).all()


def test_trend_signal_zero_for_missing_date():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "trend_v1" / "trend_features.parquet"
    df = pd.read_parquet(path)
    no_date = df[df["raw_release_date"].isna() | (df["raw_release_date"] == "")]
    if len(no_date) > 0:
        assert (no_date["trend_signal"] == 0).all(), "Games without release date should have trend_signal=0"


def test_trend_signal_zero_for_missing_recs():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "trend_v1" / "trend_features.parquet"
    df = pd.read_parquet(path)
    no_rec = df[~df["has_recommendations"]]
    if len(no_rec) > 0:
        assert (no_rec["trend_signal"] == 0).all(), "Games without recommendations should have trend_signal=0"


def test_artifact_config_exists():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "trend_v1" / "trend_config.json"
    assert path.exists()


def test_3y_plus_trend_signal_zero():
    path = Path(__file__).resolve().parents[1] / "artifacts" / "trend_v1" / "trend_features.parquet"
    df = pd.read_parquet(path)

    old = df[df["age_bucket"] == "3y_plus"]
    # Freshness weight is 0 for 3y_plus, so trend_signal should be 0
    assert (old["trend_signal"] == 0).all(), "3y_plus games should have trend_signal=0"


# --- 새 크롤러의 dict 형식 (src/crawl_steam.py) ---

def test_parse_release_date_accepts_dict():
    from datetime import datetime

    assert parse_release_date({"coming_soon": False, "date": "2000년 11월 1일"}) == datetime(2000, 11, 1)


def test_parse_release_date_dict_coming_soon_is_none():
    """coming_soon 플래그가 '출시 예정' 문자열 매칭보다 정확하다."""
    assert parse_release_date({"coming_soon": True, "date": "2027년"}) is None
    # 날짜 문자열이 멀쩡해도 coming_soon 이면 미출시다
    assert parse_release_date({"coming_soon": True, "date": "2030년 1월 1일"}) is None


def test_parse_release_date_dict_missing_date():
    assert parse_release_date({"coming_soon": False}) is None
    assert parse_release_date({}) is None


def test_parse_release_date_still_accepts_str():
    """구 jsonl 형식도 계속 지원한다."""
    from datetime import datetime

    assert parse_release_date("2000년 11월 1일") == datetime(2000, 11, 1)
    assert parse_release_date("출시 예정") is None
    assert parse_release_date("") is None
    assert parse_release_date(None) is None
