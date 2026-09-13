import pandas as pd

from src.validate_evaluation import fill_duplicate_judgments, validate_dataframe


def _row(rel=2, conf=2, tag="", pair="1:2"):
    return {
        "experiment_id": "e", "pair_key": pair,
        "relevance": rel, "recommendation_confidence": conf, "error_tag": tag,
    }


def test_valid_rows_pass():
    df = pd.DataFrame([_row(rel=2), _row(rel=1, tag="GENRE_ONLY"), _row(rel=0, tag="IRRELEVANT")])
    assert validate_dataframe(df) == []


def test_relevance_out_of_range():
    df = pd.DataFrame([_row(rel=4)])
    errors = validate_dataframe(df)
    assert any("relevance" in e for e in errors)


def test_low_relevance_requires_error_tag():
    df = pd.DataFrame([_row(rel=1, tag="")])
    errors = validate_dataframe(df)
    assert any("error_tag" in e for e in errors)


def test_invalid_error_tag():
    df = pd.DataFrame([_row(rel=1, tag="WRONG_TAG")])
    errors = validate_dataframe(df)
    assert any("error_tag" in e for e in errors)


def test_unjudged_row_fails():
    df = pd.DataFrame([_row(rel=pd.NA, conf=pd.NA)])
    errors = validate_dataframe(df)
    assert any("relevance" in e for e in errors)


def test_fill_duplicate_judgments():
    df = pd.DataFrame([
        _row(rel=3, conf=2, tag="", pair="1:2"),
        _row(rel=pd.NA, conf=pd.NA, tag="", pair="1:2"),
        _row(rel=pd.NA, conf=pd.NA, tag="", pair="9:9"),
    ])
    out = fill_duplicate_judgments(df)
    assert int(out.loc[1, "relevance"]) == 3
    assert int(out.loc[1, "recommendation_confidence"]) == 2
    assert pd.isna(out.loc[2, "relevance"])  # 판정 없는 pair는 그대로
