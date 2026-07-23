# tests/test_report.py
import pandas as pd

from src.report import error_distribution, failure_examples

EXPECTED_COLS = [
    "experiment_id", "anchor_name", "candidate_name", "similarity",
    "rank", "relevance", "error_tag", "candidate_genres",
]


def _judged() -> pd.DataFrame:
    """Synthetic judged Judgments 시트."""
    return pd.DataFrame({
        "experiment_id": ["exp_a"] * 4 + ["exp_b"] * 2,
        "anchor_name": ["A1", "A1", "A2", "A2", "B1", "B1"],
        "candidate_name": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "similarity": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "rank": [1, 2, 1, 2, 1, 2],
        "relevance": [0, 1, 1, 3, 0, 2],
        "error_tag": [
            "IRRELEVANT", "GENRE_ONLY", "GENRE_ONLY", "",
            "KEYWORD_MATCH", "",
        ],
        "candidate_genres": ["Action"] * 6,
    })


def test_error_distribution_counts_and_ratios():
    dist = error_distribution(_judged())

    a = dist[dist["experiment_id"] == "exp_a"].set_index("error_tag")
    assert a.loc["GENRE_ONLY", "count"] == 2
    assert a.loc["GENRE_ONLY", "ratio"] == round(2 / 3, 4)
    assert a.loc["IRRELEVANT", "count"] == 1
    assert a.loc["IRRELEVANT", "ratio"] == round(1 / 3, 4)

    b = dist[dist["experiment_id"] == "exp_b"].iloc[0]
    assert b["error_tag"] == "KEYWORD_MATCH"
    assert b["count"] == 1
    assert b["ratio"] == 1.0


def test_error_distribution_excludes_relevance_above_1():
    dist = error_distribution(_judged())
    # low(relevance <= 1) rows: 3 in exp_a + 1 in exp_b
    assert dist["count"].sum() == 4


def test_error_distribution_empty_when_no_low_relevance():
    df = _judged()
    df["relevance"] = 3
    dist = error_distribution(df)
    assert len(dist) == 0


def test_failure_examples_columns_sorted_top_n():
    ex = failure_examples(_judged(), n=2)
    assert list(ex.columns) == EXPECTED_COLS
    # similarity 내림차순: C1(0.9), C2(0.8) — relevance >= 2 인 C4/C6 제외
    assert list(ex["candidate_name"]) == ["C1", "C2"]
    assert (ex["relevance"].astype(int) <= 1).all()
    assert len(ex) == 2


def test_failure_examples_defaults_to_25_or_fewer():
    ex = failure_examples(_judged())
    assert len(ex) == 4  # low rows 4개뿐
