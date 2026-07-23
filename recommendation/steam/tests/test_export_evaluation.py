# tests/test_export_evaluation.py
import sys

import pandas as pd

import src.export_evaluation as export_mod
from src.export_evaluation import JUDGMENT_COLUMNS, build_judgments


def _tops():
    """TF-IDF/Qwen top-k 결과를 흉낸 소형 fixture.

    - anchor 10 / candidate 30 pair가 두 실험 양쪽에 등장 (pair_key 재사용 검증용)
    - anchor 999는 anchors_40 첫 10개에 없음 (pilot 필터 검증용)
    """
    tfidf = pd.DataFrame({
        "experiment_id": ["steam_s1_tfidf_v1"] * 3,
        "anchor_steam_appid": [10, 10, 20],
        "anchor_name": ["A10", "A10", "A20"],
        "anchor_genres": [["액션", "인디"]] * 2 + [["RPG"]],
        "rank": [1, 2, 1],
        "candidate_steam_appid": [30, 40, 30],
        "candidate_name": ["C30", "C40", "C30"],
        "candidate_genres": [["액션"], ["인디"], ["RPG", "액션"]],
        "similarity": [0.9, 0.8, 0.7],
        "has_metacritic": [True, False, False],
        "metacritic_score": pd.array([90, None, None], dtype="Int64"),
        "has_recommendations": [True] * 3,
        "recommendations_total": pd.array([100, 200, 300], dtype="Int64"),
    })
    qwen = pd.DataFrame({
        "experiment_id": ["steam_s1_qwen_v1"] * 3,
        "anchor_steam_appid": [10, 10, 999],
        "anchor_name": ["A10", "A10", "A999"],
        "anchor_genres": [["액션", "인디"]] * 2 + [["시뮬레이션"]],
        "rank": [1, 2, 1],
        "candidate_steam_appid": [30, 50, 60],
        "candidate_name": ["C30", "C50", "C60"],
        "candidate_genres": [["액션"], ["시뮬레이션"], ["인디"]],
        "similarity": [0.95, 0.85, 0.75],
        "has_metacritic": [True, False, False],
        "metacritic_score": pd.array([90, None, None], dtype="Int64"),
        "has_recommendations": [True] * 3,
        "recommendations_total": pd.array([100, 500, 600], dtype="Int64"),
    })
    return [tfidf, qwen]


def test_judgment_columns_match_spec_order():
    assert JUDGMENT_COLUMNS == [
        "experiment_id", "pair_key",
        "anchor_steam_appid", "anchor_name", "anchor_genres",
        "rank",
        "candidate_steam_appid", "candidate_name", "candidate_genres",
        "similarity",
        "has_metacritic", "metacritic_score",
        "has_recommendations", "recommendations_total",
        "relevance", "recommendation_confidence", "error_tag",
        "evaluator", "notes",
    ]


def test_output_columns_exact_order():
    out = build_judgments(_tops(), k=2)
    assert out.columns.tolist() == JUDGMENT_COLUMNS


def test_rank_filtered_by_k():
    out = build_judgments(_tops(), k=1)
    assert (out["rank"] <= 1).all()
    # tfidf rank1: (10,c30),(20,c30) / qwen rank1: (10,c30),(999,c60)
    assert len(out) == 4


def test_pair_key_shared_across_experiments():
    out = build_judgments(_tops(), k=2)
    tfidf_row = out[
        (out["experiment_id"] == "steam_s1_tfidf_v1")
        & (out["anchor_steam_appid"] == 10) & (out["rank"] == 1)
    ].iloc[0]
    qwen_row = out[
        (out["experiment_id"] == "steam_s1_qwen_v1")
        & (out["anchor_steam_appid"] == 10) & (out["rank"] == 1)
    ].iloc[0]
    assert tfidf_row["pair_key"] == "10:30"
    assert tfidf_row["pair_key"] == qwen_row["pair_key"]
    # 동일 pair_key라도 experiment별 행은 유지 (자동 복사 스크립트가 병합)
    assert out["pair_key"].duplicated().sum() == 1


def test_genres_joined_to_string():
    out = build_judgments(_tops(), k=2)
    row = out[
        (out["experiment_id"] == "steam_s1_tfidf_v1")
        & (out["anchor_steam_appid"] == 10) & (out["rank"] == 1)
    ].iloc[0]
    assert row["anchor_genres"] == "액션, 인디"
    assert row["candidate_genres"] == "액션"


def test_judgment_columns_blank():
    out = build_judgments(_tops(), k=2)
    assert out["relevance"].isna().all()
    assert out["recommendation_confidence"].isna().all()
    assert (out["error_tag"] == "").all()
    assert (out["evaluator"] == "").all()
    assert (out["notes"] == "").all()


def _write_synthetic_artifacts(tmp_path):
    tfidf, qwen = _tops()
    tfidf.to_parquet(tmp_path / "tfidf_top100.parquet", index=False)
    qwen.to_parquet(tmp_path / "qwen_top100.parquet", index=False)
    anchors = pd.DataFrame({
        "steam_appid": [10, 20] + list(range(100, 138)),  # 40 anchors, 999 미포함
        "name": [f"A{i}" for i in range(40)],
    })
    anchors.to_parquet(tmp_path / "anchors_40.parquet", index=False)


def test_main_full_writes_xlsx(tmp_path, monkeypatch, capsys):
    _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(export_mod, "ensure_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(export_mod, "load_config", lambda: {"retrieval": {"evaluation_k": 1}})
    monkeypatch.setattr(sys, "argv", ["export_evaluation"])

    export_mod.main()

    out = pd.read_excel(tmp_path / "evaluation.xlsx", sheet_name="Judgments")
    assert out.columns.tolist() == JUDGMENT_COLUMNS
    assert len(out) == 4  # rank<=1 인 4행
    assert set(out["experiment_id"].unique()) == {"steam_s1_tfidf_v1", "steam_s1_qwen_v1"}
    assert out["relevance"].isna().all()
    captured = capsys.readouterr()
    assert "rows=4" in captured.out
    assert "duplicated_pair_keys=1" in captured.out
    assert not (tmp_path / "evaluation_pilot.xlsx").exists()


def test_main_pilot_filters_first10_anchors(tmp_path, monkeypatch, capsys):
    _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(export_mod, "ensure_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(export_mod, "load_config", lambda: {"retrieval": {"evaluation_k": 2}})
    monkeypatch.setattr(sys, "argv", ["export_evaluation", "--pilot"])

    export_mod.main()

    out = pd.read_excel(tmp_path / "evaluation_pilot.xlsx", sheet_name="Judgments")
    assert out.columns.tolist() == JUDGMENT_COLUMNS
    # anchors_40 첫 10개([10, 20, 100..107])에 속하는 행만 → anchor 999 제외
    assert len(out) == 5
    assert 999 not in out["anchor_steam_appid"].tolist()
    captured = capsys.readouterr()
    assert "rows=5" in captured.out
