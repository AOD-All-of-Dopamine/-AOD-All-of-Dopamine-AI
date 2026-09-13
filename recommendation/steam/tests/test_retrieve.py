# tests/test_retrieve.py
import numpy as np
import pandas as pd
import pytest

import src.retrieve as retrieve_mod
from src.retrieve import assemble_top100


def _corpus():
    return pd.DataFrame({
        "steam_appid": [10, 20, 30, 40, 50],
        "name": ["A", "B", "C", "D", "E"],
        "genres": [["액션"]] * 5,
        "has_metacritic": [True, False, False, False, False],
        "metacritic_score": pd.array([90, None, None, None, None], dtype="Int64"),
        "has_recommendations": [True] * 5,
        "recommendations_total": pd.array([100, 200, 300, 400, 500], dtype="Int64"),
    })


def test_self_excluded_and_sorted():
    df = _corpus()
    anchors = pd.DataFrame({"steam_appid": [10], "name": ["A"], "genres": [["액션"]]})
    # self(10)=1.0 최고점 → 제외되어야 함. 나머지는 50 > 40 > 30 > 20 순
    sims = np.array([[1.0, 0.1, 0.3, 0.5, 0.9]])
    out = assemble_top100(df, anchors, sims, anchor_positions=[0], k=3, experiment_id="exp")
    assert len(out) == 3
    assert out["candidate_steam_appid"].tolist() == [50, 40, 30]
    assert out["rank"].tolist() == [1, 2, 3]
    assert 10 not in out["candidate_steam_appid"].tolist()
    assert out["experiment_id"].unique().tolist() == ["exp"]
    assert float(out.iloc[0]["similarity"]) == 0.9


def test_metadata_carried():
    df = _corpus()
    anchors = pd.DataFrame({"steam_appid": [20], "name": ["B"], "genres": [["인디"]]})
    sims = np.array([[0.9, 1.0, 0.1, 0.2, 0.3]])
    out = assemble_top100(df, anchors, sims, anchor_positions=[1], k=1, experiment_id="exp")
    row = out.iloc[0]
    assert row["candidate_steam_appid"] == 10
    assert bool(row["has_metacritic"]) is True
    assert row["metacritic_score"] == 90
    assert row["anchor_name"] == "B"


def _write_synthetic_artifacts(tmp_path):
    """Task 11 산출물을 흉낸 소형 fixture (corpus row 순서 ≠ dataset 순서)."""
    df = pd.DataFrame({
        "steam_appid": [10, 20, 30, 40, 50, 60],
        "name": ["A10", "A20", "A30", "A40", "A50", "A60"],
        "genres": [["액션"]] * 6,
        "has_metacritic": [True] + [False] * 5,
        "metacritic_score": pd.array([90, None, None, None, None, None], dtype="Int64"),
        "has_recommendations": [True] * 6,
        "recommendations_total": pd.array([100, 200, 300, 400, 500, 600], dtype="Int64"),
    })
    df.to_parquet(tmp_path / "dataset.parquet", index=False)

    anchors = pd.DataFrame({
        "steam_appid": [10, 20],
        "name": ["A10", "A20"],
        "genres": [["액션"], ["인디"]],
    })
    anchors.to_parquet(tmp_path / "anchors_40.parquet", index=False)

    # corpus_index 순서가 dataset 순서와 다름 → main이 idx 순서로 재정렬해야 함
    idx = pd.DataFrame({
        "embedding_row": range(6),
        "steam_appid": [30, 10, 50, 20, 60, 40],
        "name": ["A30", "A10", "A50", "A20", "A60", "A40"],
    })
    idx.to_parquet(tmp_path / "corpus_index.parquet", index=False)

    corpus = np.array([
        [1.0, 0.0],  # row0 = appid 30
        [0.0, 1.0],  # row1 = appid 10
        [0.7, 0.7],  # row2 = appid 50
        [0.2, 0.9],  # row3 = appid 20
        [0.8, 0.1],  # row4 = appid 60
        [0.5, 0.4],  # row5 = appid 40
    ], dtype="float32")
    np.save(tmp_path / "corpus_embeddings.npy", corpus)

    queries = np.array([
        [0.0, 1.0],  # anchor 10
        [1.0, 0.0],  # anchor 20
    ], dtype="float32")
    np.save(tmp_path / "anchor_embeddings.npy", queries)


def test_main_writes_qwen_top100(tmp_path, monkeypatch):
    _write_synthetic_artifacts(tmp_path)
    monkeypatch.setattr(retrieve_mod, "ensure_artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr(retrieve_mod, "load_config", lambda: {"retrieval": {"candidate_k": 3}})

    retrieve_mod.main()

    out = pd.read_parquet(tmp_path / "qwen_top100.parquet")
    assert len(out) == 2 * 3
    assert out["experiment_id"].unique().tolist() == [retrieve_mod.QWEN_EXPERIMENT_ID]

    # anchor 10 (query=[0,1]): sims 20→0.9, 50→0.7, 40→0.4 (self 제외)
    a10 = out[out["anchor_steam_appid"] == 10].sort_values("rank")
    assert a10["candidate_steam_appid"].tolist() == [20, 50, 40]
    assert a10["candidate_name"].tolist() == ["A20", "A50", "A40"]
    assert a10["rank"].tolist() == [1, 2, 3]
    assert 10 not in a10["candidate_steam_appid"].tolist()
    assert a10["similarity"].is_monotonic_decreasing
    assert float(a10.iloc[0]["similarity"]) == pytest.approx(0.9, abs=1e-6)

    # anchor 20 (query=[1,0]): sims 30→1.0, 60→0.8, 50→0.7 (self 제외)
    a20 = out[out["anchor_steam_appid"] == 20].sort_values("rank")
    assert a20["candidate_steam_appid"].tolist() == [30, 60, 50]
    assert a20["candidate_name"].tolist() == ["A30", "A60", "A50"]
    assert a20["rank"].tolist() == [1, 2, 3]
    assert 20 not in a20["candidate_steam_appid"].tolist()
    assert a20["similarity"].is_monotonic_decreasing
    assert float(a20.iloc[0]["similarity"]) == pytest.approx(1.0, abs=1e-6)


def test_qwen_experiment_id_matches_config_treatment():
    """설정의 treatment ID와 어긋나면 evaluate.py 의 비교 블록이 무효가 된다."""
    from src.config import load_config

    assert retrieve_mod.QWEN_EXPERIMENT_ID == load_config()["evaluation"]["treatment_experiment_id"]
