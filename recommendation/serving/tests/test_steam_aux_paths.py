"""Steam 부가 파일(리뷰·트렌드) 경로 규칙 — 코퍼스 폴더 안이 먼저, 없으면 예전 고정 경로."""
import sys
from pathlib import Path
import pytest

STEAM = Path(__file__).resolve().parents[2] / "steam"


@pytest.fixture()
def ranker_mod(monkeypatch):
    monkeypatch.chdir(STEAM); monkeypatch.syspath_prepend(str(STEAM))
    for m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        monkeypatch.delitem(sys.modules, m)
    import src.personalization.personalized_ranker as mod
    yield mod
    for m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        sys.modules.pop(m, None)


def test_corpus_folder_wins(ranker_mod, tmp_path):
    (tmp_path / "reviews").mkdir(); (tmp_path / "trend_features.parquet").write_bytes(b"")
    assert ranker_mod.review_dir(tmp_path) == tmp_path / "reviews"
    assert ranker_mod.trend_file(tmp_path, None) == tmp_path / "trend_features.parquet"


def test_falls_back_to_legacy_paths(ranker_mod, tmp_path):
    assert ranker_mod.review_dir(tmp_path) == ranker_mod.REVIEW_DIR
    assert ranker_mod.trend_file(tmp_path, None) == ranker_mod.TREND_DIR / "trend_features.parquet"


def test_explicit_trend_dir_still_wins(ranker_mod, tmp_path):
    (tmp_path / "trend_features.parquet").write_bytes(b"")
    other = tmp_path / "other"; other.mkdir()
    assert ranker_mod.trend_file(tmp_path, other) == other / "trend_features.parquet"
