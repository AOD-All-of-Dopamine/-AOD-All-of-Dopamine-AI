# tests/test_tfidf_baseline.py
from src.tfidf_baseline import EXPERIMENT_ID, build_tfidf_matrix


def _cfg():
    return {
        "tfidf": {
            "ngram_min": 1,
            "ngram_max": 2,
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 1000,
        }
    }


def test_build_tfidf_matrix_shape_and_params():
    texts = [
        "action adventure fantasy rpg",
        "racing car driving simulator",
        "action shooter war combat",
    ]
    vec, mat = build_tfidf_matrix(texts, _cfg())
    assert mat.shape[0] == 3
    assert 0 < mat.shape[1] <= 1000
    # ngram_max=2 → bigram 피처가 존재해야 함
    assert any(" " in f for f in vec.get_feature_names_out())
    # corpus의 모든 행에 대해 self-similarity가 최대 (유효한 정규화 확인)
    diag = (mat @ mat.T).diagonal()
    assert diag.max() <= 1.0 + 1e-9


def test_experiment_id():
    assert EXPERIMENT_ID == "steam_s1_tfidf_v1"
