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


def test_experiment_id_matches_config_baseline():
    """리터럴이 아니라 설정과 대조한다.

    예전에는 `steam_s1_tfidf_v1` 로 하드코딩돼 있었고, 실험이 v2로 올라간 뒤
    evaluate.py 의 비교 블록이 조용히 스킵됐다(품질 게이트가 한 번도 안 돌았다).
    """
    from src.config import load_config

    assert EXPERIMENT_ID == load_config()["evaluation"]["baseline_experiment_id"]
