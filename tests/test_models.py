from aod_ai.models import (
    Extraction,
    FunTagItem,
    QualityScore,
    ReviewSource,
    SelectedTarget,
    content_hash,
)


def test_extraction_round_trips_from_json():
    raw = (
        '{"fun_tags":[{"tag":"회귀","tag_score":0.9,"tag_confidence":0.8,'
        '"evidence":"회귀물","is_new":false}],'
        '"normalized_summary":"요약","profile_text":"프로파일","extraction_quality":0.7}'
    )
    e = Extraction.model_validate_json(raw)
    assert isinstance(e.fun_tags[0], FunTagItem)
    assert e.fun_tags[0].tag == "회귀"
    assert e.extraction_quality == 0.7


def test_selected_target_and_quality_fields():
    t = SelectedTarget(
        content_id=1,
        domain="WEBNOVEL",
        master_title="t",
        original_title=None,
        synopsis=None,
        genres=["판타지"],
        content_hash="h",
    )
    assert t.original_title is None
    q = QualityScore(
        bayesian_score=1.0,
        platform_rank_score=0.5,
        review_count_score=0.2,
        recency_score=0.1,
        quality_popularity_score=0.9,
    )
    assert q.quality_popularity_score == 0.9
    assert ReviewSource(content="c", url="http://u").url == "http://u"


def test_content_hash_is_stable_and_genre_order_invariant():
    a = content_hash("주술회전", None, "줄거리", ["액션", "판타지"])
    b = content_hash("주술회전", None, "줄거리", ["판타지", "액션"])
    assert a == b
    assert len(a) == 64
    assert a != content_hash("주술회전", None, "다른줄거리", ["액션", "판타지"])
