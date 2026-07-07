from aod_ai.models import Extraction, FunTagItem, ReviewSource, SelectedTarget
from aod_ai.pipeline.extract import extract_profile


def _target():
    return SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="T",
        original_title=None, synopsis="s", genres=["판타지"], content_hash="h",
    )


def test_extract_profile_recomputes_is_new_against_active_tags(fake_llm):
    model_out = Extraction(
        fun_tags=[
            FunTagItem(tag="사이다전개", tag_score=0.9, tag_confidence=0.8,
                       evidence="속 시원한 복수", is_new=True),
            FunTagItem(tag="회귀먼치킨", tag_score=0.7, tag_confidence=0.6,
                       evidence="회귀 후 무쌍", is_new=False),
        ],
        normalized_summary="요약", profile_text="임베딩용 텍스트", extraction_quality=0.75,
    )
    llm = fake_llm(model_out)

    result = extract_profile(
        llm, _target(), [ReviewSource(content="c", url="u")],
        active_tags=["사이다전개", "성장물"],
    )

    flags = {i.tag: i.is_new for i in result.fun_tags}
    assert flags == {"사이다전개": False, "회귀먼치킨": True}
    assert llm.calls[0]["active_tags"] == ["사이다전개", "성장물"]
    assert result.profile_text == "임베딩용 텍스트"
