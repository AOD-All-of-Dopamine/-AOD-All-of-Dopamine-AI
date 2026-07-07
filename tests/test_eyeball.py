from aod_ai.eyeball import write_eyeball_dump
from aod_ai.models import Extraction, FunTagItem, SelectedTarget


def test_write_eyeball_dump_contains_titles_tags_and_evidence(tmp_path):
    target = SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="전지적 독자 시점",
        original_title=None, synopsis="s", genres=["판타지", "현대"], content_hash="h",
    )
    extraction = Extraction(
        fun_tags=[
            FunTagItem(tag="사이다전개", tag_score=0.91, tag_confidence=0.82,
                       evidence="복수 장면이 통쾌하다", is_new=False),
            FunTagItem(tag="신규태그", tag_score=0.5, tag_confidence=0.4,
                       evidence="독특한 설정", is_new=True),
        ],
        normalized_summary="재난 생존 웹소설", profile_text="pt", extraction_quality=0.77,
    )
    out = tmp_path / "fun_tags_review.md"

    write_eyeball_dump([(target, extraction)], str(out))

    text = out.read_text(encoding="utf-8")
    assert "전지적 독자 시점" in text
    assert "사이다전개" in text
    assert "복수 장면이 통쾌하다" in text
    assert "(proposed/new)" in text
    assert "재난 생존 웹소설" in text
