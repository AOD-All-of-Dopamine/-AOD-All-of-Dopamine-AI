from aod_ai.models import ReviewSource, SelectedTarget
from aod_ai.pipeline.collect_reviews import VANE_SYSTEM_INSTRUCTIONS, collect_reviews


def _target():
    return SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="전지적 독자 시점",
        original_title=None, synopsis="재난", genres=["판타지"], content_hash="h",
    )


def test_collect_reviews_builds_query_dedupes_and_caps(monkeypatch, fake_vane, m1_env):
    monkeypatch.setenv("VANE_MAX_SOURCES", "3")
    raw = [ReviewSource(content=f"리뷰{i}", url=f"http://u/{i}") for i in range(5)]
    raw.append(ReviewSource(content="중복", url="http://u/0"))
    vane = fake_vane(raw)

    out = collect_reviews(vane, _target())

    assert len(out) == 3
    assert [s.url for s in out] == ["http://u/0", "http://u/1", "http://u/2"]
    call = vane.calls[0]
    # WEBNOVEL → 웹소설 (도메인 라벨은 target.domain에서 파생)
    assert call["query"] == '"전지적 독자 시점" 웹소설 리뷰 후기 재미'
    assert call["sources"] == ["web", "discussions"]
    assert call["system_instructions"] == VANE_SYSTEM_INSTRUCTIONS
