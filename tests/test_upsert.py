import pytest

from aod_ai.models import (
    Extraction, FunTagItem, QualityScore, ReviewSource, SelectedTarget,
)
from aod_ai.pipeline.upsert import upsert_assets


def _target():
    return SelectedTarget(
        content_id=7, domain="WEBNOVEL", master_title="업서트대상",
        original_title=None, synopsis="s", genres=["판타지"], content_hash="HASH7",
    )


def _extraction():
    return Extraction(
        fun_tags=[
            FunTagItem(tag="사이다전개", tag_score=0.9, tag_confidence=0.8,
                       evidence="속시원", is_new=False),
            FunTagItem(tag="신규태그", tag_score=0.6, tag_confidence=0.5,
                       evidence="새로움", is_new=True),
        ],
        normalized_summary="요약", profile_text="프로파일 텍스트", extraction_quality=0.7,
    )


def _quality():
    return QualityScore(
        bayesian_score=3.5, platform_rank_score=0.5, review_count_score=1.0,
        recency_score=1.0, quality_popularity_score=0.73,
    )


def test_upsert_assets_writes_tables_isolates_proposed_and_is_idempotent(db, m1_env):
    with db.cursor() as cur:
        cur.execute("INSERT INTO aod_ai.fun_tag_dict (name, status) VALUES ('사이다전개', 'active')")
    sources = [ReviewSource(content="리뷰원문", url="http://u/1")]

    upsert_assets(db, _target(), _extraction(), [0.1] * 1024, _quality(), sources)

    with db.cursor() as cur:
        cur.execute(
            "SELECT profile_text, source_count, content_hash "
            "FROM aod_ai.content_semantic_profile WHERE content_id = 7"
        )
        assert cur.fetchone() == ("프로파일 텍스트", 1, "HASH7")
        cur.execute("SELECT tag FROM aod_ai.content_fun_tag WHERE content_id = 7 ORDER BY tag")
        assert [r[0] for r in cur.fetchall()] == ["사이다전개"]  # proposed 태그는 매칭 테이블 제외
        cur.execute("SELECT status FROM aod_ai.fun_tag_dict WHERE name = '신규태그'")
        assert cur.fetchone() == ("proposed",)
        cur.execute("SELECT dim FROM aod_ai.content_embedding WHERE content_id = 7")
        assert cur.fetchone() == (1024,)
        cur.execute(
            "SELECT quality_popularity_score FROM aod_ai.content_quality_score WHERE content_id = 7"
        )
        assert cur.fetchone()[0] == pytest.approx(0.73)

    upsert_assets(db, _target(), _extraction(), [0.2] * 1024, _quality(), sources)  # 재실행 멱등
    with db.cursor() as cur:
        cur.execute("SELECT count(*) FROM aod_ai.content_fun_tag WHERE content_id = 7")
        assert cur.fetchone() == (1,)
        cur.execute("SELECT count(*) FROM aod_ai.content_semantic_profile WHERE content_id = 7")
        assert cur.fetchone() == (1,)


def test_upsert_assets_stores_canonical_dict_name_for_case_variant_tags(db, m1_env):
    # 계약 §9 [정규화 태그 저장]: LLM이 대소문자/공백만 다르게 뱉어도
    # content_fun_tag.tag 에는 canonical dict name 이 저장돼야 M2 후보쿼리(tag = ANY)와 매칭된다.
    with db.cursor() as cur:
        cur.execute("INSERT INTO aod_ai.fun_tag_dict (name, status) VALUES ('먼치킨', 'active')")
    extraction = Extraction(
        fun_tags=[
            FunTagItem(tag=" 먼치킨 ", tag_score=0.8, tag_confidence=0.7,
                       evidence="압도적", is_new=False),
        ],
        normalized_summary="요약", profile_text="pt", extraction_quality=0.6,
    )

    upsert_assets(db, _target(), extraction, [0.1] * 1024, _quality(), [])

    with db.cursor() as cur:
        cur.execute("SELECT tag FROM aod_ai.content_fun_tag WHERE content_id = 7")
        assert [r[0] for r in cur.fetchall()] == ["먼치킨"]  # raw " 먼치킨 " 아님
