from aod_ai.models import SelectedTarget
from aod_ai.pipeline.select_targets import (
    compute_content_hash,
    select_recollect_targets,
    select_targets,
)


def _insert_webnovel(db, content_id, title, synopsis, genres):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, original_title, synopsis) "
            "VALUES (%s, 'WEBNOVEL', %s, NULL, %s)",
            (content_id, title, synopsis),
        )
        cur.execute(
            "INSERT INTO public.webnovel_contents (content_id, genres, platforms) "
            "VALUES (%s, %s, %s)",
            (content_id, genres, []),
        )


def test_select_targets_includes_missing_and_changed_skips_uptodate(db):
    _insert_webnovel(db, 1, "전지적 독자 시점", "재난 웹소설", ["판타지"])
    _insert_webnovel(db, 2, "달빛조각사", "게임 판타지", ["게임판타지"])
    _insert_webnovel(db, 3, "나 혼자만 레벨업", "헌터물", ["액션"])
    h2 = compute_content_hash("달빛조각사", None, "게임 판타지", ["게임판타지"])
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO aod_ai.content_semantic_profile "
            "(content_id, domain, profile_text, content_hash) "
            "VALUES (2, 'WEBNOVEL', 'x', %s)",
            (h2,),
        )
        cur.execute(
            "INSERT INTO aod_ai.content_semantic_profile "
            "(content_id, domain, profile_text, content_hash) "
            "VALUES (3, 'WEBNOVEL', 'x', 'STALE')"
        )

    targets = select_targets(db, "WEBNOVEL", 200)

    assert [t.content_id for t in targets] == [1, 3]
    assert all(isinstance(t, SelectedTarget) for t in targets)
    assert targets[0].genres == ["판타지"]
    assert targets[0].content_hash == compute_content_hash(
        "전지적 독자 시점", None, "재난 웹소설", ["판타지"]
    )


def test_select_recollect_targets_picks_only_sourceless_profiles(db):
    # M1.1 재수집: 프로파일이 있고 source_count=0인 콘텐츠만 대상 (해시 무관)
    _insert_webnovel(db, 1, "소스없음", "s1", ["판타지"])
    _insert_webnovel(db, 2, "소스있음", "s2", ["판타지"])
    _insert_webnovel(db, 3, "미처리", "s3", ["판타지"])
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO aod_ai.content_semantic_profile "
            "(content_id, domain, profile_text, content_hash, source_count) "
            "VALUES (1, 'WEBNOVEL', 'x', 'h1', 0)"
        )
        cur.execute(
            "INSERT INTO aod_ai.content_semantic_profile "
            "(content_id, domain, profile_text, content_hash, source_count) "
            "VALUES (2, 'WEBNOVEL', 'x', 'h2', 5)"
        )

    targets = select_recollect_targets(db, "WEBNOVEL", 200)

    assert [t.content_id for t in targets] == [1]  # 소스있음(2)·미처리(3) 제외
    assert isinstance(targets[0], SelectedTarget)
    assert targets[0].master_title == "소스없음"
