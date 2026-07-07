from datetime import date

import aod_ai.run as run
from aod_ai.models import Extraction, FunTagItem, ReviewSource


def test_parse_args_defaults_and_values():
    args = run._parse_args(["--domain", "WEBNOVEL", "--limit", "200"])
    assert args.domain == "WEBNOVEL"
    assert args.limit == 200
    assert args.recollect is False
    assert args.delay == 0.0


def test_parse_args_recollect_and_delay():
    args = run._parse_args(
        ["--domain", "WEBNOVEL", "--recollect", "--delay", "20"])
    assert args.recollect is True
    assert args.delay == 20.0


def test_run_pipeline_recollect_reprocesses_sourceless_profiles(
        db, fake_vane, fake_llm, fake_emb, m1_env):
    # M1.1: 이미 프로파일이 있어도 source_count=0이면 재수집 모드로 다시 처리돼야 한다
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents (content_id, domain, master_title) "
            "VALUES (1, 'WEBNOVEL', '재수집대상')")
        cur.execute(
            "INSERT INTO public.webnovel_contents (content_id, genres, platforms) "
            "VALUES (1, %s, %s)", (["판타지"], []))
        cur.execute(
            "INSERT INTO aod_ai.content_semantic_profile "
            "(content_id, domain, profile_text, content_hash, source_count) "
            "VALUES (1, 'WEBNOVEL', '옛프로파일', 'STALE', 0)")

    vane = fake_vane([ReviewSource(content="드디어 리뷰", url="http://u/1")])
    extraction = Extraction(
        fun_tags=[], normalized_summary="새요약", profile_text="새프로파일",
        extraction_quality=0.8)
    records = run.run_pipeline(
        db, "WEBNOVEL", 200, vane=vane, llm=fake_llm(extraction),
        emb=fake_emb([0.05] * 1024), recollect=True)

    assert len(records) == 1
    with db.cursor() as cur:
        cur.execute(
            "SELECT profile_text, source_count FROM aod_ai.content_semantic_profile "
            "WHERE content_id = 1")
        assert cur.fetchone() == ("새프로파일", 1)  # 재처리로 갱신됨


class _PoisonLlm:
    """첫 번째 호출만 실패하는 LLM — 포이즌 필 격리 검증용."""

    def __init__(self, extraction):
        self.extraction = extraction
        self.calls = 0

    def extract(self, *, metadata, sources, active_tags):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("LLM 응답 파싱 실패 (모든 재시도 소진)")
        return self.extraction


def test_run_pipeline_isolates_per_content_failures(db, fake_vane, fake_emb, m1_env):
    # 리뷰 F#1: 콘텐츠 하나가 계속 실패해도 나머지는 처리되고 실패 목록이 보고돼야 한다.
    with db.cursor() as cur:
        for cid, title in [(1, "포이즌"), (2, "정상작품")]:
            cur.execute(
                "INSERT INTO public.contents (content_id, domain, master_title) "
                "VALUES (%s, 'WEBNOVEL', %s)",
                (cid, title),
            )
            cur.execute(
                "INSERT INTO public.webnovel_contents (content_id, genres, platforms) "
                "VALUES (%s, %s, %s)",
                (cid, ["판타지"], []),
            )
    vane = fake_vane([ReviewSource(content="리뷰", url="http://u/1")])
    extraction = Extraction(
        fun_tags=[], normalized_summary="요약", profile_text="pt", extraction_quality=0.5,
    )
    llm = _PoisonLlm(extraction)
    emb = fake_emb([0.05] * 1024)
    failed_ids: list[int] = []

    records = run.run_pipeline(
        db, "WEBNOVEL", 200, vane=vane, llm=llm, emb=emb, failed_ids=failed_ids
    )

    assert [t.content_id for t, _ in records] == [2]
    assert failed_ids == [1]
    with db.cursor() as cur:
        cur.execute("SELECT content_id FROM aod_ai.content_semantic_profile ORDER BY content_id")
        assert [r[0] for r in cur.fetchall()] == [2]


def test_run_pipeline_end_to_end(db, fake_vane, fake_llm, fake_emb, m1_env):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, average_score, review_count, release_date) "
            "VALUES (1, 'WEBNOVEL', '전독시', 4.0, 10, %s)",
            (date.today(),),
        )
        cur.execute(
            "INSERT INTO public.webnovel_contents (content_id, genres, platforms) "
            "VALUES (1, %s, %s)",
            (["판타지"], []),
        )
        cur.execute("INSERT INTO aod_ai.fun_tag_dict (name, status) VALUES ('사이다전개', 'active')")

    vane = fake_vane([ReviewSource(content="리뷰원문", url="http://u/1")])
    extraction = Extraction(
        fun_tags=[
            FunTagItem(tag="사이다전개", tag_score=0.9, tag_confidence=0.8,
                       evidence="속시원", is_new=True),          # 실제로는 active → False 로 교정돼야
            FunTagItem(tag="신규발명태그", tag_score=0.5, tag_confidence=0.4,
                       evidence="신규", is_new=False),            # 실제로는 신규 → proposed 로 교정돼야
        ],
        normalized_summary="요약", profile_text="프로파일", extraction_quality=0.7,
    )
    llm = fake_llm(extraction)
    emb = fake_emb([0.05] * 1024)

    records = run.run_pipeline(db, "WEBNOVEL", 200, vane=vane, llm=llm, emb=emb)

    assert len(records) == 1
    with db.cursor() as cur:
        cur.execute("SELECT content_hash FROM aod_ai.content_semantic_profile WHERE content_id = 1")
        assert cur.fetchone()[0]
        cur.execute("SELECT tag FROM aod_ai.content_fun_tag WHERE content_id = 1 ORDER BY tag")
        assert [r[0] for r in cur.fetchall()] == ["사이다전개"]
        cur.execute("SELECT status FROM aod_ai.fun_tag_dict WHERE name = '신규발명태그'")
        assert cur.fetchone() == ("proposed",)
        cur.execute("SELECT dim FROM aod_ai.content_embedding WHERE content_id = 1")
        assert cur.fetchone() == (1024,)
        cur.execute(
            "SELECT quality_popularity_score FROM aod_ai.content_quality_score WHERE content_id = 1"
        )
        assert cur.fetchone()[0] is not None
