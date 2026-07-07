from datetime import date

import pytest

from aod_ai.models import SelectedTarget
from aod_ai.pipeline.quality import compute_quality


def _seed(db):
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, average_score, review_count, release_date) "
            "VALUES (1, 'WEBNOVEL', 'A', 4.0, 10, %s)",
            (date.today(),),
        )
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, average_score, review_count, release_date) "
            "VALUES (2, 'WEBNOVEL', 'B', 2.0, 0, %s)",
            (date.today(),),
        )
        cur.execute(
            "INSERT INTO public.external_ranking (platform, ranking, content_id, title) "
            "VALUES ('NaverSeries', 1, 1, 'A')"
        )


def test_compute_quality_bayesian_and_components(db):
    _seed(db)
    target = SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="A",
        original_title=None, synopsis=None, genres=[], content_hash="h",
    )

    q = compute_quality(db, target)

    assert q.bayesian_score == pytest.approx(3.5)         # 0.5*4 + 0.5*3(C=global mean)
    assert q.platform_rank_score == pytest.approx(0.5)    # 1/(1+1)
    assert q.review_count_score == pytest.approx(1.0)     # log1p(10)/log1p(10)
    assert q.recency_score == pytest.approx(1.0)          # released today
    assert q.quality_popularity_score == pytest.approx(0.73)  # .4*.7+.3*.5+.2*1+.1*1


def test_compute_quality_clamps_future_release_date(db):
    # 리뷰 F#3: 미래 출시일이면 recency가 1.0을 넘지 않아야 한다.
    from datetime import timedelta

    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, average_score, review_count, release_date) "
            "VALUES (9, 'WEBNOVEL', '미래작', 4.0, 10, %s)",
            (date.today() + timedelta(days=30),),
        )
    target = SelectedTarget(
        content_id=9, domain="WEBNOVEL", master_title="미래작",
        original_title=None, synopsis=None, genres=[], content_hash="h",
    )

    q = compute_quality(db, target)

    assert q.recency_score == pytest.approx(1.0)


def test_compute_quality_global_stats_are_domain_scoped(db):
    # 리뷰 F#2: 타 도메인 대작이 있어도 웹소설의 C/max_reviews에 영향을 주면 안 된다.
    _seed(db)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.contents "
            "(content_id, domain, master_title, average_score, review_count, release_date) "
            "VALUES (100, 'GAME', '초대작게임', 5.0, 1000000, %s)",
            (date.today(),),
        )
    target = SelectedTarget(
        content_id=1, domain="WEBNOVEL", master_title="A",
        original_title=None, synopsis=None, genres=[], content_hash="h",
    )

    q = compute_quality(db, target)

    # WEBNOVEL만 집계: C=3.0, max_reviews=10 → 기존 기대값 그대로
    assert q.bayesian_score == pytest.approx(3.5)
    assert q.review_count_score == pytest.approx(1.0)
