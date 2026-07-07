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
