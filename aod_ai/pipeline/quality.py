import math
from datetime import date

from aod_ai.models import QualityScore

BAYESIAN_MIN_VOTES = 10.0
RECENCY_HALFLIFE_DAYS = 365.0

_TARGET_SQL = """
SELECT c.average_score, c.review_count, c.release_date,
       (SELECT MIN(er.ranking) FROM public.external_ranking er
        WHERE er.content_id = c.content_id) AS best_rank
FROM public.contents c
WHERE c.content_id = %s
"""
_GLOBAL_SQL = """
SELECT AVG(average_score), MAX(review_count)
FROM public.contents
WHERE average_score IS NOT NULL
"""


def compute_quality(conn, target) -> QualityScore:
    with conn.cursor() as cur:
        cur.execute(_TARGET_SQL, (target.content_id,))
        avg_score, review_count, release_date, best_rank = cur.fetchone()
        cur.execute(_GLOBAL_SQL)
        global_avg, max_reviews = cur.fetchone()

    C = float(global_avg or 0.0)
    R = float(avg_score) if avg_score is not None else C
    v = float(review_count or 0)
    m = BAYESIAN_MIN_VOTES
    bayesian = (v / (v + m)) * R + (m / (v + m)) * C

    platform_rank = 1.0 / (1.0 + float(best_rank)) if best_rank else 0.0

    max_reviews = float(max_reviews or 0)
    review_count_score = math.log1p(v) / math.log1p(max_reviews) if max_reviews > 0 else 0.0

    if release_date is not None:
        days = (date.today() - release_date).days
        recency = 0.5 ** (days / RECENCY_HALFLIFE_DAYS)
    else:
        recency = 0.0

    quality_popularity = (
        0.4 * (bayesian / 5.0)
        + 0.3 * platform_rank
        + 0.2 * review_count_score
        + 0.1 * recency
    )

    return QualityScore(
        bayesian_score=bayesian,
        platform_rank_score=platform_rank,
        review_count_score=review_count_score,
        recency_score=recency,
        quality_popularity_score=quality_popularity,
    )
