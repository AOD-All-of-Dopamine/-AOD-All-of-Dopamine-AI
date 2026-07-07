import json
import logging

from pgvector import Vector

from aod_ai.config import Settings

logger = logging.getLogger(__name__)

_PROFILE_SQL = """
INSERT INTO aod_ai.content_semantic_profile
  (content_id, domain, normalized_summary, profile_text, evidence,
   extraction_quality, source_count, content_hash, processed_at)
VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, now())
ON CONFLICT (content_id) DO UPDATE SET
  domain = EXCLUDED.domain,
  normalized_summary = EXCLUDED.normalized_summary,
  profile_text = EXCLUDED.profile_text,
  evidence = EXCLUDED.evidence,
  extraction_quality = EXCLUDED.extraction_quality,
  source_count = EXCLUDED.source_count,
  content_hash = EXCLUDED.content_hash,
  processed_at = now()
"""
_DICT_SQL = """
INSERT INTO aod_ai.fun_tag_dict (name, status)
VALUES (%s, 'proposed')
ON CONFLICT (name) DO NOTHING
"""
_FUN_TAG_SQL = """
INSERT INTO aod_ai.content_fun_tag (content_id, tag, tag_score, tag_confidence)
VALUES (%s, %s, %s, %s)
ON CONFLICT (content_id, tag) DO UPDATE SET
  tag_score = EXCLUDED.tag_score,
  tag_confidence = EXCLUDED.tag_confidence
"""
_EMB_SQL = """
INSERT INTO aod_ai.content_embedding (content_id, embedding, model, dim)
VALUES (%s, %s, %s, %s)
ON CONFLICT (content_id) DO UPDATE SET
  embedding = EXCLUDED.embedding, model = EXCLUDED.model, dim = EXCLUDED.dim
"""
_QUALITY_SQL = """
INSERT INTO aod_ai.content_quality_score
  (content_id, bayesian_score, platform_rank_score, review_count_score,
   recency_score, quality_popularity_score, computed_at)
VALUES (%s, %s, %s, %s, %s, %s, now())
ON CONFLICT (content_id) DO UPDATE SET
  bayesian_score = EXCLUDED.bayesian_score,
  platform_rank_score = EXCLUDED.platform_rank_score,
  review_count_score = EXCLUDED.review_count_score,
  recency_score = EXCLUDED.recency_score,
  quality_popularity_score = EXCLUDED.quality_popularity_score,
  computed_at = now()
"""
_ACTIVE_NAMES_SQL = "SELECT name FROM aod_ai.fun_tag_dict WHERE status = 'active'"


def upsert_assets(conn, target, extraction, vector, quality, sources) -> None:
    evidence = json.dumps(
        [{"content": s.content, "url": s.url} for s in sources], ensure_ascii=False
    )
    model = Settings().embedding_model
    with conn.transaction():
        with conn.cursor() as cur:
            # 계약 §9 [정규화 태그 저장]: is_new=false 태그는 raw tag가 아니라
            # active dict의 canonical name으로 치환해 저장한다 (strip/lower 매칭).
            cur.execute(_ACTIVE_NAMES_SQL)
            canonical = {name.strip().lower(): name for (name,) in cur.fetchall()}

            cur.execute(
                _PROFILE_SQL,
                (
                    target.content_id, target.domain, extraction.normalized_summary,
                    extraction.profile_text, evidence, extraction.extraction_quality,
                    len(sources), target.content_hash,
                ),
            )
            cur.execute(
                "DELETE FROM aod_ai.content_fun_tag WHERE content_id = %s",
                (target.content_id,),
            )
            for item in extraction.fun_tags:
                if item.is_new:
                    cur.execute(_DICT_SQL, (item.tag,))
                else:
                    tag = canonical.get(item.tag.strip().lower())
                    if tag is None:
                        logger.warning(
                            "active dict 매칭 실패, raw tag 유지: %r (content_id=%s)",
                            item.tag, target.content_id,
                        )
                        tag = item.tag
                    cur.execute(
                        _FUN_TAG_SQL,
                        (target.content_id, tag, item.tag_score, item.tag_confidence),
                    )
            cur.execute(
                _EMB_SQL,
                (target.content_id, Vector(vector), model, len(vector)),  # 명시 Vector 래핑
            )
            cur.execute(
                _QUALITY_SQL,
                (
                    target.content_id, quality.bayesian_score, quality.platform_rank_score,
                    quality.review_count_score, quality.recency_score,
                    quality.quality_popularity_score,
                ),
            )
