from __future__ import annotations

import hashlib

from pydantic import BaseModel


class FunTagItem(BaseModel):
    tag: str
    tag_score: float
    tag_confidence: float
    evidence: str
    is_new: bool


class Extraction(BaseModel):
    fun_tags: list[FunTagItem]
    normalized_summary: str
    profile_text: str
    extraction_quality: float


class ReviewSource(BaseModel):
    content: str
    url: str


class SelectedTarget(BaseModel):
    content_id: int
    domain: str
    master_title: str
    original_title: str | None
    synopsis: str | None
    genres: list[str]
    content_hash: str


class QualityScore(BaseModel):
    bayesian_score: float
    platform_rank_score: float
    review_count_score: float
    recency_score: float
    quality_popularity_score: float


def content_hash(
    master_title: str,
    original_title: str | None,
    synopsis: str | None,
    genres: list[str],
) -> str:
    raw = f"{master_title}|{original_title}|{synopsis}|{sorted(genres)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
