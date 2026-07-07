from __future__ import annotations

import json
import os

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from aod_ai.models import Extraction, ReviewSource, SelectedTarget

_SCHEMA_HINT = json.dumps(
    {
        "fun_tags": [
            {
                "tag": "str",
                "tag_score": 0.0,
                "tag_confidence": 0.0,
                "evidence": "str",
                "is_new": False,
            }
        ],
        "normalized_summary": "str",
        "profile_text": "str",
        "extraction_quality": 0.0,
    },
    ensure_ascii=False,
)


class LlmClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def extract(
        self,
        *,
        metadata: SelectedTarget,
        sources: list[ReviewSource],
        active_tags: list[str],
    ) -> Extraction:
        prompt = self._build_prompt(metadata, sources, active_tags)
        max_retries = int(os.getenv("EXTRACT_MAX_RETRIES", "3"))  # config §5

        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, max=10),
            reraise=True,
        )
        def _call() -> Extraction:
            resp = self._client.chat.completions.create(
                model=self._model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You extract structured fun_tag JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            return Extraction.model_validate_json(resp.choices[0].message.content)

        return _call()

    def _build_prompt(
        self,
        metadata: SelectedTarget,
        sources: list[ReviewSource],
        active_tags: list[str],
    ) -> str:
        evidence = "\n".join(f"- {s.content} ({s.url})" for s in sources)
        return (
            "Return only a JSON object matching this schema (respond in json):\n"
            f"{_SCHEMA_HINT}\n\n"
            f"Active fun_tag dictionary: {active_tags}\n"
            f"Title: {metadata.master_title}\n"
            f"Domain: {metadata.domain}\n"
            f"Synopsis: {metadata.synopsis}\n"
            f"Genres: {metadata.genres}\n"
            f"Review evidence:\n{evidence}\n"
        )
