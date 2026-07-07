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
                    {
                        "role": "system",
                        "content": (
                            "너는 한국 콘텐츠 추천을 위한 fun_tag 추출기다. "
                            "모든 텍스트 필드(normalized_summary, profile_text, evidence)는 반드시 한국어로 작성한다. "
                            "지정된 JSON 스키마 객체만 반환한다."
                        ),
                    },
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
            "규칙:\n"
            "1. fun_tags는 독자가 느끼는 '재미 요소'만 뽑는다. 아래 Genres에 이미 있는 장르명"
            "(판타지, 로판, 무협 등)이나 그 동의어를 새 태그로 제안하지 말 것. "
            "단, Active 사전에 이미 있는 태그는 재미 요소로서 근거가 있으면 사용 가능.\n"
            "2. 같은 tag를 두 번 넣지 말 것 — 가장 강한 근거 하나로 합친다.\n"
            "3. evidence는 가능한 한 Review evidence의 원문 표현을 인용한다. 리뷰가 아닌 "
            "시놉시스/홍보문구뿐이면 해당 tag_confidence를 0.5 이하로 낮춘다.\n"
            "4. Review evidence가 3건 미만이거나 내용이 빈약하면 extraction_quality를 0.5 이하로 준다.\n"
            "5. 모든 텍스트는 한국어로 작성한다.\n\n"
            f"Active fun_tag dictionary: {active_tags}\n"
            f"Title: {metadata.master_title}\n"
            f"Domain: {metadata.domain}\n"
            f"Synopsis: {metadata.synopsis}\n"
            f"Genres: {metadata.genres}\n"
            f"Review evidence:\n{evidence}\n"
        )
