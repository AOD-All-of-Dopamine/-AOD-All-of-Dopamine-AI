# tests/test_embed_qwen.py
from src.embed_qwen import QUERY_PROMPT


def test_query_prompt_format():
    assert QUERY_PROMPT.startswith("Instruct: ")
    assert "\nQuery: " in QUERY_PROMPT
    assert QUERY_PROMPT.endswith("Query: ")
    assert "retrieve other Steam games" in QUERY_PROMPT
