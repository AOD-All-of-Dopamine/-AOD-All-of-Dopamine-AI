from aod_ai.config import Settings


def test_settings_reads_env_keys(monkeypatch):
    monkeypatch.setenv("AOD_DB_HOST", "db.local")
    monkeypatch.setenv("AOD_DB_NAME", "aod")
    monkeypatch.setenv("AOD_DB_USER", "ai")
    monkeypatch.setenv("AOD_DB_PASSWORD", "secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("LLM_MODEL", "qwen-llm")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen-emb")
    s = Settings(_env_file=None)
    assert s.aod_db_host == "db.local"
    assert s.aod_db_port == 5432
    assert s.aod_db_name == "aod"
    assert s.embedding_dim == 1024
    assert s.vane_base_url == "http://localhost:3000"
    assert s.vane_max_sources == 8
    assert s.extract_max_retries == 3
