from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # DB (읽기: public 읽기전용 role, 쓰기: aod_ai)
    aod_db_host: str
    aod_db_port: int = 5432
    aod_db_name: str
    aod_db_user: str
    aod_db_password: str

    # OpenAI 호환
    openai_base_url: str
    openai_api_key: str
    llm_model: str
    embedding_model: str
    embedding_dim: int = 1024

    # Vane
    vane_base_url: str = "http://localhost:3000"

    # 파이프라인 튜닝
    vane_max_sources: int = 8
    extract_max_retries: int = 3
