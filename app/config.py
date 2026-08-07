from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_base_url: str | None = None

    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    retrieval_top_k: int = 3
    max_repair_attempts: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
