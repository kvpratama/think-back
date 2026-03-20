"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All configuration values are loaded from environment variables via python-dotenv.
    Never hardcode secrets - always use environment variables.

    Attributes:
        supabase_url: Supabase project URL.
        supabase_key: Supabase service role key.
        openai_api_key: OpenAI API key for LLM access.
        gemini_api_key: Google Gemini API key for embeddings.
        telegram_bot_token: Telegram bot authentication token.
        llm_model: LLM model name to use for generation.
        embedding_model: Embedding model name for vector generation.
        vector_dimensions: Number of dimensions for vector embeddings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required settings (loaded from environment variables)
    supabase_url: str
    supabase_key: str
    openai_api_key: str
    gemini_api_key: str
    telegram_bot_token: str

    # Optional settings with defaults
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    llm_provider_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "gemini-embedding-001"
    vector_dimensions: int = 768


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    Returns:
        The shared Settings instance.
    """
    return Settings()
