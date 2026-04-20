"""Application configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# COUPLING CONSTRAINT: This value must match the output dimensions of the
# configured embedding_model (Settings.embedding_model) AND the vector column
# size in the database schema (supabase/schema.sql). Changing one without
# updating the others will cause dimension mismatch errors.
VECTOR_DIMENSIONS = 768


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
        llm_provider: LLM provider name (e.g., 'openai').
        llm_provider_base_url: Base URL for the LLM provider API.
        embedding_model: Embedding model name for vector generation.
        search_top_k: Number of top results to return from search.
        webhook_url: Public HTTPS URL to enable webhook mode (empty for polling).
        webhook_secret: Secret token for Telegram webhook request verification.
        port: Port for the webhook server.
        eval_llm_model: LLM model name for evaluation tasks.
        eval_llm_provider: LLM provider name for evaluation tasks.
        eval_llm_provider_base_url: Base URL for the evaluation LLM provider API.
        eval_llm_api_key: API key for the evaluation LLM provider.
        eval_jury_judges: Configuration for evaluation jury judges.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required settings (loaded from environment variables)
    supabase_url: SecretStr
    supabase_key: SecretStr
    openai_api_key: SecretStr
    gemini_api_key: SecretStr
    telegram_bot_token: SecretStr

    # Optional settings with defaults
    llm_model: str = "gpt-4o-mini"
    llm_provider: str = "openai"
    llm_provider_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "gemini-embedding-001"
    search_top_k: int = 3

    # Webhook (set WEBHOOK_URL to enable webhook mode; leave empty for polling)
    webhook_url: str = ""

    @field_validator("webhook_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        """Remove trailing slashes to avoid double-slash in URL construction."""
        return v.rstrip("/")

    webhook_secret: SecretStr = SecretStr("")
    port: int = 8000

    # LLM for evaluation
    eval_llm_model: str = "gpt-4o"
    eval_llm_provider: str = "openai"
    eval_llm_provider_base_url: str = "https://api.openai.com/v1"
    eval_llm_api_key: SecretStr = SecretStr("")
    eval_jury_judges: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton.

    Returns:
        The shared Settings instance.
    """
    return Settings()  # type: ignore[call-arg]
