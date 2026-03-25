"""Tests for the Settings configuration module."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

# Shared config that suppresses .env file loading
_NO_DOTENV = SettingsConfigDict(
    env_file=None,
    env_file_encoding="utf-8",
    case_sensitive=False,
    extra="ignore",
)


def test_settings_loads_from_env() -> None:
    """Test that Settings loads values from environment variables."""
    from src.core.config import Settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
            "OPENAI_API_KEY": "sk-test123",
            "NVIDIA_API_KEY": "nvidia-test-key",
            "GEMINI_API_KEY": "gemini-test-key",
            "TELEGRAM_BOT_TOKEN": "123:ABC-DEF",
        },
        clear=True,
    ):
        with patch.object(Settings, "model_config", _NO_DOTENV):
            settings = Settings()  # type: ignore[call-arg]

    assert settings.supabase_url.get_secret_value() == "https://test.supabase.co"
    assert settings.supabase_key.get_secret_value() == "test-key"
    assert settings.openai_api_key.get_secret_value() == "sk-test123"
    assert settings.gemini_api_key.get_secret_value() == "gemini-test-key"
    assert settings.telegram_bot_token.get_secret_value() == "123:ABC-DEF"


def test_settings_requires_all_env_vars() -> None:
    """Test that Settings raises ValidationError when required env vars are missing."""
    from src.core.config import Settings

    with patch.dict(os.environ, {}, clear=True):
        with patch.object(Settings, "model_config", _NO_DOTENV):
            with pytest.raises(ValidationError):
                Settings()  # type: ignore[call-arg]


def test_settings_has_default_llm_model() -> None:
    """Test that Settings has a default LLM model."""
    from src.core.config import Settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
            "OPENAI_API_KEY": "sk-test",
            "NVIDIA_API_KEY": "nvidia-test",
            "GEMINI_API_KEY": "gemini-test",
            "TELEGRAM_BOT_TOKEN": "123:ABC",
        },
        clear=True,
    ):
        with patch.object(Settings, "model_config", _NO_DOTENV):
            settings = Settings()  # type: ignore[call-arg]

    assert settings.llm_model == "gpt-4o-mini"


def test_settings_has_default_embedding_model() -> None:
    """Test that Settings has a default embedding model."""
    from src.core.config import Settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
            "OPENAI_API_KEY": "sk-test",
            "NVIDIA_API_KEY": "nvidia-test",
            "GEMINI_API_KEY": "gemini-test",
            "TELEGRAM_BOT_TOKEN": "123:ABC",
        },
        clear=True,
    ):
        with patch.object(Settings, "model_config", _NO_DOTENV):
            settings = Settings()  # type: ignore[call-arg]

    assert settings.embedding_model == "gemini-embedding-001"


def test_vector_dimensions_is_fixed_constant() -> None:
    """Test that VECTOR_DIMENSIONS is a fixed constant, not configurable via Settings."""
    from src.core.config import VECTOR_DIMENSIONS, Settings

    assert VECTOR_DIMENSIONS == 768

    # Verify it's not a configurable field on Settings
    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
            "OPENAI_API_KEY": "sk-test",
            "NVIDIA_API_KEY": "nvidia-test",
            "GEMINI_API_KEY": "gemini-test",
            "TELEGRAM_BOT_TOKEN": "123:ABC",
        },
        clear=True,
    ):
        with patch.object(Settings, "model_config", _NO_DOTENV):
            settings = Settings()  # type: ignore[call-arg]

    assert not hasattr(settings, "vector_dimensions")
