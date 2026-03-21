"""Tests for the Settings configuration module."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that Settings loads values from environment variables."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC-DEF")

    from src.core.config import Settings

    settings = Settings(
        supabase_url="https://test.supabase.co",
        supabase_key="test-key",
        openai_api_key="sk-test123",
        gemini_api_key="gemini-test-key",
        telegram_bot_token="123:ABC-DEF",
    )

    assert settings.supabase_url == "https://test.supabase.co"
    assert settings.supabase_key == "test-key"
    assert settings.openai_api_key == "sk-test123"
    assert settings.gemini_api_key == "gemini-test-key"
    assert settings.telegram_bot_token == "123:ABC-DEF"


def test_settings_requires_all_env_vars() -> None:
    """Test that Settings raises ValidationError when required env vars are missing."""
    from src.core.config import Settings

    # Clear environment and don't load from .env
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(
            Settings,
            "model_config",
            SettingsConfigDict(
                env_file=None,  # Don't load from .env
                extra="ignore",
            ),
        ):
            with pytest.raises(ValidationError):
                Settings()  # type: ignore[call-arg]


def test_settings_has_default_llm_model() -> None:
    """Test that Settings has a default LLM model."""
    from src.core.config import Settings

    # Provide minimal required env vars
    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
            "OPENAI_API_KEY": "sk-test",
            "GEMINI_API_KEY": "gemini-test",
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "LLM_MODEL": "gpt-4o-mini",
        },
        clear=True,
    ):
        from src.core.config import Settings

        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            openai_api_key="sk-test",
            gemini_api_key="gemini-test",
            telegram_bot_token="123:ABC",
            llm_model="gpt-4o-mini",
        )
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
            "GEMINI_API_KEY": "gemini-test",
            "TELEGRAM_BOT_TOKEN": "123:ABC",
            "EMBEDDING_MODEL": "gemini-embedding-001",
        },
        clear=True,
    ):
        from src.core.config import Settings

        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            openai_api_key="sk-test",
            gemini_api_key="gemini-test",
            telegram_bot_token="123:ABC",
            embedding_model="gemini-embedding-001",
        )
        assert settings.embedding_model == "gemini-embedding-001"


def test_settings_has_default_vector_dimensions() -> None:
    """Test that Settings has default vector dimensions."""
    from src.core.config import Settings

    with patch.dict(
        os.environ,
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
            "OPENAI_API_KEY": "sk-test",
            "GEMINI_API_KEY": "gemini-test",
            "TELEGRAM_BOT_TOKEN": "123:ABC",
        },
        clear=True,
    ):
        from src.core.config import Settings

        settings = Settings(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
            openai_api_key="sk-test",
            gemini_api_key="gemini-test",
            telegram_bot_token="123:ABC",
        )
        assert settings.vector_dimensions == 768
